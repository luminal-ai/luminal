//! THE ARENA: one runtime-owned slab of device bytes, and the
//! DEVICE-FREE pass that decides who lives where in it.
//!
//! # Why this module exists (#422, superseding #401)
//!
//! CL-2's executor materialized EVERY plan buffer up front — one
//! `alloc_zeros` per `BufferId`, all of them live for the whole call.
//! That is the sum of every buffer the plan ever names, whether or not
//! two of them are ever live at the same instant. The bufferizer
//! already computes the lifetimes (`BufferAlloc` brings storage into
//! existence, `BufferFree` ends it, and the containment certificate
//! guarantees every toucher sits between the two); nothing consumed
//! them. This pass does.
//!
//! Austin's division of labour (2026-09-03): "first produce the
//! bufferizer, this will do the allocations / frees. Then a separate
//! thing will map those allocation / frees to slices on memory in the
//! arena allocator." This IS the separate thing. It reads a bufferized
//! plan and answers two questions:
//!
//!  1. **In what order should the runtime issue the plan's nodes?**
//!     Any topological order is legal (the plan supplies dependency
//!     structure only — see the BufferCopy contract); the order chosen
//!     here is LIVENESS-AWARE, because the order is what sets the
//!     high-water mark.
//!  2. **Which slab range backs each buffer, over that order?** A
//!     first-fit free-list walk of the alloc/free events.
//!
//! # The three ownership rows (`Owner` × `FreedBy`), and why only one is in the slab
//!
//! | row | `Owner` | `FreedBy` | alloc? | free? | arena treatment |
//! |---|---|---|---|---|---|
//! | BOUNDARY | `Caller` | `Caller` | no | no | `standalone` |
//! | DONATED | `Caller` | `Program` | no | yes | `donated` |
//! | ESCAPING | `System` | `Caller` | yes | no | `standalone` |
//! | INTERIOR | `System` | `Program` | yes | yes | **slab member** |
//!
//! Only the INTERIOR row has BOTH ends of a lifetime inside the
//! program, which is exactly the precondition for handing its bytes to
//! a later buffer. An ESCAPING buffer's bytes are the caller's from
//! return on — recycling them would hand the caller a range the next
//! call overwrites. A DONATED buffer's storage came from the caller;
//! the program's free RELEASES it, it does not license the arena to
//! re-let it. BOUNDARY storage is never the program's at all.
//!
//! # Sizing
//!
//! `bytes_of` is the caller's, so tests can plan with mock sizes and
//! the executor can pass the one real rule (`literal_span_elements() *
//! dtype_bytes(dtype)` — see `crate::device`). Symbolic extents are the
//! caller's error to raise. Buckets are always concrete (D7), so the
//! slab is always a number.
//!
//! # What is NOT here
//!
//! No device types, no cudarc: this file compiles and its tests run on
//! a laptop. Search-time slab policy (sizing the slab across the
//! candidate plans a search evaluates) is Phase 4's; this pass sizes
//! ONE installed plan.

use anyhow::{bail, Result};
use luminal::bufferize::{Buffer, BufferId, BufferIrGraph, BufferNode, Owner, PlanLayout};
use luminal::layout_ir::FreedBy;
use luminal::prelude::{petgraph, FxHashMap, NodeIndex};
use petgraph::visit::{EdgeRef, NodeIndexable};
use std::collections::{BTreeMap, BinaryHeap};

/// Device allocations are 256-byte aligned, so every slab range is too:
/// a sub-range handed to a kernel must satisfy the same alignment the
/// driver would have given it for its own allocation (vectorized loads
/// and cuBLASLt's `ld` arithmetic both assume it).
pub const ARENA_ALIGN: usize = 256;

fn align_up(bytes: usize) -> usize {
    bytes.div_ceil(ARENA_ALIGN) * ARENA_ALIGN
}

/// One buffer's home in the slab. `bytes` is the buffer's TRUE size
/// (what a memcpy of it moves); the range RESERVED is `align_up(bytes)`,
/// which is what disjointness is checked over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaSlice {
    pub offset: usize,
    pub bytes: usize,
}

impl ArenaSlice {
    /// The reserved (aligned) extent — `bytes` rounded up to
    /// [`ARENA_ALIGN`].
    pub fn reserved(&self) -> usize {
        align_up(self.bytes)
    }
}

/// The answer: an issue order, a slab size, and who sits where.
#[derive(Debug, Clone, Default)]
pub struct ArenaPlan {
    /// The order the runtime issues plan nodes in — a topological order
    /// of the dag (Data and Anti edges alike), chosen for a small
    /// high-water mark. Every node of the dag appears exactly once.
    pub order: Vec<NodeIndex>,
    /// The high-water mark: how many bytes the slab must hold for this
    /// plan under this order.
    pub slab_bytes: usize,
    /// Slab members (the INTERIOR row) and their ranges.
    pub slices: FxHashMap<BufferId, ArenaSlice>,
    /// Buffers that live OUTSIDE the slab in their own allocations: the
    /// BOUNDARY row (inputs and caller-bound outputs) and the ESCAPING
    /// row (minted storage handed to the caller).
    pub standalone: Vec<BufferId>,
    /// The DONATED row: caller storage the program frees. Its bytes are
    /// the caller's staged storage; it gets no slab range, and its free
    /// releases rather than recycles.
    pub donated: Vec<BufferId>,
}

/// Which of the four rows a buffer is in, as the arena treats it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Row {
    Slab,
    Standalone,
    Donated,
}

/// Node kinds, for the order policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Free,
    Ordinary,
    Alloc,
}

fn kind_of<L: PlanLayout>(node: &BufferNode<L>) -> Kind {
    match node {
        BufferNode::Compute { op, .. } => match op.label() {
            "BufferAlloc" => Kind::Alloc,
            "BufferFree" => Kind::Free,
            _ => Kind::Ordinary,
        },
        _ => Kind::Ordinary,
    }
}

/// The buffer a `BufferAlloc` brings into existence (its single result).
fn allocated<L: PlanLayout>(node: &BufferNode<L>) -> Option<&BufferId> {
    match node {
        BufferNode::Compute { op, writes, .. } if op.label() == "BufferAlloc" => writes.first(),
        _ => None,
    }
}

/// The buffer a `BufferFree` ends (its single operand).
fn freed<L: PlanLayout>(node: &BufferNode<L>) -> Option<&BufferId> {
    match node {
        BufferNode::Compute { op, reads, .. } if op.label() == "BufferFree" => reads.first(),
        _ => None,
    }
}

/// THE ISSUE ORDER — a topological order chosen for a small high-water
/// mark.
///
/// A raw `petgraph::algo::toposort` is a legal order and a terrible
/// one: `BufferAlloc` nodes have in-degree zero (they consume nothing),
/// so Kahn's queue hoists EVERY alloc to the front, every buffer is
/// live from the first instant, and the high-water mark equals the sum
/// of all of them — the very number this pass exists to beat (verdict
/// C7 of the #420/#422 soundness review).
///
/// The fix is a priority among the READY nodes, not a different
/// algorithm:
///
///  1. a `BufferFree` whose in-edges are all discharged goes FIRST —
///     its range returns to the free list at the earliest instant the
///     dependency structure allows;
///  2. then any ordinary node (compute, copy, boundary);
///  3. a `BufferAlloc` goes LAST, only when nothing else can run — so
///     a buffer is born at the latest instant, immediately before the
///     first node that needed it to exist.
///
/// Ties inside a class break on node index, which is the bufferizer's
/// own emission order ("synthesized allocs slot in before their
/// buffer's first toucher, frees after its last toucher"), so among
/// equally-ready allocs we mint the one the planner placed first.
fn issue_order<L: PlanLayout>(plan: &BufferIrGraph<L>) -> Result<Vec<NodeIndex>> {
    let mut indegree: Vec<usize> = vec![0; plan.dag.node_bound()];
    for index in plan.dag.node_indices() {
        indegree[index.index()] = plan
            .dag
            .edges_directed(index, petgraph::Direction::Incoming)
            .count();
    }
    // Three ready queues, each min-ordered by node index (`Reverse`).
    let mut frees: BinaryHeap<std::cmp::Reverse<usize>> = BinaryHeap::new();
    let mut ordinary: BinaryHeap<std::cmp::Reverse<usize>> = BinaryHeap::new();
    let mut allocs: BinaryHeap<std::cmp::Reverse<usize>> = BinaryHeap::new();
    let push = |index: NodeIndex,
                frees: &mut BinaryHeap<std::cmp::Reverse<usize>>,
                ordinary: &mut BinaryHeap<std::cmp::Reverse<usize>>,
                allocs: &mut BinaryHeap<std::cmp::Reverse<usize>>| {
        match kind_of(&plan.dag[index]) {
            Kind::Free => frees.push(std::cmp::Reverse(index.index())),
            Kind::Ordinary => ordinary.push(std::cmp::Reverse(index.index())),
            Kind::Alloc => allocs.push(std::cmp::Reverse(index.index())),
        }
    };
    for index in plan.dag.node_indices() {
        if indegree[index.index()] == 0 {
            push(index, &mut frees, &mut ordinary, &mut allocs);
        }
    }
    let mut order = Vec::with_capacity(plan.dag.node_count());
    loop {
        let next = frees
            .pop()
            .or_else(|| ordinary.pop())
            .or_else(|| allocs.pop());
        let Some(std::cmp::Reverse(raw)) = next else {
            break;
        };
        let index = NodeIndex::new(raw);
        order.push(index);
        for edge in plan
            .dag
            .edges_directed(index, petgraph::Direction::Outgoing)
        {
            let target = edge.target();
            indegree[target.index()] -= 1;
            if indegree[target.index()] == 0 {
                push(target, &mut frees, &mut ordinary, &mut allocs);
            }
        }
    }
    if order.len() != plan.dag.node_count() {
        bail!("plan dag has a cycle");
    }
    Ok(order)
}

/// The free list: holes strictly below `top`, plus the wilderness above
/// it. First fit in offset order (cheap, and it keeps low addresses
/// busy so the tail stays coalesced).
#[derive(Debug, Default)]
struct FreeList {
    /// offset -> length, disjoint and never adjacent (always coalesced).
    holes: BTreeMap<usize, usize>,
    /// The high-water mark: everything at or above this is virgin.
    top: usize,
}

impl FreeList {
    fn alloc(&mut self, need: usize) -> usize {
        if let Some((&offset, &len)) = self.holes.iter().find(|(_, &len)| len >= need) {
            self.holes.remove(&offset);
            if len > need {
                self.holes.insert(offset + need, len - need);
            }
            return offset;
        }
        // No hole fits. If the LAST hole runs right up to the top, grow
        // through it instead of stranding it (coalescing with the
        // wilderness — the classic dlmalloc move).
        if let Some((&offset, &len)) = self.holes.iter().next_back() {
            if offset + len == self.top {
                self.holes.remove(&offset);
                self.top = offset + need;
                return offset;
            }
        }
        let offset = self.top;
        self.top += need;
        offset
    }

    fn free(&mut self, offset: usize, len: usize) {
        let mut offset = offset;
        let mut len = len;
        // Coalesce with the predecessor hole, if it ends here.
        if let Some((&prev, &prev_len)) = self.holes.range(..offset).next_back() {
            if prev + prev_len == offset {
                self.holes.remove(&prev);
                offset = prev;
                len += prev_len;
            }
        }
        // …and with the successor, if it starts where we end.
        if let Some((&next, &next_len)) = self.holes.range(offset + len..).next() {
            if next == offset + len {
                self.holes.remove(&next);
                len += next_len;
            }
        }
        self.holes.insert(offset, len);
    }
}

/// Plan the arena for one bufferized plan.
///
/// `bytes_of` sizes a buffer (the executor passes the span-of-layout
/// rule; tests pass whatever they like). It is called for SLAB MEMBERS
/// only — standalone and donated buffers are the executor's to size, in
/// its own allocation phase, exactly as before.
pub fn plan_arena<L: PlanLayout>(
    plan: &BufferIrGraph<L>,
    bytes_of: impl Fn(&Buffer<L>) -> Result<usize>,
) -> Result<ArenaPlan> {
    plan_arena_over(plan, bytes_of, issue_order(plan)?)
}

/// [`plan_arena`] over a CALLER-SUPPLIED issue order — the seam the
/// order-policy comparison test uses to price one order against
/// another. The order must be a topological order of `plan.dag`
/// covering every node; nothing here re-checks that.
pub(crate) fn plan_arena_over<L: PlanLayout>(
    plan: &BufferIrGraph<L>,
    bytes_of: impl Fn(&Buffer<L>) -> Result<usize>,
    order: Vec<NodeIndex>,
) -> Result<ArenaPlan> {
    // ---- classification: the ownership rows -------------------------
    //
    // A slab member needs BOTH ends of its lifetime in the program. The
    // rows say which buffers those are; the dag says whether the nodes
    // that mark the ends are actually there. A plan built by hand (or
    // an older plan loaded from disk) may carry an INTERIOR buffer with
    // no alloc/free pair — bufferize's `optimize` always mints them,
    // but nothing here re-derives them. Such a buffer is DEMOTED to
    // standalone: it gets its own allocation for the whole call, which
    // is precisely CL-2's pre-arena behaviour, and the plan still runs.
    let mut alloc_node: FxHashMap<BufferId, NodeIndex> = FxHashMap::default();
    let mut free_node: FxHashMap<BufferId, NodeIndex> = FxHashMap::default();
    for index in plan.dag.node_indices() {
        if let Some(buffer) = allocated(&plan.dag[index]) {
            if alloc_node.insert(buffer.clone(), index).is_some() {
                bail!(
                    "arena: buffer {buffer:?} is allocated twice — one \
                     BufferAlloc per buffer is the plan's invariant, and a \
                     second one would re-let a live range"
                );
            }
        }
        if let Some(buffer) = freed(&plan.dag[index]) {
            if free_node.insert(buffer.clone(), index).is_some() {
                bail!(
                    "arena: buffer {buffer:?} is freed twice — one BufferFree \
                     per buffer is the plan's invariant, and a second one \
                     would hand a live range to the next allocation"
                );
            }
        }
    }

    let mut rows: Vec<(BufferId, Row)> = Vec::with_capacity(plan.buffers.len());
    for (id, buffer) in &plan.buffers {
        let row = match (buffer.owner, buffer.freed_by) {
            // INTERIOR — the only recyclable row, and only with both
            // lifetime ends present in the dag.
            (Owner::System, FreedBy::Program)
                if alloc_node.contains_key(id) && free_node.contains_key(id) =>
            {
                Row::Slab
            }
            (Owner::System, FreedBy::Program) => Row::Standalone,
            // DONATED — caller storage the program frees.
            (Owner::Caller, FreedBy::Program) => Row::Donated,
            // BOUNDARY and ESCAPING — the caller's bytes after the call.
            (_, FreedBy::Caller) => Row::Standalone,
        };
        rows.push((id.clone(), row));
    }
    // `plan.buffers` is a hash map; sort so the reported vectors read
    // the same on every run (the slab LAYOUT is already deterministic —
    // it follows `order`, not this iteration).
    rows.sort_by_key(|(id, _)| format!("{id:?}"));

    let mut arena = ArenaPlan {
        order,
        ..Default::default()
    };
    let mut sizes: FxHashMap<BufferId, usize> = FxHashMap::default();
    for (id, row) in &rows {
        match row {
            Row::Slab => {
                let buffer = &plan.buffers[id];
                sizes.insert(id.clone(), bytes_of(buffer)?);
            }
            Row::Standalone => arena.standalone.push(id.clone()),
            Row::Donated => arena.donated.push(id.clone()),
        }
    }

    // ---- the walk: first-fit over the issue order -------------------
    let mut free_list = FreeList::default();
    // CONTRACT-1, LIVE-RANGE FORM. The whole-plan disjointness assert
    // the executor used to run at bind time is vacuous under a slab
    // (every range is a sub-range of one allocation) — and it was never
    // the right question anyway: what folded-view reads and WAR
    // ordering need is that two SIMULTANEOUSLY BOUND BufferIds do not
    // share a byte. That is a property of THIS walk, so it is checked
    // here, once per allocation, against the live set's neighbours (a
    // sorted disjoint set stays disjoint iff each insertion clears its
    // two neighbours). The executor keeps `binding_check::assert_disjoint`
    // for the allocations it makes itself.
    let mut live: BTreeMap<usize, (usize, BufferId)> = BTreeMap::new();
    for &index in &arena.order {
        if let Some(buffer) = allocated(&plan.dag[index]) {
            let Some(&bytes) = sizes.get(buffer) else {
                continue; // not a slab member (demoted, escaping, …)
            };
            let need = align_up(bytes.max(1));
            let offset = free_list.alloc(need);
            if let Some((&prev, (prev_len, prev_id))) = live.range(..offset).next_back() {
                if prev + prev_len > offset {
                    bail!(
                        "CONTRACT-1 violation (arena): {prev_id:?} holds \
                         [{prev}, {}) and {buffer:?} was given [{offset}, {}) \
                         — simultaneously bound BufferIds must be disjoint",
                        prev + prev_len,
                        offset + need
                    );
                }
            }
            if let Some((&next, (_, next_id))) = live.range(offset..).next() {
                if offset + need > next {
                    bail!(
                        "CONTRACT-1 violation (arena): {buffer:?} was given \
                         [{offset}, {}) and {next_id:?} holds [{next}, …) — \
                         simultaneously bound BufferIds must be disjoint",
                        offset + need
                    );
                }
            }
            live.insert(offset, (need, buffer.clone()));
            arena
                .slices
                .insert(buffer.clone(), ArenaSlice { offset, bytes });
        }
        if let Some(buffer) = freed(&plan.dag[index]) {
            if let Some(slice) = arena.slices.get(buffer) {
                let need = align_up(slice.bytes.max(1));
                live.remove(&slice.offset);
                free_list.free(slice.offset, need);
            }
        }
    }
    arena.slab_bytes = free_list.top;
    Ok(arena)
}

#[cfg(test)]
mod tests {
    use super::*;
    use luminal::index_expr::IotaExpr;
    use luminal::layout_ir::Access;
    use luminal::test_support::{bufferize_mock, MockLayout, MockOp, MockViewWithMap, TestGraph};

    /// Every buffer is the same size, so the numbers below are counts of
    /// buffers and nothing else.
    const UNIT: usize = 1000;
    const RESERVED: usize = 1024; // align_up(1000)

    fn unit_bytes(_buffer: &Buffer<MockLayout>) -> Result<usize> {
        Ok(UNIT)
    }

    /// A straight chain: input `x`, then `steps` out-of-place reads, the
    /// last pinned to an output slot. Every intermediate result gets its
    /// own System buffer with a synthesized alloc/free pair, and no two
    /// non-adjacent results are ever live together.
    fn chain(steps: usize) -> luminal::bufferize::BufferIrGraph<MockLayout> {
        let mut g = TestGraph::new();
        let mut value = g.input("x", "xb", Access::ReadOnly, "rm");
        for step in 0..steps {
            value = g.op(
                Box::new(MockOp {
                    reads: vec![true],
                    ..Default::default()
                }),
                &[&value],
                &[(&format!("v{step}"), "rm")],
            )[0]
            .clone();
        }
        g.output(&value, "out");
        bufferize_mock(&g.build()).expect("chain bufferizes")
    }

    fn positions(arena: &ArenaPlan) -> FxHashMap<NodeIndex, usize> {
        arena
            .order
            .iter()
            .enumerate()
            .map(|(at, &index)| (index, at))
            .collect()
    }

    /// Every node that READS OR WRITES `buffer` for real — allocs and
    /// frees excluded (they mark the lifetime, they do not touch bytes).
    fn touchers<L: PlanLayout>(
        plan: &luminal::bufferize::BufferIrGraph<L>,
        buffer: &BufferId,
    ) -> Vec<NodeIndex> {
        plan.dag
            .node_indices()
            .filter(|&index| match &plan.dag[index] {
                BufferNode::Compute {
                    op, reads, writes, ..
                } => {
                    !matches!(op.label(), "BufferAlloc" | "BufferFree")
                        && (reads.contains(buffer) || writes.contains(buffer))
                }
                BufferNode::BufferCopy { src, dst } => src == buffer || dst == buffer,
                BufferNode::BufferInput { slots } => slots.iter().any(|s| &s.buffer == buffer),
                BufferNode::BufferOutput { slots } => slots.iter().any(|s| &s.buffer == buffer),
            })
            .collect()
    }

    /// t1 — RECYCLING: a chain of interior buffers costs the PEAK, not
    /// the SUM. Only producer and consumer are ever live together, so
    /// however long the chain, two ranges suffice.
    #[test]
    fn chain_of_interior_buffers_costs_the_peak_not_the_sum() {
        let plan = chain(5);
        let arena = plan_arena(&plan, unit_bytes).expect("arena plans");
        let members = arena.slices.len();
        assert!(
            members >= 3,
            "want at least three interior buffers to recycle, got {members}:\n{}",
            plan.summary()
        );
        let sum: usize = arena.slices.values().map(|s| align_up(s.bytes)).sum();
        assert_eq!(
            arena.slab_bytes,
            2 * RESERVED,
            "producer + consumer are the only pair ever live together \
             ({members} members, sum {sum}):\n{}",
            plan.summary()
        );
        assert!(
            arena.slab_bytes < sum,
            "peak {} must be under the sum {sum}",
            arena.slab_bytes
        );
        // …and the order policy is why. The bufferizer's own node-index
        // order is a legal topological order here; a raw `toposort`
        // hoists every in-degree-zero alloc to the front (verdict C7).
        let by_index = plan_arena_over(
            &plan,
            unit_bytes,
            plan.dag.node_indices().collect::<Vec<_>>(),
        )
        .expect("node-index order plans");
        let raw = plan_arena_over(
            &plan,
            unit_bytes,
            petgraph::algo::toposort(&plan.dag, None).expect("acyclic"),
        )
        .expect("raw toposort plans");
        println!(
            "high-water: liveness-aware {} | bufferizer node index {} | raw toposort {} \
             | sum-of-members {sum}",
            arena.slab_bytes, by_index.slab_bytes, raw.slab_bytes
        );
        assert!(
            arena.slab_bytes <= by_index.slab_bytes,
            "the liveness-aware order is never worse than emission order"
        );
        assert_eq!(
            raw.slab_bytes, sum,
            "a raw toposort really does pay the sum"
        );
    }

    /// t2 — ESCAPING (`Owner::System` + `FreedBy::Caller`): minted
    /// storage the caller receives. It has an alloc and NO free, so its
    /// bytes must outlive the call: outside the slab.
    #[test]
    fn escaping_minted_storage_stays_out_of_the_slab() {
        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        let p = g.op(
            Box::new(MockOp {
                reads: vec![true],
                ..Default::default()
            }),
            &[&x],
            &[("p", "rm")],
        )[0]
        .clone();
        let v = g.op(
            Box::new(MockViewWithMap {
                entries: vec![IotaExpr::Coord(0), IotaExpr::Coord(1)],
            }),
            &[&p],
            &[("v", "t")],
        )[0]
        .clone();
        g.output(&v, "E");
        let plan = bufferize_mock(&g.build()).expect("escape bufferizes");
        let escaping = plan
            .buffers
            .iter()
            .find(|(_, b)| b.owner == Owner::System && b.freed_by == FreedBy::Caller)
            .map(|(id, _)| id.clone())
            .unwrap_or_else(|| panic!("no escaping buffer:\n{}", plan.summary()));
        let arena = plan_arena(&plan, unit_bytes).expect("arena plans");
        assert!(
            !arena.slices.contains_key(&escaping),
            "escaping storage must not be a slab member:\n{}",
            plan.summary()
        );
        assert!(
            arena.standalone.contains(&escaping),
            "escaping storage is standalone:\n{}",
            plan.summary()
        );
    }

    /// t3 — DONATED (`Owner::Caller` + `FreedBy::Program`): the caller's
    /// bytes, released by the program. No slab range; the free lands
    /// after every toucher.
    #[test]
    fn donated_caller_storage_gets_no_slab_range_and_frees_last() {
        let mut g = TestGraph::new();
        let x = g.input_binding(
            "x",
            "xb",
            Some(Access::ReadWrite),
            Some(FreedBy::Program),
            "rm",
        );
        let y = g.op(
            Box::new(MockOp {
                reads: vec![true],
                ..Default::default()
            }),
            &[&x],
            &[("y", "rm")],
        )[0]
        .clone();
        g.output(&y, "out");
        let plan = bufferize_mock(&g.build()).expect("donation bufferizes");
        let donated = plan
            .buffers
            .iter()
            .find(|(_, b)| b.owner == Owner::Caller && b.freed_by == FreedBy::Program)
            .map(|(id, _)| id.clone())
            .unwrap_or_else(|| panic!("no donated buffer:\n{}", plan.summary()));
        let arena = plan_arena(&plan, unit_bytes).expect("arena plans");
        assert!(
            arena.donated.contains(&donated),
            "donated storage is its own row:\n{}",
            plan.summary()
        );
        assert!(
            !arena.slices.contains_key(&donated),
            "donated storage gets no slab range:\n{}",
            plan.summary()
        );
        let at = positions(&arena);
        let free = plan
            .dag
            .node_indices()
            .find(|&i| freed(&plan.dag[i]) == Some(&donated))
            .unwrap_or_else(|| panic!("donated storage is freed:\n{}", plan.summary()));
        for toucher in touchers(&plan, &donated) {
            assert!(
                at[&toucher] < at[&free],
                "the free must follow every toucher of the donated buffer:\n{}",
                plan.summary()
            );
        }
    }

    /// t4 — THE RECYCLING CONTRACT: a range is only re-let after its
    /// previous occupant is done with it. Every toucher of the old
    /// occupant precedes every toucher of the new one in the issue
    /// order — which, on one stream, is execution order.
    #[test]
    fn a_recycled_range_is_only_re_let_after_its_occupant_is_finished() {
        let plan = chain(5);
        let arena = plan_arena(&plan, unit_bytes).expect("arena plans");
        let at = positions(&arena);
        let mut sharing = 0usize;
        let members: Vec<(&BufferId, &ArenaSlice)> = arena.slices.iter().collect();
        for (a, sa) in &members {
            for (b, sb) in &members {
                if a == b || sa.offset != sb.offset {
                    continue;
                }
                // Same range, two buffers: order them by their allocs.
                let alloc_a = plan
                    .dag
                    .node_indices()
                    .find(|&i| allocated(&plan.dag[i]) == Some(a))
                    .expect("slab member has an alloc");
                let alloc_b = plan
                    .dag
                    .node_indices()
                    .find(|&i| allocated(&plan.dag[i]) == Some(b))
                    .expect("slab member has an alloc");
                if at[&alloc_a] > at[&alloc_b] {
                    continue; // handled from the other side
                }
                sharing += 1;
                let last_old = touchers(&plan, a)
                    .into_iter()
                    .map(|n| at[&n])
                    .max()
                    .expect("an occupant is touched");
                let first_new = touchers(&plan, b)
                    .into_iter()
                    .map(|n| at[&n])
                    .min()
                    .expect("an occupant is touched");
                assert!(
                    last_old < first_new,
                    "range {} was re-let to {b:?} at position {first_new} while \
                     {a:?} was still touching it at {last_old}:\n{}",
                    sa.offset,
                    plan.summary()
                );
            }
        }
        assert!(
            sharing > 0,
            "the chain must actually recycle a range:\n{}",
            plan.summary()
        );
    }

    /// t5 — the issue order is a real topological order: every edge,
    /// Data and Anti alike, points forward in it.
    #[test]
    fn the_issue_order_respects_every_data_and_anti_edge() {
        let plan = chain(4);
        let arena = plan_arena(&plan, unit_bytes).expect("arena plans");
        assert_eq!(
            arena.order.len(),
            plan.dag.node_count(),
            "every node is issued exactly once"
        );
        let at = positions(&arena);
        let mut anti = 0usize;
        for edge in plan.dag.edge_references() {
            if edge.weight().kind == luminal::bufferize::EdgeKind::Anti {
                anti += 1;
            }
            assert!(
                at[&edge.source()] < at[&edge.target()],
                "edge {:?} -> {:?} ({:?}) points backwards in the issue order:\n{}",
                edge.source(),
                edge.target(),
                edge.weight().kind,
                plan.summary()
            );
        }
        println!("{} anti edges honoured", anti);
    }
}
