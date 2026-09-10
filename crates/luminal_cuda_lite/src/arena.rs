//! Physical storage planning for graph execution. One schedule places uploads,
//! bufferized nodes, and readbacks; one interval allocator packs device tensors,
//! operation scratch, parameters, and pinned staging. Logical ownership remains
//! in the bufferized plan: the arena holds private device copies, and returned
//! outputs own their host bytes.

use anyhow::{Result, anyhow, bail, ensure};
use luminal::bufferize::{Buffer, BufferId, BufferIrGraph, BufferNode, Owner, PlanLayout};
use luminal::layout_ir::FreedBy;
use luminal::prelude::{FxHashMap, NodeIndex, petgraph};
use petgraph::visit::{EdgeRef, NodeIndexable};
use std::collections::{BTreeMap, BinaryHeap};

/// THE ONE SIZING RULE: the buffer backs one tensor, its carried layout
/// gives the span in elements and the dtype fact gives the byte width
/// (ALLOCATION BY ASSIGNMENT LOOKUP, corrected contract 2026-08-31 — no
/// walk, no voting).
///
/// IT LIVES HERE, not in `crate::device`, because it is DEVICE-FREE and
/// two callers need it: the executor's `execute_plan` (which is behind
/// the `device` feature) and the search's finalist arena planning
/// (Phase 5, which runs on any host so a budget can be checked without a
/// GPU). One rule, one place, so the two can never disagree about a
/// buffer's size.
pub fn buffer_bytes(buffer: &Buffer<luminal::layouts::DecodedLayout>) -> Result<usize> {
    let numel = buffer.layout.literal_span_elements().ok_or_else(|| {
        anyhow!(
            "buffer {:?} (backing {}) has no literal span — symbolic or \
                 undisclosed-reach layouts are not executable",
            buffer.label,
            buffer.backs
        )
    })?;
    let dtype = buffer.layout.dtype.ok_or_else(|| {
        anyhow!(
            "buffer {:?} (backing {}) carries no dtype fact",
            buffer.label,
            buffer.backs
        )
    })?;
    Ok(numel * crate::host_buffer::dtype_bytes(dtype)?)
}

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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ArenaSlice {
    pub offset: usize,
    pub bytes: usize,
}

impl ArenaSlice {
    /// The reserved (aligned) extent — `bytes` rounded up to
    /// [`ARENA_ALIGN`].
    pub fn reserved(&self) -> usize {
        align_up(self.bytes.max(1))
    }
}

/// The exact schedule consumed by graph construction. Transfers of multiple
/// output views sharing a buffer are grouped within each output boundary.
#[derive(Debug, Clone)]
pub enum ArenaStep {
    Upload {
        buffer: BufferId,
        staging: ArenaSlice,
    },
    Node(NodeIndex),
    Download {
        buffer: BufferId,
        node: NodeIndex,
        slots: Vec<usize>,
        staging: ArenaSlice,
    },
}

#[derive(Debug, Clone, Default)]
pub struct ArenaPlan {
    /// A topological order respecting data and anti-dependencies, with late
    /// allocations and eager frees. `steps` expands this order with transfers.
    pub order: Vec<NodeIndex>,
    pub steps: Vec<ArenaStep>,
    pub slab_bytes: usize,
    /// Peak simultaneous reservations, including parameters and scratch.
    pub peak_live_bytes: usize,
    pub slices: FxHashMap<BufferId, ArenaSlice>,
    /// Scratch is live only during its owning operation's child graph.
    pub workspaces: FxHashMap<NodeIndex, ArenaSlice>,
    pub parameters: ArenaSlice,
    pub staging_parameters: ArenaSlice,
    pub staging_bytes: usize,
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
/// Two changes to Kahn's algorithm, both aimed at the same thing —
/// keeping a buffer's lifetime as short as the dependency structure
/// allows:
///
///  1. A `BufferFree` whose in-edges are all discharged goes FIRST. Its
///     in-edges are Data from the final resident's producer plus Anti
///     from every other toucher, so this is precisely "free the instant
///     the last toucher has run".
///  2. A `BufferAlloc` IS NEVER QUEUED AT ALL. It is PULLED: its edge to
///     its first toucher is left out of that toucher's in-degree, and
///     when the toucher is popped, its not-yet-issued alloc predecessors
///     are emitted immediately before it. So an alloc lands where
///     bufferize meant it to land — "before its buffer's first toucher"
///     — no matter which of the ready nodes the frontier happens to
///     pick.
///
/// The pull is what makes the difference on real plans. QUEUEING allocs
/// at the lowest priority is not enough, and the failure mode is worth
/// recording: when the frontier stalls (every compute node waits on its
/// own destination's alloc), the scheduler must issue SOME alloc, and a
/// node-index tie-break issues one whose toucher is nowhere near ready.
/// Measured on a two-layer mini-llama block (d=128, 484 nodes) under the
/// queued policy: the six d x ff weight materializations were allocated
/// at positions 1..27 and first touched at 447..475, and the high-water
/// mark came to 99% of the naive sum. Pulling instead of queueing is
/// what closes that gap.
///
/// An alloc that is dead (no outgoing edge) or that somehow carries
/// in-edges of its own cannot be pulled; it stays an ordinary queued
/// node, which is the pre-arena behaviour for it and always correct.
///
/// Ties inside a queue break on node index, which is the bufferizer's
/// own emission order, so equally-ready work runs in the order the
/// planner wrote it.
fn issue_order<L: PlanLayout>(plan: &BufferIrGraph<L>) -> Result<Vec<NodeIndex>> {
    let bound = plan.dag.node_bound();
    let incoming = |index: NodeIndex| {
        plan.dag
            .edges_directed(index, petgraph::Direction::Incoming)
            .count()
    };
    // An alloc is PULLABLE iff it depends on nothing and something
    // depends on it: then it can be emitted, always legally, at the
    // moment its first consumer is emitted.
    let mut pullable = vec![false; bound];
    for index in plan.dag.node_indices() {
        pullable[index.index()] = kind_of(&plan.dag[index]) == Kind::Alloc
            && incoming(index) == 0
            && plan
                .dag
                .edges_directed(index, petgraph::Direction::Outgoing)
                .next()
                .is_some();
    }
    // In-degrees COUNT ONLY non-pulled predecessors: a pulled alloc's
    // edge is discharged by the pull itself.
    let mut indegree: Vec<usize> = vec![0; bound];
    for index in plan.dag.node_indices() {
        indegree[index.index()] = plan
            .dag
            .edges_directed(index, petgraph::Direction::Incoming)
            .filter(|edge| !pullable[edge.source().index()])
            .count();
    }
    // Two ready queues, each min-ordered by node index (`Reverse`).
    let mut frees: BinaryHeap<std::cmp::Reverse<usize>> = BinaryHeap::new();
    let mut ordinary: BinaryHeap<std::cmp::Reverse<usize>> = BinaryHeap::new();
    let push = |index: NodeIndex,
                frees: &mut BinaryHeap<std::cmp::Reverse<usize>>,
                ordinary: &mut BinaryHeap<std::cmp::Reverse<usize>>| {
        match kind_of(&plan.dag[index]) {
            Kind::Free => frees.push(std::cmp::Reverse(index.index())),
            _ => ordinary.push(std::cmp::Reverse(index.index())),
        }
    };
    for index in plan.dag.node_indices() {
        if !pullable[index.index()] && indegree[index.index()] == 0 {
            push(index, &mut frees, &mut ordinary);
        }
    }
    let mut order = Vec::with_capacity(plan.dag.node_count());
    let mut issued = vec![false; bound];
    while let Some(std::cmp::Reverse(raw)) = frees.pop().or_else(|| ordinary.pop()) {
        let index = NodeIndex::new(raw);
        // THE PULL: this node's storage comes into existence right here,
        // not at the top of the program.
        for edge in plan
            .dag
            .edges_directed(index, petgraph::Direction::Incoming)
        {
            let source = edge.source();
            if pullable[source.index()] && !issued[source.index()] {
                issued[source.index()] = true;
                order.push(source);
            }
        }
        issued[index.index()] = true;
        order.push(index);
        for edge in plan
            .dag
            .edges_directed(index, petgraph::Direction::Outgoing)
        {
            let target = edge.target();
            if pullable[target.index()] {
                continue; // an alloc is never unlocked; it is pulled
            }
            indegree[target.index()] -= 1;
            if indegree[target.index()] == 0 {
                push(target, &mut frees, &mut ordinary);
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
    fn alloc(&mut self, need: usize) -> Result<usize> {
        // FIRST fit, in offset order — MEASURED against the obvious
        // alternative and kept. First fit leaves about a fifth of the
        // slab in holes on the two-layer mini-llama block (596480 B
        // high-water against a 498176 B peak live, which is the
        // fragmentation-free lower bound over the same order), and BEST
        // fit — the tightest hole that holds the request — is WORSE:
        // 663040 B on the same plan, because it shaves every large hole
        // down into slivers nothing later fits into. If this is ever
        // revisited, the thing to try is not another fit rule but
        // offset assignment over whole lifetimes (the greedy-by-size
        // arena planners), which is a different pass, not a different
        // line.
        if let Some((&offset, &len)) = self.holes.iter().find(|&(_, &len)| len >= need) {
            self.holes.remove(&offset);
            if len > need {
                self.holes.insert(offset + need, len - need);
            }
            return Ok(offset);
        }
        // No hole fits. If the LAST hole runs right up to the top, grow
        // through it instead of stranding it (coalescing with the
        // wilderness — the classic dlmalloc move).
        if let Some((&offset, &len)) = self.holes.iter().next_back()
            && offset + len == self.top
        {
            self.holes.remove(&offset);
            self.top = offset
                .checked_add(need)
                .ok_or_else(|| anyhow!("arena size overflow"))?;
            return Ok(offset);
        }
        let offset = self.top;
        self.top = self
            .top
            .checked_add(need)
            .ok_or_else(|| anyhow!("arena size overflow"))?;
        Ok(offset)
    }

    fn free(&mut self, offset: usize, len: usize) {
        let mut offset = offset;
        let mut len = len;
        // Coalesce with the predecessor hole, if it ends here.
        if let Some((&prev, &prev_len)) = self.holes.range(..offset).next_back()
            && prev + prev_len == offset
        {
            self.holes.remove(&prev);
            offset = prev;
            len += prev_len;
        }
        // …and with the successor, if it starts where we end.
        if let Some((&next, &next_len)) = self.holes.range(offset + len..).next()
            && next == offset + len
        {
            self.holes.remove(&next);
            len += next_len;
        }
        self.holes.insert(offset, len);
    }
}

/// Half-open lifetime in the execution schedule. Requests alive at the same
/// step must be disjoint, including a copy's source and destination.
#[derive(Debug, Clone, Copy)]
struct Lifetime {
    start: usize,
    end: usize,
    bytes: usize,
}

/// The same allocator serves device memory and pinned host memory. Only their
/// alignment and lifetimes differ. Returned slices follow request order.
fn pack(lifetimes: &[Lifetime], alignment: usize) -> Result<(Vec<ArenaSlice>, usize, usize)> {
    let mut events = Vec::with_capacity(lifetimes.len() * 2);
    for (id, life) in lifetimes.iter().enumerate() {
        ensure!(life.start < life.end, "empty physical lifetime");
        events.push((life.start, true, id));
        events.push((life.end, false, id));
    }
    events.sort_unstable(); // releases before allocations at the same boundary
    let mut slices = vec![ArenaSlice::default(); lifetimes.len()];
    let mut free_list = FreeList::default();
    let mut live = BTreeMap::<usize, usize>::new();
    let mut live_bytes = 0usize;
    let mut peak = 0;
    for (_, alloc, id) in events {
        let bytes = lifetimes[id].bytes;
        let need = bytes
            .max(1)
            .checked_add(alignment - 1)
            .map(|v| v / alignment * alignment)
            .ok_or_else(|| anyhow!("arena alignment overflow"))?;
        if alloc {
            let offset = free_list.alloc(need)?;
            ensure!(
                live.range(..=offset)
                    .next_back()
                    .is_none_or(|(&p, &n)| p + n <= offset),
                "arena overlaps a live predecessor"
            );
            ensure!(
                live.range(offset..)
                    .next()
                    .is_none_or(|(&p, _)| offset + need <= p),
                "arena overlaps a live successor"
            );
            live.insert(offset, need);
            live_bytes = live_bytes
                .checked_add(need)
                .ok_or_else(|| anyhow!("live size overflow"))?;
            peak = peak.max(live_bytes);
            slices[id] = ArenaSlice { offset, bytes };
        } else {
            let offset = slices[id].offset;
            ensure!(
                live.remove(&offset) == Some(need),
                "release of non-live range"
            );
            live_bytes -= need;
            free_list.free(offset, need);
        }
    }
    Ok((slices, free_list.top, peak))
}

/// Plan private device copies of a bufferized program. Unlike logical caller
/// storage, these ranges only need to survive through their GPU uses/readbacks.
/// The CUDA adapter also supplies parameter and per-operation scratch sizes.
pub fn plan_arena<L: PlanLayout>(
    plan: &BufferIrGraph<L>,
    bytes_of: impl Fn(&Buffer<L>) -> Result<usize>,
) -> Result<ArenaPlan> {
    plan_arena_over(plan, bytes_of, |_| Ok(0), 0, issue_order(plan)?)
}

pub(crate) fn plan_with_workspace<L: PlanLayout>(
    plan: &BufferIrGraph<L>,
    bytes_of: impl Fn(&Buffer<L>) -> Result<usize>,
    scratch_of: impl Fn(NodeIndex) -> Result<usize>,
    parameter_bytes: usize,
) -> Result<ArenaPlan> {
    plan_arena_over(
        plan,
        bytes_of,
        scratch_of,
        parameter_bytes,
        issue_order(plan)?,
    )
}

fn plan_arena_over<L: PlanLayout>(
    plan: &BufferIrGraph<L>,
    bytes_of: impl Fn(&Buffer<L>) -> Result<usize>,
    scratch_of: impl Fn(NodeIndex) -> Result<usize>,
    parameter_bytes: usize,
    order: Vec<NodeIndex>,
) -> Result<ArenaPlan> {
    let mut allocs = FxHashMap::default();
    let mut frees = FxHashMap::default();
    for &node in &order {
        if let Some(id) = allocated(&plan.dag[node]) {
            ensure!(
                allocs.insert(id.clone(), node).is_none(),
                "buffer {id:?} allocated twice"
            );
            ensure!(
                plan.buffers[id].owner == Owner::System,
                "caller buffer {id:?} has an alloc"
            );
        }
        if let Some(id) = freed(&plan.dag[node]) {
            ensure!(
                frees.insert(id.clone(), node).is_none(),
                "buffer {id:?} freed twice"
            );
            ensure!(
                plan.buffers[id].freed_by == FreedBy::Program,
                "caller-freed buffer {id:?} has a free"
            );
        }
    }
    // Hand-built plans without alloc/free markers keep their existing implicit
    // bindings. Explicit markers are authoritative and checked for containment.
    let mut live: std::collections::HashSet<_> = plan
        .buffers
        .keys()
        .filter(|id| !allocs.contains_key(*id))
        .cloned()
        .collect();
    let mut uploaded = std::collections::HashSet::new();
    let mut steps = vec![];
    for &node in &order {
        let op = &plan.dag[node];
        if let Some(id) = allocated(op) {
            ensure!(live.insert(id.clone()), "alloc of live buffer {id:?}");
        }
        let mut touched: Vec<&BufferId> = match op {
            BufferNode::Compute { reads, writes, .. } => reads.iter().chain(writes).collect(),
            BufferNode::BufferCopy { src, dst } => vec![src, dst],
            BufferNode::BufferOutput { slots } => slots.iter().map(|s| &s.buffer).collect(),
            BufferNode::BufferInput { .. } => vec![],
        };
        // Preserve operand order, deduplicating tied operands/results.
        let mut seen = std::collections::HashSet::new();
        touched.retain(|id| seen.insert((*id).clone()));
        for id in touched {
            ensure!(
                live.contains(id),
                "node {node:?} touches non-live buffer {id:?}"
            );
            if plan.buffers[id].lit.is_some() && uploaded.insert(id.clone()) {
                steps.push(ArenaStep::Upload {
                    buffer: id.clone(),
                    staging: ArenaSlice::default(),
                });
            }
        }
        steps.push(ArenaStep::Node(node));
        if let BufferNode::BufferOutput { slots } = op {
            let mut groups: Vec<(BufferId, Vec<usize>)> = vec![];
            for (i, slot) in slots.iter().enumerate() {
                ensure!(
                    plan.buffers[&slot.buffer].freed_by == FreedBy::Caller,
                    "output slot {} has NON-ESCAPING buffer",
                    slot.index
                );
                if let Some((_, indices)) = groups.iter_mut().find(|(id, _)| id == &slot.buffer) {
                    indices.push(i);
                } else {
                    groups.push((slot.buffer.clone(), vec![i]));
                }
            }
            for (buffer, slots) in groups {
                steps.push(ArenaStep::Download {
                    buffer,
                    node,
                    slots,
                    staging: ArenaSlice::default(),
                });
            }
        }
        if let Some(id) = freed(op) {
            ensure!(live.remove(id), "free of non-live buffer {id:?}");
        }
    }

    let mut intervals = vec![];
    let mut buffers = FxHashMap::<BufferId, usize>::default();
    let mut workspaces = FxHashMap::default();
    let parameter = (parameter_bytes > 0).then(|| {
        intervals.push(Lifetime {
            start: 0,
            end: steps.len().max(1),
            bytes: parameter_bytes,
        });
        0
    });
    let mut touch = |id: &BufferId, at: usize, intervals: &mut Vec<Lifetime>| -> Result<()> {
        if let Some(&i) = buffers.get(id) {
            intervals[i].end = at + 1;
        } else {
            buffers.insert(id.clone(), intervals.len());
            intervals.push(Lifetime {
                start: at,
                end: at + 1,
                bytes: bytes_of(&plan.buffers[id])?,
            });
        }
        Ok(())
    };
    for (at, step) in steps.iter().enumerate() {
        match step {
            ArenaStep::Upload { buffer, .. } | ArenaStep::Download { buffer, .. } => {
                touch(buffer, at, &mut intervals)?
            }
            ArenaStep::Node(node) => {
                match &plan.dag[*node] {
                    BufferNode::Compute { reads, writes, .. } => {
                        for id in reads.iter().chain(writes) {
                            touch(id, at, &mut intervals)?;
                        }
                    }
                    BufferNode::BufferCopy { src, dst } => {
                        touch(src, at, &mut intervals)?;
                        touch(dst, at, &mut intervals)?;
                    }
                    _ => {}
                }
                let scratch = scratch_of(*node)?;
                if scratch > 0 {
                    workspaces.insert(*node, intervals.len());
                    intervals.push(Lifetime {
                        start: at,
                        end: at + 1,
                        bytes: scratch,
                    });
                }
            }
        }
    }
    let (slices, slab_bytes, peak_live_bytes) = pack(&intervals, ARENA_ALIGN)?;
    let parameters = parameter.map(|i| slices[i]).unwrap_or_default();
    let buffers: FxHashMap<_, _> = buffers.into_iter().map(|(id, i)| (id, slices[i])).collect();
    let workspaces = workspaces
        .into_iter()
        .map(|(node, i)| (node, slices[i]))
        .collect();

    // All host inputs are populated before launch and must survive until their
    // upload; each output survives from readback through host result collection.
    // A separate time 0 represents the parameter upload preceding `steps`.
    let mut staging = vec![];
    let staging_parameter = parameter.map(|_| {
        staging.push(Lifetime {
            start: 0,
            end: 1,
            bytes: parameter_bytes,
        });
        0
    });
    let mut transfers = vec![];
    for (at, step) in steps.iter().enumerate() {
        let (buffer, start, end) = match step {
            ArenaStep::Upload { buffer, .. } => (buffer, 0, at + 2),
            ArenaStep::Download { buffer, .. } => (buffer, at + 1, steps.len() + 2),
            ArenaStep::Node(_) => continue,
        };
        transfers.push((at, staging.len()));
        staging.push(Lifetime {
            start,
            end,
            bytes: buffers[buffer].bytes,
        });
    }
    let (staging_slices, staging_bytes, _) = pack(&staging, 1)?;
    for (at, i) in transfers {
        match &mut steps[at] {
            ArenaStep::Upload { staging, .. } | ArenaStep::Download { staging, .. } => {
                *staging = staging_slices[i]
            }
            ArenaStep::Node(_) => unreachable!(),
        }
    }
    Ok(ArenaPlan {
        order,
        steps,
        slab_bytes,
        peak_live_bytes,
        slices: buffers,
        workspaces,
        parameters,
        staging_parameters: staging_parameter
            .map(|i| staging_slices[i])
            .unwrap_or_default(),
        staging_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use luminal::index_expr::IotaExpr;
    use luminal::layout_ir::Access;
    use luminal::test_support::{MockLayout, MockOp, MockViewWithMap, TestGraph, bufferize_mock};

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
            |_| Ok(0),
            0,
            plan.dag.node_indices().collect::<Vec<_>>(),
        )
        .expect("node-index order plans");
        let raw = plan_arena_over(
            &plan,
            unit_bytes,
            |_| Ok(0),
            0,
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
        assert!(
            raw.slab_bytes > arena.slab_bytes,
            "hoisted allocations cost more"
        );
    }

    /// Escaping storage keeps its logical ownership, while its private device
    /// copy participates in physical packing through the output readback.
    #[test]
    fn escaping_minted_storage_is_packed_without_a_logical_free() {
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
        assert!(arena.slices.contains_key(&escaping));
        assert!(plan.dag.node_weights().all(|n| freed(n) != Some(&escaping)));
        assert!(arena.steps.iter().any(|step| matches!(step,
            ArenaStep::Download { buffer, .. } if buffer == &escaping)));
    }

    /// Donation's explicit free remains authoritative, and the private copy
    /// receives an ordinary arena range.
    #[test]
    fn donated_device_copy_is_packed_and_frees_after_all_uses() {
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
        assert!(arena.slices.contains_key(&donated));
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
        let mut sharing = 0usize;
        let lifetime = |id: &BufferId| {
            let uses: Vec<_> = arena
                .steps
                .iter()
                .enumerate()
                .filter_map(|(i, step)| {
                    let touches = match step {
                        ArenaStep::Upload { buffer, .. } | ArenaStep::Download { buffer, .. } => {
                            buffer == id
                        }
                        ArenaStep::Node(node) => match &plan.dag[*node] {
                            BufferNode::Compute { reads, writes, .. } => {
                                reads.contains(id) || writes.contains(id)
                            }
                            BufferNode::BufferCopy { src, dst } => src == id || dst == id,
                            _ => false,
                        },
                    };
                    touches.then_some(i)
                })
                .collect();
            (*uses.first().unwrap(), *uses.last().unwrap())
        };
        let members: Vec<_> = arena.slices.iter().collect();
        for (i, (a, sa)) in members.iter().enumerate() {
            for (b, sb) in &members[i + 1..] {
                if sa.offset >= sb.offset + sb.reserved() || sb.offset >= sa.offset + sa.reserved()
                {
                    continue;
                }
                sharing += 1;
                let (start_a, end_a) = lifetime(a);
                let (start_b, end_b) = lifetime(b);
                assert!(
                    end_a < start_b || end_b < start_a,
                    "overlapping ranges have intersecting lifetimes: {a:?}, {b:?}"
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
    #[test]
    fn interval_packing_checks_all_live_ranges_with_fixed_seed() {
        // Vary sizes, lifetimes, and alignment independently of bufferization.
        let mut seed = 42u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            seed as usize
        };
        for alignment in [1, ARENA_ALIGN] {
            for _ in 0..30 {
                let lives: Vec<_> = (0..80)
                    .map(|_| {
                        let start = next() % 40;
                        Lifetime {
                            start,
                            end: start + 1 + next() % 15,
                            bytes: next() % 2049,
                        }
                    })
                    .collect();
                let (slices, total, peak) = pack(&lives, alignment).unwrap();
                let reservation = |bytes: usize| bytes.max(1).div_ceil(alignment) * alignment;
                let expected_peak = (0..55)
                    .map(|t| {
                        lives
                            .iter()
                            .filter(|l| l.start <= t && t < l.end)
                            .map(|l| reservation(l.bytes))
                            .sum::<usize>()
                    })
                    .max()
                    .unwrap();
                assert_eq!(peak, expected_peak);
                assert!(total >= peak);
                for (i, a) in lives.iter().enumerate() {
                    let sa = slices[i];
                    assert_eq!(sa.offset % alignment, 0);
                    assert!(sa.offset + reservation(sa.bytes) <= total);
                    for (j, b) in lives.iter().enumerate().skip(i + 1) {
                        if a.start < b.end && b.start < a.end {
                            let sb = slices[j];
                            assert!(
                                sa.offset + reservation(sa.bytes) <= sb.offset
                                    || sb.offset + reservation(sb.bytes) <= sa.offset
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn scratch_reuses_tensor_storage_and_staging_reuses_uploaded_inputs() {
        let plan = chain(6);
        let nodes: Vec<_> = plan
            .dag
            .node_indices()
            .filter(|&n| {
                matches!(&plan.dag[n],
            BufferNode::Compute { op, .. } if !matches!(op.label(), "BufferAlloc" | "BufferFree"))
            })
            .collect();
        let BufferNode::Compute { writes, .. } = &plan.dag[nodes[0]] else {
            unreachable!()
        };
        let large = &writes[0];
        let bytes = |b: &Buffer<MockLayout>| Ok(if &b.id == large { 8 * RESERVED } else { UNIT });
        let baseline = plan_with_workspace(&plan, bytes, |_| Ok(0), 8).unwrap();
        let host = *nodes.last().unwrap();
        let arena = plan_with_workspace(
            &plan,
            bytes,
            |n| Ok(if n == host { 4 * RESERVED } else { 0 }),
            8,
        )
        .unwrap();
        let scratch = arena.workspaces[&host];
        let early = arena.slices[large];
        assert!(
            scratch.offset < early.offset + early.reserved()
                && early.offset < scratch.offset + scratch.reserved(),
            "scratch must reuse the dead large tensor's range"
        );
        assert_eq!(
            arena.slab_bytes, baseline.slab_bytes,
            "scratch fits inside the existing high-water mark"
        );
        let staging_sum: usize = arena
            .steps
            .iter()
            .map(|s| match s {
                ArenaStep::Upload { staging, .. } | ArenaStep::Download { staging, .. } => {
                    staging.bytes
                }
                _ => 0,
            })
            .sum();
        assert!(
            arena.staging_bytes < staging_sum + 8,
            "staging must reuse completed uploads"
        );
    }

    #[test]
    fn packing_rejects_overflow_and_invalid_lifetimes() {
        assert!(
            pack(
                &[Lifetime {
                    start: 0,
                    end: 1,
                    bytes: usize::MAX
                }],
                ARENA_ALIGN
            )
            .is_err()
        );
        assert!(
            pack(
                &[Lifetime {
                    start: 2,
                    end: 2,
                    bytes: 1
                }],
                1
            )
            .is_err()
        );
    }
}
