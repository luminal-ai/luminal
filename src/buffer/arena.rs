//! Physical storage planning for graph execution. The serial adapter packs
//! device tensors, operation scratch, and parameters. Boundary storage is
//! supplied by the application; outputs remain on the device until return.
//! Tensor uploads and readbacks are never part of this schedule.

use crate::bufferize::{Buffer, BufferId, BufferIrGraph, BufferNode, Owner, PlanLayout};
use crate::layout_ir::FreedBy;
use crate::prelude::{FxHashMap, FxHashSet, NodeIndex, petgraph};
use anyhow::{Result, anyhow, bail, ensure};
use petgraph::visit::{EdgeRef, NodeIndexable};
use std::collections::{BTreeMap, BinaryHeap};

/// Placement policy for a bounded slab. First fit is deterministic in request
/// order and may fail even when a different placement would fit.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum SlabAlgorithm {
    #[default]
    FirstFit,
}

/// A physical buffer is live at, and between, all of these DAG nodes. Include
/// every read, write, transfer, and any use outside the computation itself.
/// IDs are application-defined; independent programs may namespace their IDs.
#[derive(Debug, Clone)]
pub struct SlabBuffer<Id> {
    pub id: Id,
    pub bytes: usize,
    pub uses: Vec<NodeIndex>,
}

/// Assign aligned offsets within `capacity` without changing DAG parallelism.
/// Two buffers may alias only if every use of one strictly precedes every use
/// of the other. Unordered branches and operands of the same node cannot alias.
/// The result follows request order and contains no execution schedule.
///
/// `alignment` applies to the entire plan, including reserved sizes. Empty
/// tensors reserve one aligned unit so their addresses remain distinct while
/// live. The caller owns the slab and must provide an aligned base address.
/// A capacity error means this algorithm did not find a placement, not that no
/// possible placement exists. This function makes no tensor geometry or access
/// permission assumptions.
pub fn plan_slab<N, E, Id: Clone + Eq + std::hash::Hash>(
    dag: &petgraph::graph::DiGraph<N, E>,
    buffers: &[SlabBuffer<Id>],
    capacity: usize,
    alignment: usize,
    algorithm: SlabAlgorithm,
) -> Result<Vec<(Id, usize)>> {
    ensure!(alignment > 0, "slab alignment must be positive");
    let order = petgraph::algo::toposort(dag, None)
        .map_err(|_| anyhow!("slab lifetime graph must be acyclic"))?;
    let mut ids = std::collections::HashSet::new();
    let sizes: Vec<_> = buffers
        .iter()
        .map(|buffer| {
            ensure!(ids.insert(&buffer.id), "duplicate slab buffer ID");
            ensure!(!buffer.uses.is_empty(), "slab buffer has no lifetime nodes");
            ensure!(
                buffer.uses.iter().all(|&n| dag.node_weight(n).is_some()),
                "slab lifetime references an absent DAG node"
            );
            aligned_size(buffer.bytes, alignment)
        })
        .collect::<Result<_>>()?;
    // Transitive closure of the supplied dependency DAG, never an invented
    // topological execution order. Bitsets keep queries cheap for large plans.
    let words = dag.node_bound().div_ceil(64);
    let mut after = vec![vec![0u64; words]; dag.node_bound()];
    for &node in order.iter().rev() {
        for next in dag.neighbors(node) {
            after[node.index()][next.index() / 64] |= 1 << (next.index() % 64);
            let [current, successor] = after
                .get_disjoint_mut([node.index(), next.index()])
                .unwrap();
            for (word, &reachable) in current.iter_mut().zip(successor.iter()) {
                *word |= reachable;
            }
        }
    }
    let precedes = |a: &SlabBuffer<Id>, b: &SlabBuffer<Id>| {
        a.uses.iter().all(|u| {
            b.uses
                .iter()
                .all(|v| after[u.index()][v.index() / 64] & (1 << (v.index() % 64)) != 0)
        })
    };
    let mut offsets = Vec::with_capacity(buffers.len());
    match algorithm {
        SlabAlgorithm::FirstFit => {
            for (i, buffer) in buffers.iter().enumerate() {
                let mut occupied: Vec<_> = (0..i)
                    .filter(|&j| !precedes(buffer, &buffers[j]) && !precedes(&buffers[j], buffer))
                    .map(|j| (offsets[j], sizes[j]))
                    .collect();
                occupied.sort_unstable();
                let mut offset = 0usize;
                for (start, bytes) in occupied {
                    if offset.checked_add(sizes[i]).is_some_and(|end| end <= start) {
                        break;
                    }
                    offset = offset.max(start + bytes);
                }
                ensure!(
                    offset
                        .checked_add(sizes[i])
                        .is_some_and(|end| end <= capacity),
                    "slab capacity {capacity} exceeded while placing request {i} ({} bytes)",
                    buffer.bytes
                );
                offsets.push(offset);
            }
        }
    }
    Ok(buffers
        .iter()
        .zip(offsets)
        .map(|(buffer, offset)| (buffer.id.clone(), offset))
        .collect())
}

/// Describe physical buffer uses using the bufferizer's own DAG node IDs.
/// Applications may filter caller-owned boundaries before planning or add
/// transfer nodes when their execution protocol extends these lifetimes.
pub fn buffer_lifetimes<L: PlanLayout>(
    plan: &BufferIrGraph<L>,
    bytes_of: impl Fn(&Buffer<L>) -> Result<usize>,
) -> Result<Vec<SlabBuffer<BufferId>>> {
    let mut buffers: Vec<SlabBuffer<BufferId>> = Vec::new();
    let mut indices = FxHashMap::default();
    for node in plan.dag.node_indices() {
        let ids: Vec<_> = match &plan.dag[node] {
            BufferNode::BufferInput { slots } => slots.iter().map(|s| &s.buffer).collect(),
            BufferNode::Compute { reads, writes, .. } => reads.iter().chain(writes).collect(),
            BufferNode::BufferCopy { src, dst } => vec![src, dst],
            BufferNode::BufferOutput { slots } => slots.iter().map(|s| &s.buffer).collect(),
        };
        for id in ids {
            let index = if let Some(&index) = indices.get(id) {
                index
            } else {
                let index = buffers.len();
                let buffer = plan
                    .buffers
                    .get(id)
                    .ok_or_else(|| anyhow!("unknown buffer {id:?}"))?;
                buffers.push(SlabBuffer {
                    id: id.clone(),
                    bytes: bytes_of(buffer)?,
                    uses: vec![],
                });
                indices.insert(id.clone(), index);
                index
            };
            if buffers[index].uses.last() != Some(&node) {
                buffers[index].uses.push(node);
            }
        }
    }
    Ok(buffers)
}

fn aligned_size(bytes: usize, alignment: usize) -> Result<usize> {
    ensure!(alignment > 0, "slab alignment must be positive");
    bytes
        .max(1)
        .checked_add(alignment - 1)
        .map(|v| v / alignment * alignment)
        .ok_or_else(|| anyhow!("slab alignment overflow"))
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

/// An operation in the serial device schedule. Tensor transfers belong to callers.
#[derive(Debug, Clone)]
pub enum ArenaStep {
    Node(NodeIndex),
}

#[derive(Debug, Clone, Default)]
pub struct ArenaPlan {
    /// A topological order respecting data and anti-dependencies, with late
    /// allocations and eager frees. `steps` contains only device operations.
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
    /// Buffers whose storage is caller-owned device memory (zero-copy
    /// boundaries). They reserve no slab range and get no upload/download step;
    /// the executor resolves them through its external-pointer map. A buffer in
    /// here that the executor has no pointer for is a hard error, never a
    /// silent arena fallback.
    pub externals: FxHashSet<BufferId>,
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
pub fn issue_order<L: PlanLayout>(plan: &BufferIrGraph<L>) -> Result<Vec<NodeIndex>> {
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
fn pack(
    lifetimes: &[Lifetime],
    alignment: usize,
    capacity: usize,
    algorithm: SlabAlgorithm,
) -> Result<(Vec<ArenaSlice>, usize, usize)> {
    ensure!(alignment > 0, "slab alignment must be positive");
    let SlabAlgorithm::FirstFit = algorithm;
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
                free_list.top <= capacity,
                "slab capacity {capacity} exceeded"
            );
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

/// Plan device storage for a bufferized program. Caller boundary storage stays
/// live through the call; intermediate ranges can be reused after their last use.
/// The CUDA adapter also supplies parameter and per-operation scratch sizes.
pub fn plan_arena<L: PlanLayout>(
    plan: &BufferIrGraph<L>,
    bytes_of: impl Fn(&Buffer<L>) -> Result<usize>,
) -> Result<ArenaPlan> {
    plan_arena_over(plan, bytes_of, |_| Ok(0), 0, issue_order(plan)?)
}

pub fn plan_with_workspace<L: PlanLayout>(
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
    plan_external_over(
        plan,
        bytes_of,
        scratch_of,
        parameter_bytes,
        order,
        &Default::default(),
    )
}

/// Plan a serial execution schedule, excluding caller-owned device boundaries.
/// Executors must obey the returned steps for these offsets to remain valid.
#[allow(clippy::too_many_arguments)]
pub fn plan_external_over<L: PlanLayout>(
    plan: &BufferIrGraph<L>,
    bytes_of: impl Fn(&Buffer<L>) -> Result<usize>,
    scratch_of: impl Fn(NodeIndex) -> Result<usize>,
    parameter_bytes: usize,
    order: Vec<NodeIndex>,
    external_buffers: &FxHashSet<BufferId>,
) -> Result<ArenaPlan> {
    let is_external = |id: &BufferId| external_buffers.contains(id);
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
    let mut steps = vec![];
    for &node in &order {
        let op = &plan.dag[node];
        if let Some(id) = allocated(op) {
            ensure!(live.insert(id.clone()), "alloc of live buffer {id:?}");
        }
        let touched: Vec<&BufferId> = match op {
            BufferNode::Compute { reads, writes, .. } => reads.iter().chain(writes).collect(),
            BufferNode::BufferCopy { src, dst } => vec![src, dst],
            BufferNode::BufferOutput { slots } => {
                for slot in slots {
                    ensure!(
                        plan.buffers[&slot.buffer].freed_by == FreedBy::Caller,
                        "output slot {} has NON-ESCAPING buffer",
                        slot.index
                    );
                }
                slots.iter().map(|s| &s.buffer).collect()
            }
            BufferNode::BufferInput { slots } => slots.iter().map(|s| &s.buffer).collect(),
        };
        for id in touched {
            ensure!(
                live.contains(id),
                "node {node:?} touches non-live buffer {id:?}"
            );
        }
        steps.push(ArenaStep::Node(node));
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
        if is_external(id) {
            return Ok(());
        }
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
                    BufferNode::BufferInput { slots } => {
                        for slot in slots {
                            touch(&slot.buffer, at, &mut intervals)?;
                        }
                    }
                    BufferNode::BufferOutput { slots } => {
                        for slot in slots {
                            touch(&slot.buffer, at, &mut intervals)?;
                        }
                    }
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
    // Inputs exist before the first operation; escaped results must survive
    // until return. No implicit upload/readback can shorten these lifetimes.
    for (id, &i) in &buffers {
        if plan.buffers[id].owner == Owner::Caller {
            intervals[i].start = 0;
        }
        if plan.buffers[id].freed_by == FreedBy::Caller {
            intervals[i].end = steps.len().max(1);
        }
    }
    let (slices, slab_bytes, peak_live_bytes) =
        pack(&intervals, ARENA_ALIGN, usize::MAX, SlabAlgorithm::FirstFit)?;
    let parameters = parameter.map(|i| slices[i]).unwrap_or_default();
    let buffers: FxHashMap<_, _> = buffers.into_iter().map(|(id, i)| (id, slices[i])).collect();
    let workspaces = workspaces
        .into_iter()
        .map(|(node, i)| (node, slices[i]))
        .collect();

    Ok(ArenaPlan {
        order,
        steps,
        slab_bytes,
        peak_live_bytes,
        slices: buffers,
        workspaces,
        parameters,
        staging_parameters: ArenaSlice {
            offset: 0,
            bytes: parameter_bytes,
        },
        staging_bytes: parameter_bytes,
        externals: external_buffers.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index_expr::IotaExpr;
    use crate::layout_ir::Access;
    use crate::test_support::{MockLayout, MockOp, MockViewWithMap, TestGraph, bufferize_mock};

    #[test]
    fn slab_preserves_parallel_branches_and_reuses_after_join() {
        let mut dag = petgraph::graph::DiGraph::<(), ()>::new();
        let a = dag.add_node(());
        let b = dag.add_node(());
        let join = dag.add_node(());
        dag.add_edge(a, join, ());
        dag.add_edge(b, join, ());
        let requests = vec![
            SlabBuffer {
                id: "a",
                bytes: 257,
                uses: vec![a],
            },
            SlabBuffer {
                id: "b",
                bytes: 256,
                uses: vec![b],
            },
            SlabBuffer {
                id: "joined",
                bytes: 512,
                uses: vec![join],
            },
        ];
        assert!(plan_slab(&dag, &requests, 767, 256, SlabAlgorithm::FirstFit).is_err());
        assert_eq!(
            plan_slab(&dag, &requests, 768, 256, SlabAlgorithm::FirstFit).unwrap(),
            vec![("a", 0), ("b", 512), ("joined", 0)]
        );
    }

    #[test]
    fn slab_lifetime_includes_all_uses_and_gaps_between_them() {
        let mut dag = petgraph::graph::DiGraph::<(), ()>::new();
        let start = dag.add_node(());
        let middle = dag.add_node(());
        let end = dag.add_node(());
        dag.add_edge(start, middle, ());
        dag.add_edge(middle, end, ());
        let requests = vec![
            SlabBuffer {
                id: 0,
                bytes: 16,
                uses: vec![start, end],
            },
            SlabBuffer {
                id: 1,
                bytes: 16,
                uses: vec![middle],
            },
            SlabBuffer {
                id: 2,
                bytes: 16,
                uses: vec![end],
            },
        ];
        assert_eq!(
            plan_slab(&dag, &requests, 32, 16, SlabAlgorithm::FirstFit).unwrap(),
            vec![(0, 0), (1, 16), (2, 16)]
        );
        assert!(plan_slab(&dag, &requests, 31, 16, SlabAlgorithm::FirstFit).is_err());
    }

    #[test]
    fn slab_rejects_invalid_requests() {
        let mut dag = petgraph::graph::DiGraph::<(), ()>::new();
        let node = dag.add_node(());
        let request = SlabBuffer {
            id: 0,
            bytes: 1,
            uses: vec![node],
        };
        assert!(
            plan_slab(
                &dag,
                std::slice::from_ref(&request),
                1,
                0,
                SlabAlgorithm::FirstFit
            )
            .is_err()
        );
        assert!(
            plan_slab(
                &dag,
                &[request.clone(), request.clone()],
                8,
                1,
                SlabAlgorithm::FirstFit
            )
            .is_err()
        );
        assert!(
            plan_slab(
                &dag,
                &[SlabBuffer {
                    bytes: usize::MAX,
                    ..request.clone()
                }],
                usize::MAX,
                256,
                SlabAlgorithm::FirstFit
            )
            .is_err()
        );
        assert!(
            plan_slab(
                &dag,
                &[SlabBuffer {
                    uses: vec![],
                    ..request.clone()
                }],
                8,
                1,
                SlabAlgorithm::FirstFit
            )
            .is_err()
        );
        dag.add_edge(node, node, ());
        assert!(plan_slab(&dag, &[request], 8, 1, SlabAlgorithm::FirstFit).is_err());
    }

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
    fn chain(steps: usize) -> crate::bufferize::BufferIrGraph<MockLayout> {
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
        plan: &crate::bufferize::BufferIrGraph<L>,
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
            4 * RESERVED,
            "two boundary buffers plus producer + consumer remain live \
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

    /// Escaping storage remains in the device arena until the caller consumes it.
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
        assert!(plan.dag.node_weights().any(|node| matches!(node,
            BufferNode::BufferOutput { slots } if slots.iter().any(|slot| slot.buffer == escaping))));
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
            if edge.weight().kind == crate::bufferize::EdgeKind::Anti {
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
                let (slices, total, peak) =
                    pack(&lives, alignment, usize::MAX, SlabAlgorithm::FirstFit).unwrap();
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
    fn scratch_reuses_dead_tensor_storage_without_tensor_staging() {
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
        assert_eq!(
            arena.staging_bytes, 8,
            "only dimension parameters use staging"
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
                ARENA_ALIGN,
                usize::MAX,
                SlabAlgorithm::FirstFit
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
                1,
                usize::MAX,
                SlabAlgorithm::FirstFit
            )
            .is_err()
        );
    }
}
