//! CUDA-lite on the native ladder.
//!
//! The same six-method ladder as the reference `ReferenceRuntime`
//! (`load → bind_* → search → set_data → execute → get_*`), consuming
//! the same `BufferIrGraph` plans, claiming ops through the same
//! allow-list seam — but executing on a CUDA device with
//! NVRTC-compiled kernels instead of host loops.
//!
//! THE OP SET IS CHOSEN AT INITIALIZATION (ruling 2026-09-03, #420/#422
//! rejoin Phase 2): a runtime's matcher vocabulary and its derived allow
//! list come from the `Vec<RegisteredOp>` handed to
//! [`CudaRuntime::load_with_registry`] — [`cuda_registry`] (the DEFAULT:
//! every row this crate ships, cuBLASLt markers included since the
//! 2026-09-04 ruling) and [`cuda_registry_without_cublaslt`] (the
//! decomposed route on purpose) are the presets, and
//! [`cuda_registry_filtered`] narrows either without editing this crate.
//!
//! THIS CRATE OWNS ITS SEARCH (ruling 2026-09-03, #420/#422 rejoin
//! Phase 1): [`extractor`] and [`search`] are this runtime's own copies
//! — core keeps the shared machinery (program, assembly, `dps_rewrite`,
//! `decode_layout_table`, `bufferize`) and nothing that decides which
//! implementation wins. Candidates rank, by default, through
//! [`heuristic`], a device-free byte-move prior (CL-5 below is the
//! opt-in device measurement); there is no profiler trait anywhere any
//! more, in core or here. Host payloads are [`host_buffer::HostBuffer`],
//! not the reference runtime's `TypedBuffer`.
//!
//! Stage discipline (M4 kickoff ruling, 2026-08-17: "just focus on
//! getting cuda lite up and running"):
//! - CL-1: plan-level runtime + kernel codegen, buildable and testable
//!   everywhere; device execution behind the `device` feature.
//! - CL-2: bring-up on a real device; fidelity vs the reference over
//!   the mini battery.
//! - CL-3: CUDA-native ops (cuBLASLt first).
//! - CL-4: in-place ties (the Mutating family), views + resident
//!   geometry.
//! - CL-5 (#420/#422 rejoin Phase 4, 2026-09-03): PROFILING ON DEVICE,
//!   mirroring the reference evaluator's design —
//!   `CompileOptions::profile_on_device` ranks candidates by measured
//!   device time ([`profile`]) instead of by [`heuristic`]'s weak prior.
//!   The heuristic remains the default and the device-free hosts' only
//!   option; the two are never blended.
//! - CL-6 (#420/#422 rejoin Phase 5, 2026-09-03): FINALISTS AND THE
//!   BUCKET LATTICE ([`finalists`], [`lattice`]). The search keeps a
//!   ranked list of genomes and the plan that gets INSTALLED is chosen by
//!   a best-first walk over the buckets' finalist ranks under one
//!   aggregate constraint — `CompileOptions::device_budget_bytes` bounds
//!   the arena slab the runtime will hold. Unconstrained (the default)
//!   the walk installs the search's own winner and costs nothing.
//!
//! Execution always launches CUDA graphs. Owned kernels and copies are parent
//! graph nodes; HostOps prepare library calls and capture opaque child graphs.
//! Dynamic dimensions live in an arena-resident parameter block. Buckets overlay
//! one capacity-sized arena, including boundary device copies and host-op scratch.
//! Input staging and output readback are graph nodes backed by one shared pinned
//! host allocation. Returned outputs own their host bytes.
//!
//! Bufferization supplies alias/lifetime contracts and dependency edges. The
//! arena preserves them in a serial issue order; generated kernels must fully
//! initialize their destinations, since temporary ranges are reused without
//! clearing. cuBLASLt's accumulate forms follow their declared alias contracts.

pub mod arena;
pub mod binding_check;
pub mod bindings;
/// The e-graph walk, in core (#420/#422 rejoin Phase 8): every runtime
/// calls it with its own matcher list and it names no runtime type.
/// Kept under this crate's old module name so call sites read the same.
pub use luminal::extraction as extractor;
#[cfg(feature = "device")]
mod cuda_graph;
/// FINALISTS (Phase 5 of the #420/#422 rejoin): a bucket's ranked
/// genomes, re-materialized one at a time under a hard filter.
pub mod finalists;
pub mod heuristic;
pub mod host;
pub mod host_buffer;
pub mod kernels;
/// THE BUCKET LATTICE (Phase 5): best-first selection of ONE finalist
/// per bucket under a coordinate-monotone aggregate.
pub mod lattice;
pub mod layouts;
pub mod op;
pub mod ops;
pub mod resident;
pub mod runtime;
pub mod search;
mod storage;
pub mod symbolic;

#[cfg(feature = "device")]
pub mod device;
/// ON-DEVICE CANDIDATE PROFILING (Phase 4): the search's device
/// evaluator. Device builds only — it measures, so it needs a device.
#[cfg(feature = "device")]
pub mod profile;

pub use bindings::CudaBindings;
pub use host::HostOp;
pub use host_buffer::HostBuffer;
pub use kernels::KernelOp;
pub use layouts::CudaPlan;
pub use op::{CudaOpInterface, as_host_op, as_kernel_op};
pub use ops::{
    RegisteredOp, cuda_registry, cuda_registry_filtered, cuda_registry_without_cublaslt,
};
pub use runtime::CudaRuntime;
pub use search::{CompileOptions, Evaluator, SearchOutcome, harness_search_options};

/// PLAN-TRANSPARENT (M4 Phase 5): claimable WITHOUT a kernel iff the
/// op's DECLARED EFFECTS prove the planner folds it — no operand ever
/// reads memory, no result ever writes memory, exactly one Must tie
/// binding result 0 into operand 0's storage, and no DPS form (nothing
/// is written, so there is no destination to pass). This is the
/// allow-list face of the lowering fold in `luminal::bufferize` (the
/// view-shaped predicate at lowering, plus the unfolded-view plan
/// validator as the fence): an op these predicates admit never reaches
/// the device as a kernel — it lowers to a producer redirect and its
/// consumers read through the recorded composed access. Derived from
/// trait answers on a prototype instance, NEVER from an op-name list.
pub fn plan_transparent(op: &dyn luminal::layout_ir::LayoutIrOp) -> bool {
    use luminal::layout_ir::{AliasInfo, Sharing};
    let ties = op.alias_info();
    ties == [AliasInfo {
        operand: 0,
        result: 0,
        sharing: Sharing::Must,
    }] && !op.operand_reads_memory(0)
        && !op.result_writes_memory(0)
        && op.to_dps().is_none()
}

/// The op labels the DEFAULT registry preset claims — the CUDA analogue
/// of `reference_allow_list()`: search may only elect ops the backend can
/// actually EXECUTE (through `KernelOp` or `HostOp`) or provably
/// FOLD (the plan-transparent class above). Labels follow house policy:
/// the egglog constructor minus the `LayoutTensorOp` prefix, nothing
/// else added or stripped.
///
/// A LOADED RUNTIME'S OWN claim set is
/// [`CudaRuntime::active_allow_list`], derived from the registry it was
/// initialized with ([`CudaRuntime::load_with_registry`]); this
/// crate-level function is the preset's, for callers with no graph in
/// hand.
pub fn cuda_allow_list() -> Vec<&'static str> {
    CudaRuntime::allow_list()
        .into_iter()
        .map(|constructor| {
            constructor
                .strip_prefix("LayoutTensorOp")
                .unwrap_or(constructor)
        })
        .collect()
}
