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
//! [`CudaRuntime::load_with_registry`] — [`cuda_registry`] and
//! [`cuda_registry_with_cublaslt`] are presets, and
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
//! - CL-1: plan-level runtime + codegen table, buildable and testable
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
//! Out-of-place by design: kernels read operand buffers and write a
//! destination that shares no byte with any simultaneously bound buffer
//! — the same alias-safety PROPERTY as the reference executor, by a
//! different mechanism. Since Phase 3 (#420/#422 rejoin) the destination
//! is a recycled slab range assigned by [`arena`] (its CONTRACT-1
//! live-range check; [`binding_check`] covers the standalone rows), NOT
//! a fresh zeroed slice: nothing is zeroed, so every kernel must write
//! every element it owns and never read its destination — the KERNEL
//! INVARIANT recorded at the top of `device.rs`. `ties` and `Anti` edges
//! are honored in the toposort order but no in-place claim is made.

pub mod arena;
pub mod binding_check;
pub mod bindings;
pub mod extractor;
/// FINALISTS (Phase 5 of the #420/#422 rejoin): a bucket's ranked
/// genomes, re-materialized one at a time under a hard filter.
pub mod finalists;
pub mod heuristic;
pub mod host_buffer;
pub mod kernels;
/// THE BUCKET LATTICE (Phase 5): best-first selection of ONE finalist
/// per bucket under a coordinate-monotone aggregate.
pub mod lattice;
pub mod layouts;
pub mod ops;
pub mod runtime;
pub mod search;

#[cfg(feature = "device")]
pub mod device;
/// ON-DEVICE CANDIDATE PROFILING (Phase 4): the search's device
/// evaluator. Device builds only — it measures, so it needs a device.
#[cfg(feature = "device")]
pub mod profile;

pub use bindings::CudaBindings;
pub use host_buffer::HostBuffer;
pub use layouts::CudaPlan;
pub use ops::{RegisteredOp, cuda_registry, cuda_registry_filtered, cuda_registry_with_cublaslt};
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
/// actually EXECUTE (a codegen row in the kernel table) or provably
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
