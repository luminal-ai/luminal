//! CUDA-lite's OWN op registry (ruling 2026-08-17: every runtime owns
//! its executable ops — structs, matchers, snippets, and codegen live
//! per-op here; the shared crate supplies only the IR traits and the
//! search machinery).
//!
//! One module per op, mirroring the reference layout: adding an op =
//! writing its module and adding its two lines here (mod + matcher).
//! FUNCTIONAL forms only in CL-1 — the runtime is out-of-place by
//! design, so the mutating/alias-safe family is deliberately absent
//! from this assembly (it arrives with the in-place ties in CL-4).
//!
//! THE REGISTRY IS A VALUE, NOT A CONSTANT (ruling 2026-09-03, #420/#422
//! rejoin Phase 2: *"you should select the allowed ops when you
//! initialize the runtime ... You should not need to edit CL in order to
//! modify this. It should be configurable."*). [`cuda_registry`] and
//! [`cuda_registry_without_cublaslt`] are just the two PRESETS this
//! crate ships; what a [`crate::CudaRuntime`] assembles, saturates, searches
//! and claims with is the `Vec<RegisteredOp>` it was handed at
//! [`crate::CudaRuntime::load_with_registry`]. A caller narrows the
//! vocabulary with [`cuda_registry_filtered`] over
//! [`RegisteredOp::label`] / [`RegisteredOp::constructor`], or extends it
//! with [`RegisteredOp::new`] — from OUTSIDE this crate, with no edit
//! here.
//!
//! WHAT IS STILL CL-ONLY: adding a KERNEL-BEARING op. Claim derivation
//! reads three classes (see [`crate::CudaRuntime::allow_list`]); the
//! matcher-only ones — plan-transparent and host-dispatchable — are
//! trait answers on the prototype and travel with a `RegisteredOp`, but
//! a row that must actually be EXECUTED needs a codegen row in
//! [`crate::kernels`], which is keyed by `TypeId` inside this crate.
//! The kernel-bearing class is matched by LABEL (the constructor minus
//! `LayoutTensorOp`, `Generic` suffix tolerated) against the kernel
//! table, while codegen is looked up by `TypeId` at execute. An outside
//! row whose label matches no kernel-table label never reaches the allow
//! list, so search refuses. An outside row that REUSES a kernel-table
//! label IS claimed and can be elected; if the op it extracts is not the
//! CL type behind that label, the plan is refused at
//! [`crate::CudaRuntime::execute`] (`no cuda codegen for <label>`) —
//! loud, never a wrong plan, but not at search. Composing an external
//! kernel superset onto Lite's codegen ("cuda heavy") is PUNTED — no
//! execution face on `RegisteredOp`, no change to the kernel table's
//! keying.

pub mod add;
pub mod cast;
pub mod constant;
pub mod cublaslt;
pub mod div;
pub mod exp;
pub mod exp2;
pub mod gather;
pub mod index_map_apply_materialize;
pub mod index_map_apply_view;
pub mod iota;
pub mod less_than;
pub mod log2;
pub mod materialize_layout_copy;
pub mod modulo;
pub mod mul;
pub mod recip;
pub mod reduce_max;
pub mod reduce_sum;
pub mod scatter;
pub mod sin;
pub mod sqrt;
pub mod trunc_div;
pub mod trunc_rem;

use luminal::layout_ir::{LayoutIrOp, OpMatcher};

/// One registered op: the matcher plus a PROTOTYPE instance of the op
/// it extracts. The prototype exists so claim derivation can read the
/// op's DECLARED EFFECTS (memory-effect predicates, alias contract,
/// DPS story) without an e-graph in hand — the allow list's
/// plan-transparent class is derived from these trait answers, never
/// from a name list. Prototype metadata fields (entries, ranks, axes)
/// take their cheapest value: the effect predicates of every op here
/// are metadata-independent.
pub struct RegisteredOp {
    pub matcher: Box<dyn OpMatcher>,
    pub prototype: Box<dyn LayoutIrOp>,
}

impl RegisteredOp {
    /// Register a matcher with a prototype of the op its `extract`
    /// produces — the public constructor, so a registry row can be built
    /// from outside this crate (see the module header for what such a
    /// row can and cannot claim).
    pub fn new(matcher: Box<dyn OpMatcher>, prototype: Box<dyn LayoutIrOp>) -> Self {
        Self { matcher, prototype }
    }

    /// This row's egglog constructor — the name the estate mints and the
    /// name the derived allow list carries.
    pub fn constructor(&self) -> &'static str {
        self.matcher.egglog_constructor()
    }

    /// This row's op LABEL: the egglog constructor minus the
    /// `LayoutTensorOp` prefix and NOTHING else (house policy — op names
    /// are IR identity: never add `Dps`, never strip `Generic`). It is
    /// the same string the prototype's own `LayoutIrOp::label` returns,
    /// which the `labels_agree_with_the_prototypes` pin states outright.
    ///
    /// So the `Generic` suffix IS part of a label: filter on
    /// `"ReduceMaxGeneric"`, not `"ReduceMax"`. (The kernel table's
    /// `label` column is a different, Generic-less vocabulary; the allow
    /// list's kernel-bearing test tolerates the difference, callers of
    /// this method should not have to guess about it.)
    pub fn label(&self) -> &'static str {
        let ctor = self.constructor();
        ctor.strip_prefix("LayoutTensorOp").unwrap_or(ctor)
    }
}

/// THE DEFAULT REGISTRY (cuBLASLt ON BY DEFAULT, ruling 2026-09-04):
/// every matcher this crate ships, paired with a prototype of the
/// (functional-form) op its `extract` produces — the kernel-bearing and
/// plan-transparent rows of [`cuda_registry_without_cublaslt`] PLUS the
/// four fixed-arity cuBLASLt marker contracts, whose execution row is a
/// HOST LIBRARY CALL (`cublasLtMatmul`), never an NVRTC kernel (the
/// third claim class — see [`crate::CudaRuntime::allow_list`]).
///
/// WHY THE MARKERS USED TO SIT BEHIND AN EXPLICIT SEAM, and why they no
/// longer do. Splicing the marker vocabulary into every assembly once
/// detonated the `view-arity-lock` (`Illegal merge attempted`) on all
/// seven minis, because that lock keyed a ROUTE fact (an apply's parent
/// rank) on a VALUE (its output class) and the collapse rule's sound
/// union `x ≡ Tᵀ(x)` put two different-parent-rank spellings in one
/// class; the check now lives on the `IndexMapLit` (entry count == rank
/// of the source-shape tag) and saturation succeeds. What kept the
/// default at the Train-2 set after that was the SEARCH BUDGET: the same
/// union is a re-description 2-cycle, and at the 2×4 harness budget the
/// sampler exhausts on unfit genomes on real graphs (they elect at
/// 12×16). Both are settled, so the markers are simply part of what a
/// default CL runtime searches with.
///
/// A caller who wants the DECOMPOSED route on purpose asks for
/// [`cuda_registry_without_cublaslt`]; see [`cuda_registry_filtered`]
/// for row-by-row narrowing (and for why the four marker rows are ONE
/// estate that must be kept or dropped together).
pub fn cuda_registry() -> Vec<RegisteredOp> {
    let mut registry = cuda_registry_without_cublaslt();
    for form in cublaslt::CublasLtForm::ALL {
        registry.push(RegisteredOp {
            matcher: Box::new(cublaslt::CublasLtMarkerMatcher { form }),
            prototype: Box::new(cublaslt::CublasLt { form, spec: None }),
        });
    }
    registry
}

/// The registry WITHOUT the cuBLASLt estate: the kernel-bearing and
/// plan-transparent rows only, so every matmul keeps its DECOMPOSED
/// spelling (the multiply/reduce chain CL's own NVRTC kernels execute).
///
/// This is for callers that want the decomposed route ON PURPOSE — the
/// marker-vs-decomposed tolerance comparisons, the kernel-count pins,
/// and anyone measuring what the library call is worth. It is NOT the
/// default: [`cuda_registry`] is.
pub fn cuda_registry_without_cublaslt() -> Vec<RegisteredOp> {
    fn reg(
        matcher: impl OpMatcher + 'static,
        prototype: impl LayoutIrOp + 'static,
    ) -> RegisteredOp {
        RegisteredOp {
            matcher: Box::new(matcher),
            prototype: Box::new(prototype),
        }
    }
    vec![
        reg(add::AddFunctionalMatcher, add::AddFunctional),
        reg(
            materialize_layout_copy::MaterializeLayoutCopyMatcher,
            materialize_layout_copy::MaterializeLayoutCopy,
        ),
        reg(sqrt::SqrtFunctionalMatcher, sqrt::SqrtFunctional),
        reg(exp::ExpFunctionalMatcher, exp::ExpFunctional),
        reg(mul::MulFunctionalMatcher, mul::MulFunctional),
        reg(div::DivFunctionalMatcher, div::DivFunctional),
        reg(
            trunc_div::TruncDivFunctionalMatcher,
            trunc_div::TruncDivFunctional,
        ),
        reg(
            trunc_rem::TruncRemFunctionalMatcher,
            trunc_rem::TruncRemFunctional,
        ),
        reg(
            reduce_sum::ReduceSumMatcher,
            reduce_sum::ReduceSum { axis: 0 },
        ),
        reg(
            reduce_max::ReduceMaxMatcher,
            reduce_max::ReduceMax { axis: 0 },
        ),
        reg(iota::IotaMatcher, iota::Iota { expr: None }),
        reg(gather::GatherMatcher, gather::Gather { rank: 1 }),
        reg(constant::ConstantMatcher, constant::Constant { value: 0.0 }),
        reg(
            scatter::ScatterFunctionalMatcher,
            scatter::ScatterFunctional { rank: 1 },
        ),
        reg(exp2::Exp2FunctionalMatcher, exp2::Exp2Functional),
        reg(log2::Log2FunctionalMatcher, log2::Log2Functional),
        reg(sin::SinFunctionalMatcher, sin::SinFunctional),
        reg(recip::RecipFunctionalMatcher, recip::RecipFunctional),
        reg(modulo::ModFunctionalMatcher, modulo::ModFunctional),
        reg(less_than::LessThanMatcher, less_than::LessThan),
        reg(cast::CastMatcher, cast::Cast),
        reg(
            index_map_apply_materialize::IndexMapApplyMaterializeMatcher,
            index_map_apply_materialize::IndexMapApplyMaterialize { entries: None },
        ),
        // M4 Phase 5: the view op is ELECTABLE on this runtime — no
        // kernel, claimed through the plan-transparent class its
        // declared effects prove (see `crate::plan_transparent`).
        reg(
            index_map_apply_view::IndexMapApplyViewMatcher,
            index_map_apply_view::IndexMapApplyView { entries: None },
        ),
    ]
}

/// THE CONFIGURATION SEAM: the FULL registry (every row this crate
/// ships, cuBLASLt markers included) narrowed by the caller's own
/// predicate. This is how a vocabulary is chosen without editing CL:
///
/// ```no_run
/// # use luminal_cuda_lite::{cuda_registry_filtered, CudaRuntime};
/// # let cx = luminal::graph::Graph::new();
/// // Everything except the max reduction:
/// let rt = CudaRuntime::load_with_registry(
///     &cx,
///     cuda_registry_filtered(|op| op.label() != "ReduceMaxGeneric"),
/// )?;
/// # Ok::<(), anyhow::Error>(())
/// ```
///
/// It starts from the FULL registry ON PURPOSE, so one predicate can
/// opt a row IN as easily as OUT — `keep` sees every row that exists.
/// The two presets remain available whole ([`cuda_registry`],
/// [`cuda_registry_without_cublaslt`]).
///
/// THE FOUR cuBLASLt MARKER ROWS ARE ONE ESTATE, not four independent
/// ones: the Base row's snippets declare all four constructors and mint
/// all four forms, so a predicate that drops Base must drop all four.
/// [`crate::CudaRuntime::load_with_registry`] refuses the other
/// combination; `active_allow_list` alone cannot tell you, because a
/// claim is derived from the prototype and says nothing about whether
/// the assembled program declares the constructor.
///
/// A PREDICATE THAT MATCHES NOTHING IS SILENT — it returns the full
/// registry, and the search then simply has the op available. Assert on
/// the resulting length (or on
/// [`crate::CudaRuntime::active_allow_list`]) if the narrowing is
/// load-bearing; labels carry their `Generic` suffix (see
/// [`RegisteredOp::label`]).
pub fn cuda_registry_filtered(keep: impl Fn(&RegisteredOp) -> bool) -> Vec<RegisteredOp> {
    cuda_registry()
        .into_iter()
        .filter(|entry| keep(entry))
        .collect()
}

/// The matcher set the DEFAULT preset assembles and extracts with — the
/// registry's matcher column, cuBLASLt markers included. An instance's
/// own column is derived from the registry it was loaded with, never
/// from this.
pub fn cuda_matchers() -> Vec<Box<dyn OpMatcher>> {
    cuda_registry()
        .into_iter()
        .map(|entry| entry.matcher)
        .collect()
}
