use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

use crate::prelude::*;
use crate::search::SearchSpace;
use as_any::{AsAny, Downcast};
use rustc_hash::FxHashMap;

/// A backend: the ops and rewrites that lower HLIR into its LLIR, a search
/// strategy over the saturated e-graphs core hands it, and execution of the
/// programs it selects.
pub trait Runtime {
    type Ops: IntoEgglogOp;
    type CompileArg;
    type ExecReturn;
    /// Whether HLIR ops are deleted from the e-graph after saturation so
    /// only lowered LLIR is extractable. The reference runtime, whose LLIR
    /// *is* HLIR, sets this to `false`.
    const CLEANUP_HLIR: bool = true;
    /// Backend-provided egglog layers that run after the normal full-egraph
    /// cleanup schedule. Core keeps this empty; runtimes can use it for
    /// backend-specific analyses and cleanup passes without adding those rules
    /// to Luminal core.
    fn late_egglog_passes(
        _ops: &[Arc<Box<dyn EgglogOp>>],
        _options: &crate::graph::CompileOptions,
        _dyn_map: &DynMap,
    ) -> Vec<crate::egglog_utils::LateEgglogPass>
    where
        Self: Sized,
    {
        vec![]
    }
    /// Backend-provided egglog text spliced after the op constructor and
    /// op-owned declarations, before the rewrite rules. Core keeps this empty;
    /// runtimes can use it for backend-wide program text that is not naturally
    /// owned by one registered op, without adding it to Luminal core.
    fn extra_egglog() -> String
    where
        Self: Sized,
    {
        String::new()
    }
    fn initialize(arg: Self::CompileArg) -> Self;
    /// Choose one program per bucket of `space` — by any strategy — and leave
    /// the runtime ready to [`Runtime::execute`].
    ///
    /// `dyn_map` is the graph's dyn map after `options.search_dims`; the
    /// representative values of each bucket come from
    /// [`SearchSpace::bucket_contexts`]. `rng` is the caller's (possibly
    /// seeded) RNG. Core ships the building blocks in [`crate::search`]:
    /// [`crate::search::extract_one`] for a runtime with nothing to rank on,
    /// [`crate::search::GeneticSearch`] and friends for a runtime that
    /// profiles, and [`crate::search::genetic_search`] composing them into
    /// the stock strategy.
    fn compile(
        &mut self,
        space: &SearchSpace,
        dyn_map: &DynMap,
        options: &crate::graph::CompileOptions,
        rng: &mut dyn rand::RngCore,
    );
    fn selected_schedule(&self) -> Option<crate::graph::SelectedSchedule> {
        None
    }
    /// Load one LLIR graph as the executable. [`Runtime::compile`] normally
    /// loads what it selects; this is the direct path for callers that
    /// already hold an LLIR graph.
    fn load_llir(&mut self, llir_graph: &LLIRGraph);
    fn load_llir_buckets(
        &mut self,
        _dim_buckets: &FxHashMap<Symbol, Vec<crate::graph::DimBucket>>,
        bucket_llirs: &[crate::graph::BucketLLIR],
    ) {
        assert_eq!(
            bucket_llirs.len(),
            1,
            "runtime does not support LLIR buckets"
        );
        self.load_llir(&bucket_llirs[0].2);
    }
    fn execute(&mut self, dyn_map: &DynMap) -> Self::ExecReturn;
}

/// Optional runtime instrumentation for collecting execution statistics.
pub trait RuntimeStats: Runtime {
    fn execute_with_stats(&mut self, dyn_map: &DynMap) -> Option<ExecutionStats>;
}

/// Shared early-stop predicate for duration-metric runtimes: true once a
/// candidate's running mean trial time exceeds `best * factor`, i.e. the
/// candidate has already lost by at least the configured margin and further
/// trials can only refine a metric that is out of contention.
pub fn early_stop_exceeded(
    mean: std::time::Duration,
    best: std::time::Duration,
    factor: f64,
) -> bool {
    mean.as_secs_f64() > best.as_secs_f64() * factor
}

/// Timing method used for execution statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimingMethod {
    /// Device-side timing (e.g. GPU timestamps / CUDA events).
    DeviceTimestamp,
    /// Host-side wall-clock timing.
    /// Includes any host/device synchronization overhead.
    #[default]
    WallClock,
}

impl std::fmt::Display for TimingMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimingMethod::DeviceTimestamp => write!(f, "Device"),
            TimingMethod::WallClock => write!(f, "Wall"),
        }
    }
}

/// Detailed execution statistics from a single run.
///
/// This struct captures basic counters and timing.
#[derive(Debug, Clone, Default)]
pub struct ExecutionStats {
    /// Execution time in microseconds.
    pub execution_time_us: f64,
    /// Total bytes read.
    pub bytes_loaded: usize,
    /// Total bytes written.
    pub bytes_stored: usize,
    /// Total floating-point operations.
    pub flops: usize,
    /// Timing method used for this measurement.
    pub timing_method: TimingMethod,
}

impl ExecutionStats {
    pub fn new(
        execution_time_us: f64,
        bytes_loaded: usize,
        bytes_stored: usize,
        flops: usize,
    ) -> Self {
        Self {
            execution_time_us,
            bytes_loaded,
            bytes_stored,
            flops,
            timing_method: TimingMethod::DeviceTimestamp,
        }
    }

    /// Create new execution stats with explicit timing method.
    pub fn with_timing_method(
        execution_time_us: f64,
        bytes_loaded: usize,
        bytes_stored: usize,
        flops: usize,
        timing_method: TimingMethod,
    ) -> Self {
        Self {
            execution_time_us,
            bytes_loaded,
            bytes_stored,
            flops,
            timing_method,
        }
    }

    /// Total bytes transferred (loaded + stored).
    pub fn total_bytes(&self) -> usize {
        self.bytes_loaded + self.bytes_stored
    }

    pub fn merge(&mut self, other: &ExecutionStats) {
        self.execution_time_us += other.execution_time_us;
        self.bytes_loaded += other.bytes_loaded;
        self.bytes_stored += other.bytes_stored;
        self.flops += other.flops;
    }
}

impl std::fmt::Display for ExecutionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ExecutionStats {{ time: {:.2}µs ({}), bytes: {:.2}MB, flops: {:.2}M }}",
            self.execution_time_us,
            self.timing_method,
            self.total_bytes() as f64 / 1_000_000.0,
            self.flops as f64 / 1_000_000.0
        )
    }
}

pub trait EgglogOp: Debug {
    fn sort(&self) -> crate::egglog_utils::api::SortDef;

    /// Shared egglog declarations required by this op's rewrites. These are
    /// emitted once, before any rewrite text, so relations/functions shared by
    /// multiple ops do not depend on tuple registration order. Identical
    /// declaration strings are deduplicated while preserving first-seen order.
    fn egglog_declarations(&self) -> Vec<String> {
        vec![]
    }

    fn rewrites(&self) -> Vec<crate::egglog_utils::api::Rule> {
        vec![]
    }
    fn cleanup(&self) -> bool;

    /// Additional IR datatype variants this op needs (e.g. `"(ConsumedBuffer IR)"`).
    /// These are injected into the IR datatype definition.
    fn ir_defs(&self) -> Vec<String> {
        vec![]
    }

    /// Number of IR inputs this op takes (from IList).
    /// Used by generic IList walking during extraction.
    fn n_inputs(&self) -> usize {
        0
    }

    /// Extract this op from the egraph.
    /// - `kind_children`: metadata fields from OpKind enode (shapes, strides, dtypes, etc.)
    /// - `input_enodes`: IR inputs from IList, already walked and resolved
    #[allow(unused_variables)]
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        panic!("Extraction not implemented for {self:?}!");
    }
}

crate::impl_into_ops!(EgglogOp);

pub trait CustomOp: Debug {
    fn to_llir_op(&self) -> LLIROp;
}

/// The main HLIROp trait.
///
/// Defines an HLIROp that implements a logical operation.
pub trait HLIROp: Debug + Display + as_any::AsAny {
    fn to_egglog(&self, inputs: &[(NodeIndex, String)]) -> String;

    /// Return this operation's concrete output dtype from the concrete dtypes
    /// of its ordered inputs.
    ///
    /// Most logical operations preserve the dtype of their first input.
    /// Operations with different dtype semantics override this method. Keeping
    /// the rule on the operation makes graph transforms such as loop rolling
    /// consume the same type contract as the op itself instead of maintaining
    /// a second opcode-specific inference table.
    fn output_dtype(&self, input_dtypes: &[DType]) -> DType {
        *input_dtypes
            .first()
            .unwrap_or_else(|| panic!("{self} has no input and does not declare an output dtype"))
    }
}

impl<T: HLIROp> HLIROp for Box<T> {
    fn to_egglog(&self, inputs: &[(NodeIndex, String)]) -> String {
        <T as HLIROp>::to_egglog(self, inputs)
    }

    fn output_dtype(&self, input_dtypes: &[DType]) -> DType {
        <T as HLIROp>::output_dtype(self, input_dtypes)
    }
}

#[derive(Debug, Clone)]
pub struct LLIROp(Arc<Box<dyn DialectOpTrait>>);

impl LLIROp {
    /// Store an op in a generic LLIR op. **Make sure to erase type into your dialect trait!** i.e. `as Box<dyn BlockOp>`
    pub fn new<T: ?Sized>(op: Box<T>) -> Self
    where
        Box<T>: Debug + 'static,
    {
        assert!(
            op.type_name().contains("dyn")
                || op.type_name().contains("Input")
                || op.type_name().contains("Output")
                || op.type_name().contains("LoopStart")
                || op.type_name().contains("LoopEnd")
                || op.type_name().contains("LoopInput")
                || op.type_name().contains("LoopOutput"),
            "op types must be erased into dialect traits for dialect casting to work!"
        );
        Self(Arc::new(Box::new(DialectOp::new(op))))
    }

    pub fn to_dialect<T: ?Sized + 'static>(&self) -> Option<&Arc<Box<T>>> {
        (**self.0).downcast_ref::<DialectOp<Box<T>>>().map(|i| &i.0)
    }

    pub fn to_op<T: 'static>(&self) -> Option<&T> {
        (**self.0)
            .downcast_ref::<DialectOp<Box<T>>>()
            .map(|d| &**d.0)
    }
}

impl std::fmt::Display for LLIROp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug)]
struct DialectOp<T>(pub Arc<T>);

impl<T> DialectOp<T> {
    pub fn new(op: T) -> Self {
        Self(Arc::new(op))
    }
}

impl<T: Debug + 'static> DialectOpTrait for DialectOp<T> {}

pub trait DialectOpTrait: AsAny + Debug {}

#[macro_export]
macro_rules! __impl_tuple_into_dyn_arcbox_concat_arity {
    ($tr:ident; $($T:ident),+ $(,)?) => {
        $crate::paste!{
        impl<$($T),+> [<Into $tr>] for ($($T,)+)
        where
            $(
                $T: [<Into $tr>],
            )+
        {
            #[inline]
            fn append_into(
                out: &mut ::std::vec::Vec<
                    ::std::sync::Arc<::std::boxed::Box<dyn $tr + 'static>>
                >
            ) {
                $(
                    <$T as [<Into $tr>]>::append_into(out);
                )+
            }
        }
        }
    };
}

#[macro_export]
macro_rules! impl_into_ops {
    ($tr:ident) => {
        $crate::paste!{
        pub trait [<Into $tr>] {
            fn append_into(
                out: &mut ::std::vec::Vec<
                    ::std::sync::Arc<::std::boxed::Box<dyn $tr + 'static>>
                >
            );

            #[inline]
            fn into_vec() -> ::std::vec::Vec<
                ::std::sync::Arc<::std::boxed::Box<dyn $tr + 'static>>
            > {
                let mut out = ::std::vec::Vec::new();
                Self::append_into(&mut out);
                out
            }
        }

        // base
        impl [<Into $tr>] for () {
            #[inline]
            fn append_into(
                _out: &mut ::std::vec::Vec<
                    ::std::sync::Arc<::std::boxed::Box<dyn $tr + 'static>>
                >
            ) {}
        }

        // leaf: any concrete op type
        impl<T> [<Into $tr>] for T
        where
            T: $tr + ::std::default::Default + 'static,
        {
            #[inline]
            fn append_into(
                out: &mut ::std::vec::Vec<
                    ::std::sync::Arc<::std::boxed::Box<dyn $tr + 'static>>
                >
            ) {
                out.push(::std::sync::Arc::new(::std::boxed::Box::new(
                    <T as ::std::default::Default>::default(),
                )));
            }
        }
        }

        // tuple concatenation impls (extend arity list as needed)
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I, J);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I, J, K);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I, J, K, L);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I, J, K, L, M);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I, J, K, L, M, N);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I, J, K, L, M, N, O);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y);
        $crate::__impl_tuple_into_dyn_arcbox_concat_arity!($tr; A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z);
    };
}

#[cfg(test)]
mod early_stop_tests {
    use super::early_stop_exceeded;
    use std::time::Duration;

    #[test]
    fn test_early_stop_exceeded() {
        let best = Duration::from_millis(5);
        // 2x cutoff: 10ms mean is at the boundary, not over it.
        assert!(!early_stop_exceeded(Duration::from_millis(10), best, 2.0));
        assert!(early_stop_exceeded(Duration::from_millis(11), best, 2.0));
        // A candidate faster than best never stops early.
        assert!(!early_stop_exceeded(Duration::from_millis(4), best, 2.0));
        // Factor 1.0 stops anything slower than best.
        assert!(early_stop_exceeded(Duration::from_millis(6), best, 1.0));
    }
}
