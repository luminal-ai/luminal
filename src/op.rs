use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

use crate::prelude::*;
use as_any::{AsAny, Downcast};
use rustc_hash::FxHashMap;

pub trait Runtime {
    type Ops: IntoEgglogOp;
    type CompileArg;
    type ExecReturn;
    type ProfileMetric: PartialOrd + Clone + Debug;
    fn initialize(arg: Self::CompileArg) -> Self;
    fn load_llir(&mut self, llir_graph: &LLIRGraph);
    fn execute(&mut self, dyn_map: &FxHashMap<char, usize>) -> Self::ExecReturn;
    fn profile(
        &mut self,
        llir_graph: &LLIRGraph,
        dyn_map: &FxHashMap<char, usize>,
        trials: usize,
    ) -> (Self::ProfileMetric, String);
    /// Allocate a dummy input buffer for a boundary node during per-chunk profiling.
    /// `node_index` is the HLIR node index used in the Input op's `node` field.
    /// `num_bytes` is the number of bytes to allocate.
    fn allocate_dummy_input(&mut self, _node_index: usize, _num_bytes: usize) {}
    /// Check if an HLIR buffer already exists for a given node index.
    fn has_hlir_buffer(&self, _node_index: usize) -> bool {
        false
    }
    /// Clear intermediate buffers to prepare for loading a different chunk's LLIR.
    fn clear_intermediate_buffers(&mut self) {}
    /// Total bytes of intermediate buffers currently allocated.
    fn intermediate_buffer_bytes(&self) -> usize {
        0
    }
    /// Check if the most recent execution produced NaN in any output buffer.
    /// Used by the search to reject NaN-producing graph variants.
    fn has_nan_outputs(&self, _llir_graph: &LLIRGraph, _dyn_map: &FxHashMap<char, usize>) -> bool {
        false
    }
    /// Load multiple compiled LLIR graphs, one per bucket combination.
    /// Each entry is (bucket_indices, representative_dyn_map, stitched_llir).
    /// The runtime dispatches between them in execute() based on dyn_map values.
    fn load_llir_buckets(
        &mut self,
        _dim_buckets: &FxHashMap<char, Vec<DimBucket>>,
        bucket_llirs: &[BucketLLIR],
    ) {
        if bucket_llirs.len() == 1 {
            self.load_llir(&bucket_llirs[0].2);
        } else {
            panic!("This runtime does not support bucketed compilation");
        }
    }
}

/// Optional runtime instrumentation for collecting execution statistics.
pub trait RuntimeStats: Runtime {
    fn execute_with_stats(&mut self, dyn_map: &FxHashMap<char, usize>) -> Option<ExecutionStats>;
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
    fn rewrites(&self) -> Vec<crate::egglog_utils::api::Rule> {
        vec![]
    }
    fn early_rewrites(&self) -> Vec<crate::egglog_utils::api::Rule> {
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
}

impl<T: HLIROp> HLIROp for Box<T> {
    fn to_egglog(&self, inputs: &[(NodeIndex, String)]) -> String {
        <T as HLIROp>::to_egglog(self, inputs)
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
                || op.type_name().contains("Output"),
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
