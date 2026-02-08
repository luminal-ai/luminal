use std::{
    fmt::{Debug, Display},
    sync::{Arc, Mutex},
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
    /// `num_elements` is the number of f32 elements to allocate.
    fn allocate_dummy_input(&mut self, _node_index: usize, _num_elements: usize) {}
    /// Clear intermediate buffers to prepare for loading a different chunk's LLIR.
    fn clear_intermediate_buffers(&mut self) {}
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
    fn term(&self) -> (String, Vec<OpParam>);
    fn rewrites(&self) -> Vec<String> {
        vec![]
    }
    fn early_rewrites(&self) -> Vec<String> {
        vec![]
    }
    fn cleanup(&self) -> bool;
    #[allow(unused_variables)]
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
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

/// Supported dtypes
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum DType {
    /// 32-bit float (8e23m)
    #[default]
    F32,
    /// 16-bit float (5e10m)
    F16,
    /// 16-bit float (8e7m)
    Bf16,
    /// 32-bit integer
    Int,
    /// Boolean (stored as u8, 0 or 1)
    Bool,
}

impl Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl DType {
    pub fn sizeof(&self) -> usize {
        match self {
            DType::F32 | DType::Int => 4,
            DType::Bf16 | DType::F16 => 2,
            DType::Bool => 1,
        }
    }
}

/// The main HLIROp trait.
///
/// Defines an HLIROp that implements a logical operation.
pub trait HLIROp: Debug + as_any::AsAny {
    fn to_egglog(&self, inputs: &[(NodeIndex, String, ShapeTracker)]) -> String;
}

impl<T: HLIROp> HLIROp for Box<T> {
    fn to_egglog(&self, inputs: &[(NodeIndex, String, ShapeTracker)]) -> String {
        <T as HLIROp>::to_egglog(self, inputs)
    }
}
impl<T: HLIROp> HLIROp for Arc<Mutex<T>> {
    fn to_egglog(&self, inputs: &[(NodeIndex, String, ShapeTracker)]) -> String {
        <T as HLIROp>::to_egglog(&self.lock().unwrap(), inputs)
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

pub enum OpParam {
    EList,
    Expr,
    Input,
    Int,
    Float,
    Str,
    Dty,
    IList,
}

impl Debug for OpParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpParam::EList => write!(f, "EList"),
            OpParam::Expr => write!(f, "Expression"),
            OpParam::Input => write!(f, "IR"),
            OpParam::Int => write!(f, "i64"),
            OpParam::Str => write!(f, "String"),
            OpParam::Dty => write!(f, "DType",),
            OpParam::Float => write!(f, "f64"),
            OpParam::IList => write!(f, "IList"),
        }
    }
}

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
