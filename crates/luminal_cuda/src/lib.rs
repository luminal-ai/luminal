mod binary;
mod elementwise_fusion;
mod matmul;
mod other;
mod prim;
mod quantized;
#[macro_use]
mod unary;
pub use quantized::*;

pub use cudarc::driver::CudaContext;
pub use elementwise_fusion::ElementwiseFusionCompiler;
pub use other::*;
pub use prim::PrimitiveCompiler;

#[cfg(test)]
#[macro_use]
mod tests;

use cudarc::{
    driver::{CudaFunction, CudaSlice, DeviceRepr, LaunchArgs, PushKernelArg as _},
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};
use itertools::Itertools;
use prim::CudaConstant;
use rustc_hash::FxHashMap;

use std::{collections::hash_map::DefaultHasher, fmt::Write, hash::Hasher, sync::Arc};

use luminal::{op::InputTensor, prelude::*};

/// Compile graphs to run on CUDA GPUs in supported data formats
pub type CudaCompiler<T> = (
    prim::PrimitiveCompiler<T>,
    SpecialOpsCompiler<T>,
    other::CopyCompiler<T>,
    elementwise_fusion::ElementwiseFusionCompiler<T>,
);

/// Compiler to replace cuda primops with specialized variants
pub type SpecialOpsCompiler<T> = (
    binary::SubtractionCompiler<T>,
    binary::EqualCompiler<T>,
    other::ARangeCompiler<T>,
    binary::GatherCompiler<T>,
    unary::CudaExpCompiler<T>,
    unary::CudaCosCompiler<T>,
    unary::MeanReduceCompiler<T>,
    unary::StdNormCompiler<T>,
    unary::SoftmaxCompiler<T>,
    matmul::MatMulCompiler<T>,
);

pub trait CudaFloat:
    std::fmt::Debug
    + Copy
    + Default
    + cudarc::driver::DeviceRepr
    + std::marker::Unpin
    + cudarc::driver::ValidAsZeroBits
    + 'static
{
    fn to_f32(self) -> f32;
    fn from_f32(a: f32) -> Self;
    fn is_f32() -> bool;
    fn type_name() -> &'static str;
}

impl CudaFloat for f32 {
    fn from_f32(a: f32) -> Self {
        a
    }
    fn to_f32(self) -> f32 {
        self
    }
    fn is_f32() -> bool {
        true
    }
    fn type_name() -> &'static str {
        "float"
    }
}
#[derive(Debug)]
pub struct CudaData<T>(pub CudaSlice<T>);

impl<T: DeviceRepr> Clone for CudaData<T> {
    fn clone(&self) -> Self {
        Self(self.0.try_clone().unwrap())
    }
}

impl<T: CudaFloat> Data for CudaData<T> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl CudaFloat for f16 {
    fn from_f32(a: f32) -> Self {
        f16::from_f32(a)
    }
    fn to_f32(self) -> f32 {
        self.to_f32()
    }
    fn is_f32() -> bool {
        false
    }
    fn type_name() -> &'static str {
        "__half"
    }
}

impl CudaFloat for u8 {
    fn from_f32(a: f32) -> Self {
        a as u8
    }
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn is_f32() -> bool {
        false
    }
    fn type_name() -> &'static str {
        "uint8_t"
    }
}

fn expr_to_cuda_string(expr: &Expression) -> String {
    let mut symbols = vec![];
    for term in expr.simplify().terms.read().iter() {
        let new_symbol = match term {
            Term::Num(n) => n.to_string(),
            Term::Var(c) => {
                if *c == 'z' {
                    "(int)idx".to_string()
                } else {
                    c.to_string()
                }
            }
            Term::Max => format!(
                "max((int){}, (int){})",
                symbols.pop().unwrap(),
                symbols.pop().unwrap()
            ),
            Term::Min => format!(
                "min((int){}, (int){})",
                symbols.pop().unwrap(),
                symbols.pop().unwrap()
            ),
            _ => format!(
                "({}{term:?}{})",
                symbols.pop().unwrap(),
                symbols.pop().unwrap()
            ),
        };
        symbols.push(new_symbol);
    }
    symbols.pop().unwrap()
}

fn get_idx_valid_exps(shape: ShapeTracker) -> (String, String) {
    (
        expr_to_cuda_string(&shape.index_expression()),
        expr_to_cuda_string(&shape.valid_expression()),
    )
}

fn render_dyn_dim_inputs(shapes: &[ShapeTracker]) -> (Vec<char>, String) {
    let symbols: Vec<char> = shapes
        .iter()
        .flat_map(|st| {
            st.dims()
                .into_iter()
                .chain(
                    st.padding
                        .into_iter()
                        .flat_map(|i| [i.0.into(), i.1.into()]),
                )
                .chain(st.mask.into_iter().flat_map(|i| [i.0.into(), i.1.into()]))
        })
        .flat_map(|d| d.to_symbols())
        .unique()
        .collect();
    (
        symbols.clone(),
        symbols.into_iter().fold(String::default(), |mut acc, c| {
            write!(&mut acc, ", const int {c}").unwrap();
            acc
        }),
    )
}

pub fn constant<T: CudaFloat>(num: f32) -> SelectGraph
where
    CudaData<T>: Data,
{
    let mut n = op::<CudaConstant<T>>();
    n.check(move |o, _| {
        if let Some(c) = o.as_any().downcast_ref::<CudaConstant<T>>() {
            if let luminal::op::ConstantValue::Float(f) = c.value {
                f == num
            } else {
                false
            }
        } else {
            false
        }
    });
    n
}

fn hash<T: std::hash::Hash>(obj: T) -> u64 {
    let mut hasher = DefaultHasher::new();
    obj.hash(&mut hasher);
    hasher.finish()
}

fn get_buffer_from_tensor<'a, T: CudaFloat>(tensor: &'a InputTensor) -> &'a CudaSlice<T> {
    &tensor.borrowed().downcast_ref::<CudaData<T>>().unwrap().0
}

fn input_dyn_dims(
    args: &mut LaunchArgs,
    dyn_symbols: &[char],
    dyn_map: *const FxHashMap<char, usize>,
) {
    let dyn_map = unsafe { dyn_map.as_ref().unwrap() };
    for d in dyn_symbols {
        args.arg(&dyn_map[d]);
    }
}

fn compile_and_load_kernel(mut code: String, device: &Arc<CudaContext>) -> CudaFunction {
    let name = format!("kernel_{}", hash(&code));
    code = code.replace("kernel", &name);
    let module = device
        .load_module(
            compile_ptx_with_opts(
                code,
                CompileOptions {
                    include_paths: vec!["/usr/include".into()],
                    options: vec![
                        "--gpu-architecture=compute_75".into(),
                        "--relocatable-device-code=false".into(),
                        "--std=c++14".into(),
                    ],
                    ..Default::default()
                },
            )
            .unwrap(),
        )
        .unwrap();
    module.load_function(&name).unwrap()
}

#[macro_export]
macro_rules! debug_type {
    ($t: ident) => {
        impl<T> std::fmt::Debug for $t<T> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, stringify!($t))
            }
        }
    };
}
