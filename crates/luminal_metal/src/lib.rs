use std::{
    any::{Any, TypeId},
    fmt::{Debug, Write},
    ops::Deref,
    sync::Arc,
};

#[cfg(test)]
mod tests;

pub mod binary;
pub mod command_buffer;
pub mod elementwise_fusion;
pub mod matmul;
pub mod other;
pub mod prim;
pub mod quantized;
pub mod storage_buffer;
pub mod unary;

pub use metal_rs::{Buffer, Device, MTLResourceOptions};
pub use objc::rc::autoreleasepool;

use itertools::Itertools;
use metal_rs::*;
use prim::MetalConstant;
use rustc_hash::FxHashMap;

use luminal::{op::InputTensor, prelude::*};

/// Compile graphs to run on Metal-supported macOS devices in supported data formats
pub type MetalCompiler<T> = (MetalCompilerPreBuffer<T>, BufferCompilers);

/// All metal compilers coming before buffer compilers
pub type MetalCompilerPreBuffer<T> = (
    prim::PrimitiveCompiler<T>,
    SpecialOpsCompiler<T>,
    other::CopyCompiler<T>,
    elementwise_fusion::ElementwiseFusionCompiler<T>,
);

/// Compilers to share command and storage buffers
pub type BufferCompilers = (
    command_buffer::CommandBufferCompiler,
    storage_buffer::StorageBufferCompiler,
);

/// Compiler to replace metal ops with specialized variants
pub type SpecialOpsCompiler<T> = (
    binary::MetalSubtractionCompiler<T>,
    binary::MetalEqualCompiler<T>,
    other::ARangeCompiler<T>,
    binary::MetalGatherCompiler<T>,
    unary::MetalExpCompiler<T>,
    unary::MetalCosCompiler<T>,
    unary::MeanReduceCompiler<T>,
    unary::StdNormCompiler<T>,
    matmul::MetalMatMulCompiler<T>,
);

#[derive(Debug, Clone)]
pub struct MetalBuffer(pub Buffer);

impl Deref for MetalBuffer {
    type Target = Buffer;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Data for MetalBuffer {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

pub trait MetalFloat: Copy + Debug + PartialEq + 'static + Default {
    fn to_f32(self) -> f32;
    fn from_f32(a: f32) -> Self;
    fn is_f32() -> bool;
    fn type_name() -> &'static str;
}

// Quantization types

pub trait MetalQuantizationType {
    type MatmulCompiler;
}

/// 8-bit quantization. Equivalent to the ggml Q8_0 datatype
pub struct Q8_0;

impl MetalQuantizationType for Q8_0 {
    type MatmulCompiler = matmul::MetalMatMulCompiler<f16>;
}

impl MetalQuantizationType for f32 {
    type MatmulCompiler = matmul::MetalMatMulCompiler<Self>;
}

impl MetalQuantizationType for f16 {
    type MatmulCompiler = matmul::MetalMatMulCompiler<Self>;
}

// Main metal dtypes

impl MetalFloat for f32 {
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

impl MetalFloat for f16 {
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
        "half"
    }
}

pub trait MetalKernel: Debug {
    /// Annotate the buffer sizes of the intermediate buffers
    fn intermediate_buffer_sizes(&self, _: &[ShapeTracker]) -> Vec<Expression> {
        vec![]
    }
    /// Annotate the buffer sizes of the output buffers
    fn output_buffer_sizes(&self, input_shapes: &[ShapeTracker]) -> Vec<Expression>;
    /// Set up the kernel on the buffer
    fn metal_forward(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        command_buffer: &CommandBufferRef,
        intermediate_buffers: &[&Buffer],
        output_buffers: &[&Buffer],
    );
    fn without_command_buffer(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        intermediate_buffers: &[&Buffer],
        output_buffers: &[&Buffer],
    ) {
        let dev = Device::system_default().unwrap();
        let queue = dev.new_command_queue();
        let command_buffer = queue.new_command_buffer();
        self.metal_forward(inputs, command_buffer, intermediate_buffers, output_buffers);
    }
    fn without_storage_buffers(
        &self,
        inputs: &[(&Buffer, ShapeTracker)],
        command_buffer: &CommandBufferRef,
        dyn_map: &FxHashMap<char, usize>,
    ) -> Vec<Buffer> {
        let dev = Device::system_default().unwrap();
        // Allocate storage buffers
        let inp_shapes = inputs.iter().map(|(_, s)| *s).collect::<Vec<_>>();
        let intermediate_buffers = self
            .intermediate_buffer_sizes(&inp_shapes)
            .into_iter()
            .map(|n| {
                dev.new_buffer(
                    n.exec(dyn_map).unwrap() as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            })
            .collect::<Vec<_>>();
        let intermediate_buffers_ref = intermediate_buffers.iter().collect::<Vec<_>>();
        let output_buffers = self
            .output_buffer_sizes(&inp_shapes)
            .into_iter()
            .map(|n| {
                dev.new_buffer(
                    n.exec(dyn_map).unwrap() as u64,
                    MTLResourceOptions::StorageModeShared,
                )
            })
            .collect::<Vec<_>>();
        let output_buffers_ref = output_buffers.iter().collect::<Vec<_>>();
        self.metal_forward(
            inputs,
            command_buffer,
            &intermediate_buffers_ref,
            &output_buffers_ref,
        );
        output_buffers
    }
}

#[derive(Clone)]
pub struct MetalKernelWrapper(pub Arc<Box<dyn MetalKernel>>);
impl Debug for MetalKernelWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetalKernelWrapper")
    }
}

impl Default for MetalKernelWrapper {
    fn default() -> Self {
        Self(Arc::new(Box::new(())))
    }
}

impl Deref for MetalKernelWrapper {
    type Target = Box<dyn MetalKernel>;
    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl MetalKernel for () {
    fn output_buffer_sizes(&self, _: &[ShapeTracker]) -> Vec<Expression> {
        vec![]
    }
    fn metal_forward(
        &self,
        _: &[(&Buffer, ShapeTracker)],
        _: &CommandBufferRef,
        _: &[&Buffer],
        _: &[&Buffer],
    ) {
    }
}

fn compile_lib(device: &Device, source: &str) -> Library {
    let mut source = source.replace("BF16.H", include_str!("kernels/bf16.h"));
    source = source.replace("DEFINES.H", include_str!("kernels/defines.h"));
    source = source.replace("GEMM.H", include_str!("kernels/gemm.h"));
    source = source.replace("UTILS.H", include_str!("kernels/utils.h"));
    let options = CompileOptions::new();
    options.set_fast_math_enabled(true);
    device.new_library_with_source(&source, &options).unwrap()
}

fn select_function_from_lib(
    lib: &Library,
    function: &str,
    device: &Device,
) -> ComputePipelineState {
    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor
        .set_compute_function(Some(&lib.get_function(function, None).unwrap()));
    device
        .new_compute_pipeline_state_with_function(
            pipeline_state_descriptor.compute_function().unwrap(),
        )
        .unwrap()
}

fn compile_function(name: &str, code: &str, device: &Device) -> ComputePipelineState {
    let library = compile_lib(device, &code.replace("inf", "INFINITY"));
    select_function_from_lib(&library, name, device)
}

fn is<T: Any>(type_id: TypeId) -> bool {
    type_id == TypeId::of::<T>()
}

trait DispatchNElements {
    fn dispatch_1d(&self, n: usize);
}

impl DispatchNElements for ComputeCommandEncoderRef {
    fn dispatch_1d(&self, n: usize) {
        self.dispatch_thread_groups(
            MTLSize {
                width: n.div_ceil(1024) as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1024,
                height: 1,
                depth: 1,
            },
        );
    }
}

#[allow(dead_code)]
trait SetInt {
    fn set_i32(&self, index: usize, value: i32);
    fn set_u32(&self, index: usize, value: u32);
    fn set_f32(&self, index: usize, value: f32);
    fn set_i64(&self, index: usize, value: i64);
    fn set_u64(&self, index: usize, value: u64);
    fn set_f64(&self, index: usize, value: f64);
}

impl SetInt for ComputeCommandEncoderRef {
    fn set_i32(&self, index: usize, value: i32) {
        self.set_bytes(
            index as u64,
            std::mem::size_of::<i32>() as u64,
            &value as *const i32 as *const _,
        );
    }
    fn set_u32(&self, index: usize, value: u32) {
        self.set_bytes(
            index as u64,
            std::mem::size_of::<u32>() as u64,
            &value as *const u32 as *const _,
        );
    }
    fn set_f32(&self, index: usize, value: f32) {
        self.set_bytes(
            index as u64,
            std::mem::size_of::<f32>() as u64,
            &value as *const f32 as *const _,
        );
    }
    fn set_i64(&self, index: usize, value: i64) {
        self.set_bytes(
            index as u64,
            std::mem::size_of::<i64>() as u64,
            &value as *const i64 as *const _,
        );
    }
    fn set_u64(&self, index: usize, value: u64) {
        self.set_bytes(
            index as u64,
            std::mem::size_of::<u64>() as u64,
            &value as *const u64 as *const _,
        );
    }
    fn set_f64(&self, index: usize, value: f64) {
        self.set_bytes(
            index as u64,
            std::mem::size_of::<f64>() as u64,
            &value as *const f64 as *const _,
        );
    }
}

fn input_dyn_dims(
    dyn_symbols: &[char],
    dyn_map: &FxHashMap<char, usize>,
    encoder: &ComputeCommandEncoderRef,
    index: usize,
) {
    for (i, s) in dyn_symbols.iter().enumerate() {
        encoder.set_u32(i + index, dyn_map[s] as u32);
    }
}

fn render_dyn_dim_inputs(shapes: &[ShapeTracker], offset: usize) -> (Vec<char>, String) {
    let symbols: Vec<char> = shapes
        .iter()
        .flat_map(|st| {
            st.dims()
                .into_iter()
                .chain(st.padding.into_iter().flat_map(|i| [i.0, i.1]))
                .chain(st.mask.into_iter().flat_map(|i| [i.0, i.1]))
        })
        .flat_map(|d| d.to_symbols())
        .unique()
        .collect();
    (
        symbols.clone(),
        symbols
            .into_iter()
            .enumerate()
            .fold(String::default(), |mut acc, (i, c)| {
                write!(&mut acc, ", device int& {c} [[buffer({})]]", i + offset).unwrap();
                acc
            }),
    )
}

fn expr_to_metal_string(expr: &Expression) -> String {
    let mut symbols = vec![];
    for term in expr.terms.read().clone() {
        let new_symbol = match term {
            Term::Num(n) => n.to_string(),
            Term::Var(c) => {
                if c == 'z' {
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
            Term::Lt => format!(
                "(int)({} < {})",
                symbols.pop().unwrap(),
                symbols.pop().unwrap()
            ),
            Term::Gte => format!(
                "(int)({} >= {})",
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
        expr_to_metal_string(&shape.index_expression()),
        expr_to_metal_string(&shape.valid_expression()),
    )
}

fn get_buffer_from_tensor<'a>(tensor: &'a InputTensor) -> &'a MetalBuffer {
    tensor
        .borrowed()
        .downcast_ref::<MetalBuffer>()
        .expect("Tensor does not contain a metal buffer")
}

pub fn constant<T: MetalFloat>(num: f32) -> SelectGraph {
    let mut n = op::<MetalConstant<T>>();
    n.check(move |o, _| {
        if let Some(c) = o.as_any().downcast_ref::<MetalConstant<T>>() {
            if let luminal::op::ConstantValue::Float(f) = c.0 {
                (f - num).abs() < 1e-3
            } else {
                false
            }
        } else {
            false
        }
    });
    n
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
