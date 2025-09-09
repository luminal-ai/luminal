pub mod codegen;
pub mod debug;
pub mod extract;
pub mod run;
pub mod translate;
pub mod utils;

#[cfg(test)]
mod tests;

#[cfg(feature = "cuda")]
use itertools::Itertools;
use luminal::prelude::*;
#[cfg(feature = "cuda")]
use luminal_cuda::CudaData;
#[cfg(feature = "metal")]
use objc2::{rc::Retained, runtime::ProtocolObject};
#[cfg(feature = "metal")]
use objc2_metal::{MTLBuffer, MTLDevice, MTLFunction};
#[cfg(feature = "cuda")]
use std::fs::File;
#[cfg(feature = "cuda")]
use std::io::Write;
use std::{collections::HashMap, fmt::Debug};

#[cfg(feature = "metal")]
pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
#[cfg(feature = "metal")]
pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
#[cfg(feature = "metal")]
pub type Function = Retained<ProtocolObject<dyn MTLFunction>>;

#[derive(Clone, PartialEq, Eq)]
pub enum GPUArch {
    CUDA,
    Metal(HashMap<usize, &'static str>),
}

impl GPUArch {
    fn metal_buffer_type(&self, var: usize) -> &'static str {
        match self {
            Self::Metal(m) => m.get(&var).copied().unwrap_or(""),
            _ => "",
        }
    }

    fn add_metal_buffer_type(&mut self, var: usize, buf_type: &'static str) {
        if let Self::Metal(m) = self {
            m.insert(var, buf_type);
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Kernel {
    pub code: String,
    // launch params
    pub grid: (Expression, Expression, Expression),
    pub threadblock: (Expression, Expression, Expression),
    pub smem: Expression, // sizes of required shared memory buffers
    pub outputs: Vec<Expression>,
}

#[derive(Clone, Debug)]
pub enum GMEMBuffer {
    PrevKernel { kernel: usize, output: usize },
    Input { node: NodeIndex },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphTerm {
    GMEM {
        // Signifies global memory
        label: String,
    },
    LoopIn {
        range: Expression,
        stride: Expression,
        marker: String,
    },
    LoopOut {
        range: Expression,
        stride: Expression,
        marker: String,
    },
    Add,
    Mul,
    Max,
    Exp2,
    Log2,
    Recip,
    Sin,
    Neg,
    Sqrt,
    LessThan,
    Mod,
    SMEM,     // Signifies shared memory
    SMEMLoad, // Takes in an smem pointer and a gmem pointer, copies the gmem element to smem and returns the smem pointer
    SMEMRead, // Takes in an smem pointer and an smemload, returns the smem pointer
    Custom(Kernel),
    Diff(String), // Diff a buffer
    Break,
    TCMatmul {
        a_k_stride: Expression,
        b_k_stride: Expression,
        a_inner_stride: Expression,
        b_inner_stride: Expression,
        c_inner_stride: Expression,
        k_outer_loops: Expression,
    },
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct CompatKernel(Kernel, *mut Graph);

#[cfg(feature = "cuda")]
impl Operator for CompatKernel {
    fn process(&mut self, inputs: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        use cudarc::driver::{LaunchConfig, PushKernelArg};
        use luminal_cuda::CudaData;
        let dyn_vars = &unsafe { self.1.as_ref().unwrap() }.dyn_map;
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();

        let ptx = cudarc::nvrtc::compile_ptx(&self.0.code).unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let kernel = module.load_function("kernel_name").unwrap();
        let mut builder = stream.launch_builder(&kernel);

        // set inputs
        for input in inputs
            .iter()
            .map(|(b, _)| b.borrowed().downcast_ref::<CudaData<f32>>().unwrap())
        {
            builder.arg(&input.0);
        }

        // set output
        let mut out = self
            .0
            .outputs
            .iter()
            .map(|s| {
                stream
                    .alloc_zeros::<f32>(s.exec(dyn_vars).unwrap())
                    .unwrap()
            })
            .collect_vec();
        for o in &mut out {
            builder.arg(o);
        }

        // Set dispatch
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (
                    self.0.grid.0.exec(dyn_vars).unwrap() as u32,
                    self.0.grid.1.exec(dyn_vars).unwrap() as u32,
                    self.0.grid.2.exec(dyn_vars).unwrap() as u32,
                ),
                block_dim: (
                    self.0.threadblock.0.exec(dyn_vars).unwrap() as u32,
                    self.0.threadblock.1.exec(dyn_vars).unwrap() as u32,
                    self.0.threadblock.2.exec(dyn_vars).unwrap() as u32,
                ),
                shared_mem_bytes: self.0.smem.exec(dyn_vars).unwrap() as u32,
            })
        }
        .unwrap();

        out.into_iter().map(|b| Tensor::new(CudaData(b))).collect()
    }
}

#[cfg(feature = "metal")]
impl Operator for CompatKernel {
    fn process(&mut self, _inputs: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        todo!()
    }
}

pub fn custom_kernel(
    inputs: &[GraphTensor],
    kernel: Kernel,
    output_shape: impl ToShape,
    cx: &mut Graph,
) -> GraphTensor {
    let graph_ref: *mut Graph = cx;
    let mut kernel_op = cx.add_op(CompatKernel(kernel, graph_ref));
    for input in inputs {
        kernel_op = kernel_op.input(input.id, 0, input.shape);
    }
    let kernel_op = kernel_op.finish();
    GraphTensor::from_id(kernel_op, ShapeTracker::new(output_shape), cx)
}

#[derive(Debug)]
pub struct Diff {
    name: String,
}

#[cfg(feature = "cuda")]
impl Operator for Diff {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        // Dump
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let buffer = inp[0].0.borrowed().downcast_ref::<CudaData<f32>>().unwrap();
        let data: Vec<f32> = stream.memcpy_dtov(&buffer.0).unwrap();
        let mut file = File::create(format!("{}.bin", self.name)).unwrap();
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<f32>(),
            )
        };
        file.write_all(bytes).unwrap();
        vec![Tensor::new(buffer.clone())]
    }
}

#[cfg(feature = "metal")]
impl Operator for Diff {
    fn process(&mut self, _inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        todo!()
    }
}

pub trait GT2 {
    fn diff2(self, name: impl ToString) -> Self;
    fn graph_break(self) -> Self;
}

impl GT2 for GraphTensor {
    fn diff2(mut self, name: impl ToString) -> Self {
        if !self.shape.is_contiguous() {
            self = self.contiguous();
        }
        let id = self
            .graph()
            .add_op(Diff {
                name: name.to_string(),
            })
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(id, self.shape, self.graph_ref)
    }

    fn graph_break(mut self) -> Self {
        if !self.shape.is_contiguous() {
            self = self.contiguous();
        }
        let id = self
            .graph()
            .add_op(GraphBreak)
            .input(self.id, 0, self.shape)
            .finish();
        GraphTensor::from_id(id, self.shape, self.graph_ref)
    }
}

#[derive(Debug)]
pub struct GraphBreak;

impl Operator for GraphBreak {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        inp.into_iter().map(|i| i.0.cloned()).collect()
    }
}
