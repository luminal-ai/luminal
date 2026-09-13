mod matmul;
mod ops;
pub use matmul::*;
pub use ops::*;

use crate::runtime::WebGpuBuffer;
use luminal::dtype::DType;
use luminal::op::EgglogOp;
use luminal::prelude::*;
use std::sync::Arc;

pub const DYN_SLOT_COUNT: usize = 26;
pub const PARAM_SLOT_COUNT: usize = DYN_SLOT_COUNT + 2;
pub const COUNT_PARAM_SLOT: usize = DYN_SLOT_COUNT;
pub const OFFSET_PARAM_SLOT: usize = DYN_SLOT_COUNT + 1;
pub const WORKGROUP_SIZE: u32 = 256;

#[derive(Clone)]
pub struct WebGpuPipeline {
    pipelines: Vec<Arc<wgpu::ComputePipeline>>,
}

impl WebGpuPipeline {
    pub(crate) fn single(pipeline: wgpu::ComputePipeline) -> Self {
        Self {
            pipelines: vec![Arc::new(pipeline)],
        }
    }

    pub(crate) fn new(pipelines: Vec<wgpu::ComputePipeline>) -> Self {
        Self {
            pipelines: pipelines.into_iter().map(Arc::new).collect(),
        }
    }

    pub(crate) fn get(&self, index: usize) -> &wgpu::ComputePipeline {
        &self.pipelines[index]
    }
}

pub struct WebGpuEncodeContext<'a> {
    pub(crate) device: &'a wgpu::Device,
    pub(crate) encoder: &'a mut wgpu::CommandEncoder,
    pub(crate) dyn_map: &'a FxHashMap<char, usize>,
    pub(crate) max_compute_workgroups_per_dimension: u32,
}

#[derive(Debug, Clone)]
pub struct WebGpuMulInfo {
    pub shape: Vec<Expression>,
    pub a_strides: Vec<Expression>,
    pub b_strides: Vec<Expression>,
    pub output_strides: Vec<Expression>,
}

#[derive(Debug, Clone)]
pub struct WebGpuSumReduceInfo {
    pub shape: Vec<Expression>,
    pub strides: Vec<Expression>,
    pub iters: Expression,
    pub iter_stride: Expression,
}

pub trait WebGpuKernelOp: EgglogOp {
    fn compile(
        &self,
        device: &wgpu::Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<WebGpuPipeline>;

    fn infer_output_dtype(&self, input_dtypes: &[DType]) -> DType {
        input_dtypes.first().copied().unwrap_or(DType::F32)
    }

    fn output_size(&self) -> Expression;

    fn encode_compute(
        &self,
        context: &mut WebGpuEncodeContext<'_>,
        pipeline: &WebGpuPipeline,
        inputs: &[&WebGpuBuffer],
        output: &WebGpuBuffer,
        input_dtypes: &[DType],
        output_dtype: DType,
    );

    #[allow(clippy::too_many_arguments)]
    fn encode(
        &self,
        context: &mut WebGpuEncodeContext<'_>,
        pipeline: Option<&WebGpuPipeline>,
        inputs: &[&WebGpuBuffer],
        output: &WebGpuBuffer,
        _dyn_map: &FxHashMap<char, usize>,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) {
        let pipeline = pipeline.expect("compute pipeline not compiled");
        self.encode_compute(
            context,
            pipeline,
            inputs,
            output,
            input_dtypes,
            output_dtype,
        );
    }

    fn bytes_loaded(&self, _dyn_map: &FxHashMap<char, usize>) -> usize {
        0
    }

    fn bytes_stored(&self, _dyn_map: &FxHashMap<char, usize>) -> usize {
        0
    }

    fn flops(&self, _dyn_map: &FxHashMap<char, usize>) -> usize {
        0
    }

    fn mul_info(&self) -> Option<WebGpuMulInfo> {
        None
    }

    fn sum_reduce_info(&self) -> Option<WebGpuSumReduceInfo> {
        None
    }

    fn output_aliases_input(&self) -> Option<usize> {
        None
    }

    fn is_matmul(&self) -> bool {
        false
    }
}

luminal::impl_into_ops!(WebGpuKernelOp);
