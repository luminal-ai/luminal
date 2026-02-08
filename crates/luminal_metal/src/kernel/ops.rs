use super::MetalKernelOp;
use luminal::{
    egglog_utils::SerializedEGraph, op::OpParam::*, op::*, prelude::*, shape::flatten_mul_strides,
};
use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLSize};

pub type MetalOps = (
    // Unary ops
    MetalExp2,
    MetalLog2,
    MetalSin,
    MetalSqrt,
    MetalRecip,
    // Binary ops
    MetalAdd,
    MetalMul,
    MetalMod,
    MetalLessThan,
    // Reduce ops
    MetalSumReduce,
    MetalMaxReduce,
    // Data ops
    MetalConstant,
    MetalIota,
    MetalGather,
);

fn compile_shader(device: &Device, source: &str, function_name: &str) -> ComputePipelineState {
    let library = device
        .new_library_with_source(source, &metal::CompileOptions::new())
        .expect("Failed to compile Metal shader");
    let function = library
        .get_function(function_name, None)
        .expect("Failed to get function from library");
    device
        .new_compute_pipeline_state_with_function(&function)
        .expect("Failed to create compute pipeline state")
}

macro_rules! metal_unary_op {
    ($name:ident, $op_name:expr, $metal_op:expr) => {
        #[derive(Debug, Default, Clone)]
        pub struct $name {
            shape: Vec<Expression>,
            input_strides: Vec<Expression>,
            output_strides: Vec<Expression>,
        }

        impl EgglogOp for $name {
            fn term(&self) -> (String, Vec<OpParam>) {
                ($op_name.to_string(), vec![EList, Input, EList, EList])
            }

            fn rewrites(&self) -> Vec<String> {
                vec![format!(
                    r#"(rule
                        ((= ?e ({} ?shape ?x ?x_stride ?out_stride))
                         (= ?dt (dtype ?x)))
                        ((let ?me ({} ?shape ?x ?x_stride ?out_stride))
                         (union ?e ?me)
                         (set (dtype ?me) ?dt))
                    )"#,
                    $op_name.replace("Metal", ""),
                    $op_name
                )]
            }

            fn cleanup(&self) -> bool {
                false
            }

            fn extract<'a>(
                &'a self,
                egraph: &'a SerializedEGraph,
                children: &[&'a ENodeId],
                list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
                expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
            ) -> (LLIROp, Vec<&'a ENodeId>) {
                use luminal::graph::extract_expr_list;
                (
                    LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
                        shape: extract_expr_list(egraph, children[0], list_cache, expr_cache)
                            .unwrap(),
                        input_strides: extract_expr_list(
                            egraph,
                            children[2],
                            list_cache,
                            expr_cache,
                        )
                        .unwrap(),
                        output_strides: extract_expr_list(
                            egraph,
                            children[3],
                            list_cache,
                            expr_cache,
                        )
                        .unwrap(),
                    })),
                    vec![children[1]],
                )
            }
        }

        impl MetalKernelOp for $name {
            fn compile(&self, device: &Device) -> ComputePipelineState {
                // Generate strided index expressions
                let inp_index = flatten_mul_strides(&self.shape, &self.input_strides);
                let out_index = flatten_mul_strides(&self.shape, &self.output_strides);

                // Convert expressions to Metal code
                let inp_idx = inp_index.to_kernel().replace("const_z", "idx");
                let out_idx = out_index.to_kernel().replace("const_z", "idx");

                let source = format!(
                    r#"
                    #include <metal_stdlib>
                    using namespace metal;

                    kernel void mkernel(
                        device float *inp [[buffer(0)]],
                        device float *out [[buffer(1)]],
                        device uint &n_elements [[buffer(2)]],
                        uint idx [[thread_position_in_grid]]
                    ) {{
                        if (idx < n_elements) {{
                            out[{out_idx}] = {metal_op}(inp[{inp_idx}]);
                        }}
                    }}
                    "#,
                    metal_op = $metal_op,
                    inp_idx = inp_idx,
                    out_idx = out_idx
                );
                compile_shader(device, &source, "mkernel")
            }

            fn output_size(&self) -> Expression {
                self.shape
                    .iter()
                    .cloned()
                    .product::<Expression>()
                    .max(Expression::from(1))
            }

            fn encode(
                &self,
                encoder: &ComputeCommandEncoderRef,
                pipeline: &ComputePipelineState,
                inputs: &[&Buffer],
                output: &Buffer,
                dyn_map: &FxHashMap<char, usize>,
            ) {
                let n_elements = self.output_size().exec(dyn_map).unwrap() as u32;

                encoder.set_compute_pipeline_state(pipeline);
                encoder.set_buffer(0, Some(inputs[0]), 0);
                encoder.set_buffer(1, Some(output), 0);
                encoder.set_bytes(
                    2,
                    std::mem::size_of::<u32>() as u64,
                    &n_elements as *const u32 as *const _,
                );

                let thread_group_size = MTLSize::new(256, 1, 1);
                let thread_groups = MTLSize::new((n_elements as u64).div_ceil(256), 1, 1);
                encoder.dispatch_thread_groups(thread_groups, thread_group_size);
            }
        }
    };
}

metal_unary_op!(MetalExp2, "MetalExp2", "exp2");
metal_unary_op!(MetalLog2, "MetalLog2", "log2");
metal_unary_op!(MetalSin, "MetalSin", "sin");
metal_unary_op!(MetalSqrt, "MetalSqrt", "sqrt");
metal_unary_op!(MetalRecip, "MetalRecip", "1.0f /");

#[derive(Debug, Default, Clone)]
pub struct MetalAdd {
    shape: Vec<Expression>,
    a_strides: Vec<Expression>,
    b_strides: Vec<Expression>,
    output_strides: Vec<Expression>,
}

impl EgglogOp for MetalAdd {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "MetalAdd".to_string(),
            vec![EList, Input, EList, Input, EList, EList],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![r#"(rule
            ((= ?e (Add ?shape ?a ?a_stride ?b ?b_stride ?out_stride))
             (= ?dt (dtype ?a)))
            ((let ?me (MetalAdd ?shape ?a ?a_stride ?b ?b_stride ?out_stride))
             (union ?e ?me)
             (set (dtype ?me) ?dt))
        )"#
        .to_string()]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::graph::extract_expr_list;
        (
            LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                a_strides: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                b_strides: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
                output_strides: extract_expr_list(egraph, children[5], list_cache, expr_cache)
                    .unwrap(),
            })),
            vec![children[1], children[3]],
        )
    }
}

impl MetalKernelOp for MetalAdd {
    fn compile(&self, device: &Device) -> ComputePipelineState {
        // Generate strided index expressions using 'z' = thread index
        let a_index = flatten_mul_strides(&self.shape, &self.a_strides);
        let b_index = flatten_mul_strides(&self.shape, &self.b_strides);
        let out_index = flatten_mul_strides(&self.shape, &self.output_strides);

        // Convert expressions to Metal code, replacing 'const_z' with 'idx'
        let a_idx = a_index.to_kernel().replace("const_z", "idx");
        let b_idx = b_index.to_kernel().replace("const_z", "idx");
        let out_idx = out_index.to_kernel().replace("const_z", "idx");

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                device float *a [[buffer(0)]],
                device float *b [[buffer(1)]],
                device float *out [[buffer(2)]],
                device uint &n_elements [[buffer(3)]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    out[{out_idx}] = a[{a_idx}] + b[{b_idx}];
                }}
            }}
            "#
        );
        compile_shader(device, &source, "mkernel")
    }

    fn output_size(&self) -> Expression {
        self.shape
            .iter()
            .cloned()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    ) {
        let n_elements = self.output_size().exec(dyn_map).unwrap() as u32;

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(inputs[0]), 0);
        encoder.set_buffer(1, Some(inputs[1]), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &n_elements as *const u32 as *const _,
        );

        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new((n_elements as u64).div_ceil(256), 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    }
}

#[derive(Debug, Default, Clone)]
pub struct MetalMul {
    shape: Vec<Expression>,
    a_strides: Vec<Expression>,
    b_strides: Vec<Expression>,
    output_strides: Vec<Expression>,
}

impl EgglogOp for MetalMul {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "MetalMul".to_string(),
            vec![EList, Input, EList, Input, EList, EList],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![r#"(rule
            ((= ?e (Mul ?shape ?a ?a_stride ?b ?b_stride ?out_stride))
             (= ?dt (dtype ?a)))
            ((let ?me (MetalMul ?shape ?a ?a_stride ?b ?b_stride ?out_stride))
             (union ?e ?me)
             (set (dtype ?me) ?dt))
        )"#
        .to_string()]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::graph::extract_expr_list;
        (
            LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                a_strides: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                b_strides: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
                output_strides: extract_expr_list(egraph, children[5], list_cache, expr_cache)
                    .unwrap(),
            })),
            vec![children[1], children[3]],
        )
    }
}

impl MetalKernelOp for MetalMul {
    fn compile(&self, device: &Device) -> ComputePipelineState {
        let a_index = flatten_mul_strides(&self.shape, &self.a_strides);
        let b_index = flatten_mul_strides(&self.shape, &self.b_strides);
        let out_index = flatten_mul_strides(&self.shape, &self.output_strides);

        let a_idx = a_index.to_kernel().replace("const_z", "idx");
        let b_idx = b_index.to_kernel().replace("const_z", "idx");
        let out_idx = out_index.to_kernel().replace("const_z", "idx");

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                device float *a [[buffer(0)]],
                device float *b [[buffer(1)]],
                device float *out [[buffer(2)]],
                device uint &n_elements [[buffer(3)]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    out[{out_idx}] = a[{a_idx}] * b[{b_idx}];
                }}
            }}
            "#
        );
        compile_shader(device, &source, "mkernel")
    }

    fn output_size(&self) -> Expression {
        self.shape
            .iter()
            .cloned()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    ) {
        let n_elements = self.output_size().exec(dyn_map).unwrap() as u32;

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(inputs[0]), 0);
        encoder.set_buffer(1, Some(inputs[1]), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &n_elements as *const u32 as *const _,
        );

        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new((n_elements as u64).div_ceil(256), 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    }
}

// MetalMod: a % b using fmod
#[derive(Debug, Default, Clone)]
pub struct MetalMod {
    shape: Vec<Expression>,
    a_strides: Vec<Expression>,
    b_strides: Vec<Expression>,
    output_strides: Vec<Expression>,
}

impl EgglogOp for MetalMod {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "MetalMod".to_string(),
            vec![EList, Input, EList, Input, EList, EList],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![r#"(rule
            ((= ?e (Mod ?shape ?a ?a_stride ?b ?b_stride ?out_stride))
             (= ?dt (dtype ?a)))
            ((let ?me (MetalMod ?shape ?a ?a_stride ?b ?b_stride ?out_stride))
             (union ?e ?me)
             (set (dtype ?me) ?dt))
        )"#
        .to_string()]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::graph::extract_expr_list;
        (
            LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                a_strides: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                b_strides: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
                output_strides: extract_expr_list(egraph, children[5], list_cache, expr_cache)
                    .unwrap(),
            })),
            vec![children[1], children[3]],
        )
    }
}

impl MetalKernelOp for MetalMod {
    fn compile(&self, device: &Device) -> ComputePipelineState {
        let a_index = flatten_mul_strides(&self.shape, &self.a_strides);
        let b_index = flatten_mul_strides(&self.shape, &self.b_strides);
        let out_index = flatten_mul_strides(&self.shape, &self.output_strides);

        let a_idx = a_index.to_kernel().replace("const_z", "idx");
        let b_idx = b_index.to_kernel().replace("const_z", "idx");
        let out_idx = out_index.to_kernel().replace("const_z", "idx");

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                device float *a [[buffer(0)]],
                device float *b [[buffer(1)]],
                device float *out [[buffer(2)]],
                device uint &n_elements [[buffer(3)]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    out[{out_idx}] = fmod(a[{a_idx}], b[{b_idx}]);
                }}
            }}
            "#
        );
        compile_shader(device, &source, "mkernel")
    }

    fn output_size(&self) -> Expression {
        self.shape
            .iter()
            .cloned()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    ) {
        let n_elements = self.output_size().exec(dyn_map).unwrap() as u32;

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(inputs[0]), 0);
        encoder.set_buffer(1, Some(inputs[1]), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &n_elements as *const u32 as *const _,
        );

        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new((n_elements as u64).div_ceil(256), 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    }
}

// MetalLessThan: a < b ? 1.0 : 0.0
#[derive(Debug, Default, Clone)]
pub struct MetalLessThan {
    shape: Vec<Expression>,
    a_strides: Vec<Expression>,
    b_strides: Vec<Expression>,
    output_strides: Vec<Expression>,
}

impl EgglogOp for MetalLessThan {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "MetalLessThan".to_string(),
            vec![EList, Input, EList, Input, EList, EList],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![r#"(rule
            ((= ?e (LessThan ?shape ?a ?a_stride ?b ?b_stride ?out_stride))
             (= ?dt (dtype ?a)))
            ((let ?me (MetalLessThan ?shape ?a ?a_stride ?b ?b_stride ?out_stride))
             (union ?e ?me)
             (set (dtype ?me) ?dt))
        )"#
        .to_string()]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::graph::extract_expr_list;
        (
            LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                a_strides: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                b_strides: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
                output_strides: extract_expr_list(egraph, children[5], list_cache, expr_cache)
                    .unwrap(),
            })),
            vec![children[1], children[3]],
        )
    }
}

impl MetalKernelOp for MetalLessThan {
    fn compile(&self, device: &Device) -> ComputePipelineState {
        let a_index = flatten_mul_strides(&self.shape, &self.a_strides);
        let b_index = flatten_mul_strides(&self.shape, &self.b_strides);
        let out_index = flatten_mul_strides(&self.shape, &self.output_strides);

        let a_idx = a_index.to_kernel().replace("const_z", "idx");
        let b_idx = b_index.to_kernel().replace("const_z", "idx");
        let out_idx = out_index.to_kernel().replace("const_z", "idx");

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                device float *a [[buffer(0)]],
                device float *b [[buffer(1)]],
                device float *out [[buffer(2)]],
                device uint &n_elements [[buffer(3)]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    out[{out_idx}] = (a[{a_idx}] < b[{b_idx}]) ? 1.0f : 0.0f;
                }}
            }}
            "#
        );
        compile_shader(device, &source, "mkernel")
    }

    fn output_size(&self) -> Expression {
        self.shape
            .iter()
            .cloned()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    ) {
        let n_elements = self.output_size().exec(dyn_map).unwrap() as u32;

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(inputs[0]), 0);
        encoder.set_buffer(1, Some(inputs[1]), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &n_elements as *const u32 as *const _,
        );

        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new((n_elements as u64).div_ceil(256), 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    }
}

// ============================================================================
// Reduce Operations
// ============================================================================

#[derive(Debug, Default, Clone)]
pub struct MetalSumReduce {
    out_shape: Vec<Expression>,
    iters: Expression,
    in_stride: Vec<Expression>,
    iter_stride: Expression,
    out_stride: Vec<Expression>,
}

impl EgglogOp for MetalSumReduce {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "MetalSum".to_string(),
            vec![EList, Expr, Input, EList, Expr, EList],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![r#"(rule
            ((= ?e (Sum ?out_shape ?iters ?inp ?in_stride ?iter_stride ?out_stride))
             (= ?dt (dtype ?inp)))
            ((let ?me (MetalSum ?out_shape ?iters ?inp ?in_stride ?iter_stride ?out_stride))
             (union ?e ?me)
             (set (dtype ?me) ?dt))
        )"#
        .to_string()]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::graph::extract_expr;
        use luminal::graph::extract_expr_list;
        (
            LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                iters: extract_expr(egraph, children[1], expr_cache).unwrap(),
                in_stride: extract_expr_list(egraph, children[3], list_cache, expr_cache).unwrap(),
                iter_stride: extract_expr(egraph, children[4], expr_cache).unwrap(),
                out_stride: extract_expr_list(egraph, children[5], list_cache, expr_cache).unwrap(),
            })),
            vec![children[2]],
        )
    }
}

impl MetalKernelOp for MetalSumReduce {
    fn compile(&self, device: &Device) -> ComputePipelineState {
        let in_index = flatten_mul_strides(&self.out_shape, &self.in_stride);
        let out_index = flatten_mul_strides(&self.out_shape, &self.out_stride);

        let in_idx = in_index.to_kernel().replace("const_z", "gid");
        let out_idx = out_index.to_kernel().replace("const_z", "gid");
        let iters = self.iters.to_kernel();
        let iter_stride = self.iter_stride.to_kernel();

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            #define THREADS_PER_GROUP 256

            kernel void mkernel(
                device float *out [[buffer(0)]],
                const device float *in [[buffer(1)]],
                device uint &n_outputs [[buffer(2)]],
                uint gid [[threadgroup_position_in_grid]],
                uint tid [[thread_index_in_threadgroup]],
                uint simd_lane [[thread_index_in_simdgroup]],
                uint simd_id [[simdgroup_index_in_threadgroup]]
            ) {{
                if (gid >= n_outputs) return;

                threadgroup float warp_sums[THREADS_PER_GROUP / 32];

                int in_start = {in_idx};
                int iters = {iters};
                int iter_stride_val = {iter_stride};

                // Each thread accumulates multiple elements
                float sum = 0.0f;
                for (int i = tid; i < iters; i += THREADS_PER_GROUP) {{
                    sum += in[in_start + i * iter_stride_val];
                }}

                // Warp-level reduction using simd_sum
                sum = simd_sum(sum);

                // First lane of each warp writes to shared memory
                if (simd_lane == 0) {{
                    warp_sums[simd_id] = sum;
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // First warp does final reduction
                if (simd_id == 0) {{
                    int n_warps = THREADS_PER_GROUP / 32;
                    float block_sum = (tid < uint(n_warps)) ? warp_sums[tid] : 0.0f;
                    block_sum = simd_sum(block_sum);

                    if (tid == 0) {{
                        out[{out_idx}] = block_sum;
                    }}
                }}
            }}
            "#
        );
        compile_shader(device, &source, "mkernel")
    }

    fn output_size(&self) -> Expression {
        self.out_shape
            .iter()
            .cloned()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    ) {
        let n_outputs = self.output_size().exec(dyn_map).unwrap() as u32;

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(output), 0);
        encoder.set_buffer(1, Some(inputs[0]), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &n_outputs as *const u32 as *const _,
        );

        // One threadgroup per output element
        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new(n_outputs as u64, 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    }
}

#[derive(Debug, Default, Clone)]
pub struct MetalMaxReduce {
    out_shape: Vec<Expression>,
    iters: Expression,
    in_stride: Vec<Expression>,
    iter_stride: Expression,
    out_stride: Vec<Expression>,
}

impl EgglogOp for MetalMaxReduce {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "MetalMax".to_string(),
            vec![EList, Expr, Input, EList, Expr, EList],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![r#"(rule
            ((= ?e (Max ?out_shape ?iters ?inp ?in_stride ?iter_stride ?out_stride))
             (= ?dt (dtype ?inp)))
            ((let ?me (MetalMax ?out_shape ?iters ?inp ?in_stride ?iter_stride ?out_stride))
             (union ?e ?me)
             (set (dtype ?me) ?dt))
        )"#
        .to_string()]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::graph::extract_expr;
        use luminal::graph::extract_expr_list;
        (
            LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                iters: extract_expr(egraph, children[1], expr_cache).unwrap(),
                in_stride: extract_expr_list(egraph, children[3], list_cache, expr_cache).unwrap(),
                iter_stride: extract_expr(egraph, children[4], expr_cache).unwrap(),
                out_stride: extract_expr_list(egraph, children[5], list_cache, expr_cache).unwrap(),
            })),
            vec![children[2]],
        )
    }
}

impl MetalKernelOp for MetalMaxReduce {
    fn compile(&self, device: &Device) -> ComputePipelineState {
        let in_index = flatten_mul_strides(&self.out_shape, &self.in_stride);
        let out_index = flatten_mul_strides(&self.out_shape, &self.out_stride);

        let in_idx = in_index.to_kernel().replace("const_z", "gid");
        let out_idx = out_index.to_kernel().replace("const_z", "gid");
        let iters = self.iters.to_kernel();
        let iter_stride = self.iter_stride.to_kernel();

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            #define THREADS_PER_GROUP 256
            #define NEG_INF_F (-INFINITY)

            kernel void mkernel(
                device float *out [[buffer(0)]],
                const device float *in [[buffer(1)]],
                device uint &n_outputs [[buffer(2)]],
                uint gid [[threadgroup_position_in_grid]],
                uint tid [[thread_index_in_threadgroup]],
                uint simd_lane [[thread_index_in_simdgroup]],
                uint simd_id [[simdgroup_index_in_threadgroup]]
            ) {{
                if (gid >= n_outputs) return;

                threadgroup float warp_maxs[THREADS_PER_GROUP / 32];

                int in_start = {in_idx};
                int iters = {iters};
                int iter_stride_val = {iter_stride};

                // Each thread finds max of multiple elements
                float max_val = NEG_INF_F;
                for (int i = tid; i < iters; i += THREADS_PER_GROUP) {{
                    max_val = fmax(max_val, in[in_start + i * iter_stride_val]);
                }}

                // Warp-level reduction using simd_max
                max_val = simd_max(max_val);

                // First lane of each warp writes to shared memory
                if (simd_lane == 0) {{
                    warp_maxs[simd_id] = max_val;
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);

                // First warp does final reduction
                if (simd_id == 0) {{
                    int n_warps = THREADS_PER_GROUP / 32;
                    float block_max = (tid < uint(n_warps)) ? warp_maxs[tid] : NEG_INF_F;
                    block_max = simd_max(block_max);

                    if (tid == 0) {{
                        out[{out_idx}] = block_max;
                    }}
                }}
            }}
            "#
        );
        compile_shader(device, &source, "mkernel")
    }

    fn output_size(&self) -> Expression {
        self.out_shape
            .iter()
            .cloned()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    ) {
        let n_outputs = self.output_size().exec(dyn_map).unwrap() as u32;

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(output), 0);
        encoder.set_buffer(1, Some(inputs[0]), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &n_outputs as *const u32 as *const _,
        );

        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new(n_outputs as u64, 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    }
}

// ============================================================================
// Data Operations
// ============================================================================

// MetalConstant: produces a single float value
#[derive(Debug, Default, Clone)]
pub struct MetalConstant {
    value: f32,
}

impl EgglogOp for MetalConstant {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("MetalConstant".to_string(), vec![Float])
    }

    fn rewrites(&self) -> Vec<String> {
        vec![r#"(rule
            ((= ?e (Constant ?f)))
            ((let ?me (MetalConstant ?f))
             (union ?e ?me)
             (set (dtype ?me) (F32)))
        )"#
        .to_string()]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        _: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        _: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
                value: egraph.enodes[children[0]]
                    .0
                    .replace("\"", "")
                    .parse::<f32>()
                    .unwrap(),
            })),
            vec![],
        )
    }
}

impl MetalKernelOp for MetalConstant {
    fn compile(&self, device: &Device) -> ComputePipelineState {
        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                device float *out [[buffer(0)]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx == 0) {{
                    out[0] = {value}f;
                }}
            }}
            "#,
            value = self.value
        );
        compile_shader(device, &source, "mkernel")
    }

    fn output_size(&self) -> Expression {
        Expression::from(1)
    }

    fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        _inputs: &[&Buffer],
        output: &Buffer,
        _dyn_map: &FxHashMap<char, usize>,
    ) {
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(output), 0);

        let thread_group_size = MTLSize::new(1, 1, 1);
        let thread_groups = MTLSize::new(1, 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    }
}

// MetalIota: generates sequence [expr(0), expr(1), ..., expr(range-1)]
#[derive(Debug, Default, Clone)]
pub struct MetalIota {
    expr: Expression,
    range: Expression,
}

impl EgglogOp for MetalIota {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("MetalIota".to_string(), vec![Expr, Expr])
    }

    fn rewrites(&self) -> Vec<String> {
        vec![r#"(rule
            ((= ?e (Iota ?expr ?range)))
            ((let ?me (MetalIota ?expr ?range))
             (union ?e ?me)
             (set (dtype ?me) (Int)))
        )"#
        .to_string()]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        _: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::graph::extract_expr;
        (
            LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
                expr: extract_expr(egraph, children[0], expr_cache).unwrap(),
                range: extract_expr(egraph, children[1], expr_cache).unwrap(),
            })),
            vec![],
        )
    }
}

impl MetalKernelOp for MetalIota {
    fn compile(&self, device: &Device) -> ComputePipelineState {
        // Generate the expression as Metal code
        let expr_code = self.expr.to_kernel().replace("const_z", "idx");

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                device int *out [[buffer(0)]],
                device uint &n_elements [[buffer(1)]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    out[idx] = (int)({expr});
                }}
            }}
            "#,
            expr = expr_code
        );
        compile_shader(device, &source, "mkernel")
    }

    fn output_size(&self) -> Expression {
        self.range
    }

    fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        _inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    ) {
        let n_elements = self.range.exec(dyn_map).unwrap() as u32;

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(output), 0);
        encoder.set_bytes(
            1,
            std::mem::size_of::<u32>() as u64,
            &n_elements as *const u32 as *const _,
        );

        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new((n_elements as u64).div_ceil(256), 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    }
}

// MetalGather: indexed lookup - out[i] = data[indexes[i]]
#[derive(Debug, Default, Clone)]
pub struct MetalGather {
    out_shape: Vec<Expression>,
    index_stride: Vec<Expression>,
    data_stride: Vec<Expression>,
    out_stride: Vec<Expression>,
}

impl EgglogOp for MetalGather {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "MetalGather".to_string(),
            vec![EList, Input, EList, Input, EList, EList],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        vec![r#"(rule
            ((= ?a (Gather ?indexes ?out_shape ?index_strides ?data ?data_shape ?data_strides))
             (= ?dty (dtype ?data)))
            ((let ?out_strides (RowMajor ?out_shape))
             (let ?me (MetalGather ?out_shape ?indexes ?index_strides ?data ?data_strides ?out_strides))
             (union ?a ?me)
             (set (dtype ?me) ?dty))
        )"#
        .to_string()]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::graph::extract_expr_list;
        (
            LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                index_stride: extract_expr_list(egraph, children[2], list_cache, expr_cache)
                    .unwrap(),
                data_stride: extract_expr_list(egraph, children[4], list_cache, expr_cache)
                    .unwrap(),
                out_stride: extract_expr_list(egraph, children[5], list_cache, expr_cache).unwrap(),
            })),
            vec![children[1], children[3]],
        )
    }
}

impl MetalKernelOp for MetalGather {
    fn compile(&self, device: &Device) -> ComputePipelineState {
        let out_idx = flatten_mul_strides(&self.out_shape, &self.out_stride)
            .to_kernel()
            .replace("const_z", "idx");
        let index_idx = flatten_mul_strides(&self.out_shape, &self.index_stride)
            .to_kernel()
            .replace("const_z", "idx");
        let data_idx = flatten_mul_strides(&self.out_shape, &self.data_stride)
            .to_kernel()
            .replace("const_z", "gathered_index");

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                device float *out [[buffer(0)]],
                const device int *indexes [[buffer(1)]],
                const device float *data [[buffer(2)]],
                device uint &n_elements [[buffer(3)]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    int gathered_index = indexes[{index_idx}];
                    out[{out_idx}] = data[{data_idx}];
                }}
            }}
            "#
        );
        compile_shader(device, &source, "mkernel")
    }

    fn output_size(&self) -> Expression {
        self.out_shape
            .iter()
            .cloned()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    ) {
        let n_elements = self.output_size().exec(dyn_map).unwrap() as u32;

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(output), 0);
        encoder.set_buffer(1, Some(inputs[0]), 0); // indexes
        encoder.set_buffer(2, Some(inputs[1]), 0); // data
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &n_elements as *const u32 as *const _,
        );

        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new((n_elements as u64).div_ceil(256), 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    }
}
