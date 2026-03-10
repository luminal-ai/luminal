use super::{MetalKernelOp, DYN_BUFFER_INDEX};
use luminal::{
    egglog_utils::{
        api::{app, eq, rule, set, sort, union, v, Rule, SortDef},
        base::{dtype, DTYPE, ELIST, EXPRESSION, F64, IR, OP_SORTS, SORTS},
        SerializedEGraph,
    },
    hlir::{Add, Cast, Constant, Gather, Iota, LessThan, MaxReduce, Mod, Mul, Scatter, SumReduce},
    op::*,
    prelude::*,
    shape::flatten_strides,
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
    MetalScatter,
    // Type conversion
    MetalCast,
    // Embedding lookup
    MetalEmbed,
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

fn lower_dynamic_consts(mut code: String) -> String {
    for c in b'a'..=b'y' {
        let symbol = c as char;
        code = code.replace(
            &format!("*const_{symbol}"),
            &format!("dyn[{}]", (c - b'a') as usize),
        );
    }
    code
}

pub(crate) fn lower_expression_for_metal(expr: &Expression, index_var: &str) -> String {
    lower_dynamic_consts(expr.to_kernel().replace("const_z", index_var))
}

// ============================================================================
// Performance Metrics Macros
// ============================================================================

/// Generate metrics methods for unary ops: 1 input read, 1 output write, 1 flop per element
macro_rules! impl_unary_metrics {
    ($self:ident, $dyn_map:ident) => {
        fn bytes_loaded(&$self, $dyn_map: &FxHashMap<char, usize>) -> usize {
            let n = $self.output_size().exec($dyn_map).unwrap_or(0);
            n * std::mem::size_of::<f32>()
        }

        fn bytes_stored(&$self, $dyn_map: &FxHashMap<char, usize>) -> usize {
            let n = $self.output_size().exec($dyn_map).unwrap_or(0);
            n * std::mem::size_of::<f32>()
        }

        fn flops(&$self, $dyn_map: &FxHashMap<char, usize>) -> usize {
            $self.output_size().exec($dyn_map).unwrap_or(0)
        }
    };
}

/// Generate metrics methods for binary ops: 2 inputs read, 1 output write, flops_per_elem per element
macro_rules! impl_binary_metrics {
    ($self:ident, $dyn_map:ident, $flops_per_elem:expr) => {
        fn bytes_loaded(&$self, $dyn_map: &FxHashMap<char, usize>) -> usize {
            let n = $self.output_size().exec($dyn_map).unwrap_or(0);
            n * 2 * std::mem::size_of::<f32>()
        }

        fn bytes_stored(&$self, $dyn_map: &FxHashMap<char, usize>) -> usize {
            let n = $self.output_size().exec($dyn_map).unwrap_or(0);
            n * std::mem::size_of::<f32>()
        }

        fn flops(&$self, $dyn_map: &FxHashMap<char, usize>) -> usize {
            $self.output_size().exec($dyn_map).unwrap_or(0) * $flops_per_elem
        }
    };
}

/// Generate metrics methods for reduce ops
macro_rules! impl_reduce_metrics {
    ($self:ident, $dyn_map:ident) => {
        fn bytes_loaded(&$self, $dyn_map: &FxHashMap<char, usize>) -> usize {
            let n_outputs = $self.output_size().exec($dyn_map).unwrap_or(0);
            let iters = $self.iters.exec($dyn_map).unwrap_or(0);
            n_outputs * iters * std::mem::size_of::<f32>()
        }

        fn bytes_stored(&$self, $dyn_map: &FxHashMap<char, usize>) -> usize {
            let n = $self.output_size().exec($dyn_map).unwrap_or(0);
            n * std::mem::size_of::<f32>()
        }

        fn flops(&$self, $dyn_map: &FxHashMap<char, usize>) -> usize {
            let n_outputs = $self.output_size().exec($dyn_map).unwrap_or(0);
            let iters = $self.iters.exec($dyn_map).unwrap_or(0);
            n_outputs * iters
        }
    };
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
            fn sort(&self) -> SortDef {
                OP_SORTS.unary($op_name)
            }

            fn rewrites(&self) -> Vec<Rule> {
                let hlir_name = ($op_name).strip_prefix("Metal").unwrap_or($op_name);
                let hlir_sort = OP_SORTS.unary(hlir_name);
                let (args, hlir_match) = hlir_sort.new_call();
                let metal_op = self.sort().call(&args);
                let dt = v("?__dt");
                vec![rule(union(hlir_match, metal_op.clone()))
                    .set(dtype(metal_op), dt.clone())
                    .fact(eq(dt, dtype(args["inp"].clone())))]
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
                use luminal::egglog_utils::extract_expr_list;
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
                let inp_index = flatten_strides(&self.shape, &self.input_strides);
                let out_index = flatten_strides(&self.shape, &self.output_strides);

                // Convert expressions to Metal code
                let inp_idx = lower_expression_for_metal(&inp_index, "idx");
                let out_idx = lower_expression_for_metal(&out_index, "idx");

                let source = format!(
                    r#"
                    #include <metal_stdlib>
                    using namespace metal;

                    kernel void mkernel(
                        device float *inp [[buffer(0)]],
                        device float *out [[buffer(1)]],
                        device uint &n_elements [[buffer(2)]],
                        constant int *dyn [[buffer({dyn_buffer_index})]],
                        uint idx [[thread_position_in_grid]]
                    ) {{
                        if (idx < n_elements) {{
                            out[{out_idx}] = {metal_op}(inp[{inp_idx}]);
                        }}
                    }}
                    "#,
                    metal_op = $metal_op,
                    inp_idx = inp_idx,
                    out_idx = out_idx,
                    dyn_buffer_index = DYN_BUFFER_INDEX
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

            // Performance metrics for MBU/MFU (unary: 1 read, 1 write, 1 flop per element)
            impl_unary_metrics!(self, dyn_map);
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
    fn sort(&self) -> SortDef {
        OP_SORTS.binary("MetalAdd")
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (args1, hlir_match1) = Add::default().sort().new_call();
        let metal_op1 = self.sort().call(&args1);
        let dt = v("?__dt");

        let (args2, hlir_match2) = Add::default().sort().new_call();
        let metal_op2 = self.sort().call(&args2);

        vec![
            rule(union(hlir_match1, metal_op1.clone()))
                .set(dtype(metal_op1), dt.clone())
                .fact(eq(dt, dtype(args1["inp_a"].clone()))),
            rule(union(hlir_match2, metal_op2.clone()))
                .set(dtype(metal_op2), app(&SORTS.f32_dt, vec![])),
        ]
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
        use luminal::egglog_utils::extract_expr_list;
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
        let a_index = flatten_strides(&self.shape, &self.a_strides);
        let b_index = flatten_strides(&self.shape, &self.b_strides);
        let out_index = flatten_strides(&self.shape, &self.output_strides);

        // Convert expressions to Metal code, replacing 'const_z' with 'idx'
        let a_idx = lower_expression_for_metal(&a_index, "idx");
        let b_idx = lower_expression_for_metal(&b_index, "idx");
        let out_idx = lower_expression_for_metal(&out_index, "idx");

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                device float *a [[buffer(0)]],
                device float *b [[buffer(1)]],
                device float *out [[buffer(2)]],
                device uint &n_elements [[buffer(3)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    out[{out_idx}] = a[{a_idx}] + b[{b_idx}];
                }}
            }}
            "#,
            dyn_buffer_index = DYN_BUFFER_INDEX
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

    // Performance metrics for MBU/MFU (binary: 2 reads, 1 write, 1 flop per element)
    impl_binary_metrics!(self, dyn_map, 1);
}

#[derive(Debug, Default, Clone)]
pub struct MetalMul {
    shape: Vec<Expression>,
    a_strides: Vec<Expression>,
    b_strides: Vec<Expression>,
    output_strides: Vec<Expression>,
}

impl EgglogOp for MetalMul {
    fn sort(&self) -> SortDef {
        OP_SORTS.binary("MetalMul")
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (args, hlir_match) = Mul::default().sort().new_call();
        let metal_op = self.sort().call(&args);
        let dt = v("?__dt");
        vec![rule(union(hlir_match, metal_op.clone()))
            .set(dtype(metal_op), dt.clone())
            .fact(eq(dt, dtype(args["inp_a"].clone())))]
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
        use luminal::egglog_utils::extract_expr_list;
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
        let a_index = flatten_strides(&self.shape, &self.a_strides);
        let b_index = flatten_strides(&self.shape, &self.b_strides);
        let out_index = flatten_strides(&self.shape, &self.output_strides);

        let a_idx = lower_expression_for_metal(&a_index, "idx");
        let b_idx = lower_expression_for_metal(&b_index, "idx");
        let out_idx = lower_expression_for_metal(&out_index, "idx");

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                device float *a [[buffer(0)]],
                device float *b [[buffer(1)]],
                device float *out [[buffer(2)]],
                device uint &n_elements [[buffer(3)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    out[{out_idx}] = a[{a_idx}] * b[{b_idx}];
                }}
            }}
            "#,
            dyn_buffer_index = DYN_BUFFER_INDEX
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

    // Performance metrics (binary: 2 reads, 1 write, 1 flop per element)
    impl_binary_metrics!(self, dyn_map, 1);
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
    fn sort(&self) -> SortDef {
        OP_SORTS.binary("MetalMod")
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (args, hlir_match) = Mod::default().sort().new_call();
        let metal_op = self.sort().call(&args);
        let dt = v("?__dt");
        vec![rule(union(hlir_match, metal_op.clone()))
            .set(dtype(metal_op), dt.clone())
            .fact(eq(dt, dtype(args["inp_a"].clone())))]
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
        use luminal::egglog_utils::extract_expr_list;
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
        let a_index = flatten_strides(&self.shape, &self.a_strides);
        let b_index = flatten_strides(&self.shape, &self.b_strides);
        let out_index = flatten_strides(&self.shape, &self.output_strides);

        let a_idx = lower_expression_for_metal(&a_index, "idx");
        let b_idx = lower_expression_for_metal(&b_index, "idx");
        let out_idx = lower_expression_for_metal(&out_index, "idx");

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                device float *a [[buffer(0)]],
                device float *b [[buffer(1)]],
                device float *out [[buffer(2)]],
                device uint &n_elements [[buffer(3)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    out[{out_idx}] = fmod(a[{a_idx}], b[{b_idx}]);
                }}
            }}
            "#,
            dyn_buffer_index = DYN_BUFFER_INDEX
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

    // Performance metrics (binary: 2 reads, 1 write, ~10 flops for fmod)
    impl_binary_metrics!(self, dyn_map, 10);
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
    fn sort(&self) -> SortDef {
        OP_SORTS.binary("MetalLessThan")
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (args, hlir_match) = LessThan::default().sort().new_call();
        let metal_op = self.sort().call(&args);
        let dt = v("?__dt");
        vec![rule(union(hlir_match, metal_op.clone()))
            .set(dtype(metal_op), dt.clone())
            .fact(eq(dt, dtype(args["inp_a"].clone())))]
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
        use luminal::egglog_utils::extract_expr_list;
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
        let a_index = flatten_strides(&self.shape, &self.a_strides);
        let b_index = flatten_strides(&self.shape, &self.b_strides);
        let out_index = flatten_strides(&self.shape, &self.output_strides);

        let a_idx = lower_expression_for_metal(&a_index, "idx");
        let b_idx = lower_expression_for_metal(&b_index, "idx");
        let out_idx = lower_expression_for_metal(&out_index, "idx");

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                device float *a [[buffer(0)]],
                device float *b [[buffer(1)]],
                device float *out [[buffer(2)]],
                device uint &n_elements [[buffer(3)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    out[{out_idx}] = (a[{a_idx}] < b[{b_idx}]) ? 1.0f : 0.0f;
                }}
            }}
            "#,
            dyn_buffer_index = DYN_BUFFER_INDEX
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

    // Performance metrics (binary: 2 reads, 1 write, 1 comparison per element)
    impl_binary_metrics!(self, dyn_map, 1);
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
    fn sort(&self) -> SortDef {
        OP_SORTS.reduce("MetalSum")
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (args, hlir_match) = SumReduce::default().sort().new_call();
        let metal_op = self.sort().call(&args);
        let dt = v("?__dt");
        vec![rule(union(hlir_match, metal_op.clone()))
            .set(dtype(metal_op), dt.clone())
            .fact(eq(dt, dtype(args["inp"].clone())))]
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
        use luminal::egglog_utils::extract_expr;
        use luminal::egglog_utils::extract_expr_list;
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
        let in_index = flatten_strides(&self.out_shape, &self.in_stride);
        let out_index = flatten_strides(&self.out_shape, &self.out_stride);

        let in_idx = lower_expression_for_metal(&in_index, "gid");
        let out_idx = lower_expression_for_metal(&out_index, "gid");
        let iters = lower_expression_for_metal(&self.iters, "gid");
        // iter_stride is an offset expression over the reduction-loop variable, not a scalar stride.
        let iter_offset = lower_expression_for_metal(&self.iter_stride, "i");

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            #define THREADS_PER_GROUP 256

            kernel void mkernel(
                device float *out [[buffer(0)]],
                const device float *in [[buffer(1)]],
                device uint &n_outputs [[buffer(2)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                uint gid [[threadgroup_position_in_grid]],
                uint tid [[thread_index_in_threadgroup]],
                uint simd_lane [[thread_index_in_simdgroup]],
                uint simd_id [[simdgroup_index_in_threadgroup]]
            ) {{
                if (gid >= n_outputs) return;

                threadgroup float warp_sums[THREADS_PER_GROUP / 32];

                int in_start = {in_idx};
                int iters = {iters};
                (void)dyn;

                // Each thread accumulates multiple elements
                float sum = 0.0f;
                for (int i = tid; i < iters; i += THREADS_PER_GROUP) {{
                    sum += in[in_start + {iter_offset}];
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
            "#,
            dyn_buffer_index = DYN_BUFFER_INDEX
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

    // Performance metrics for reduce ops
    impl_reduce_metrics!(self, dyn_map);
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
    fn sort(&self) -> SortDef {
        OP_SORTS.reduce("MetalMax")
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (args, hlir_match) = MaxReduce::default().sort().new_call();
        let metal_op = self.sort().call(&args);
        let dt = v("?__dt");
        vec![rule(union(hlir_match, metal_op.clone()))
            .set(dtype(metal_op), dt.clone())
            .fact(eq(dt, dtype(args["inp"].clone())))]
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
        use luminal::egglog_utils::extract_expr;
        use luminal::egglog_utils::extract_expr_list;
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
        let in_index = flatten_strides(&self.out_shape, &self.in_stride);
        let out_index = flatten_strides(&self.out_shape, &self.out_stride);

        let in_idx = lower_expression_for_metal(&in_index, "gid");
        let out_idx = lower_expression_for_metal(&out_index, "gid");
        let iters = lower_expression_for_metal(&self.iters, "gid");
        // iter_stride is an offset expression over the reduction-loop variable, not a scalar stride.
        let iter_offset = lower_expression_for_metal(&self.iter_stride, "i");

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
                constant int *dyn [[buffer({dyn_buffer_index})]],
                uint gid [[threadgroup_position_in_grid]],
                uint tid [[thread_index_in_threadgroup]],
                uint simd_lane [[thread_index_in_simdgroup]],
                uint simd_id [[simdgroup_index_in_threadgroup]]
            ) {{
                if (gid >= n_outputs) return;

                threadgroup float warp_maxs[THREADS_PER_GROUP / 32];

                int in_start = {in_idx};
                int iters = {iters};
                (void)dyn;

                // Each thread finds max of multiple elements
                float max_val = NEG_INF_F;
                for (int i = tid; i < iters; i += THREADS_PER_GROUP) {{
                    max_val = fmax(max_val, in[in_start + {iter_offset}]);
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
            "#,
            dyn_buffer_index = DYN_BUFFER_INDEX
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

    // Performance metrics for reduce ops
    impl_reduce_metrics!(self, dyn_map);
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
    fn sort(&self) -> SortDef {
        sort(IR, "MetalConstant", &[("value", F64)])
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (args, const_match) = Constant::default().sort().new_call();
        let metal_op = self.sort().call(&args);
        vec![rule(union(const_match, metal_op.clone()))
            .set(dtype(metal_op), app(&SORTS.f32_dt, vec![]))]
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
        // Ensure value is formatted with decimal point for Metal (e.g., -1.0f not -1f)
        let value_str = if self.value.fract() == 0.0 {
            format!("{:.1}", self.value)
        } else {
            format!("{}", self.value)
        };

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                device float *out [[buffer(0)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx == 0) {{
                    out[0] = {value}f;
                }}
            }}
            "#,
            value = value_str,
            dyn_buffer_index = DYN_BUFFER_INDEX
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
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "MetalIota",
            &[("expr", EXPRESSION), ("range", EXPRESSION)],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (args, iota_match) = Iota::default().sort().new_call();
        let metal_op = self.sort().call(&args);
        vec![rule(union(iota_match, metal_op.clone()))
            .set(dtype(metal_op), app(&SORTS.int_dt, vec![]))]
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
        use luminal::egglog_utils::extract_expr;
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
        let expr_code = lower_expression_for_metal(&self.expr, "idx");

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                device int *out [[buffer(0)]],
                device uint &n_elements [[buffer(1)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    out[idx] = (int)({expr});
                }}
            }}
            "#,
            expr = expr_code,
            dyn_buffer_index = DYN_BUFFER_INDEX
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
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "MetalGather",
            &[
                ("out_shape", ELIST),
                ("indexes", IR),
                ("index_strides", ELIST),
                ("data", IR),
                ("data_strides", ELIST),
                ("out_strides", ELIST),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (gather_args, gather_match) = Gather::default().sort().new_call();
        let out_strides = SORTS
            .row_major
            .call([("list".to_string(), gather_args["index_shape"].clone())]);
        let dt = v("?__dt");
        let metal_args = [
            ("out_shape".to_string(), gather_args["index_shape"].clone()),
            ("indexes".to_string(), gather_args["indexes"].clone()),
            (
                "index_strides".to_string(),
                gather_args["index_strides"].clone(),
            ),
            ("data".to_string(), gather_args["data"].clone()),
            (
                "data_strides".to_string(),
                gather_args["data_strides"].clone(),
            ),
            ("out_strides".to_string(), out_strides),
        ];
        let metal_op = self.sort().call(metal_args);
        vec![rule(union(gather_match, metal_op.clone()))
            .set(dtype(metal_op), dt.clone())
            .fact(eq(dt, dtype(gather_args["data"].clone())))]
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
        use luminal::egglog_utils::extract_expr_list;
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
        let out_idx =
            lower_expression_for_metal(&flatten_strides(&self.out_shape, &self.out_stride), "idx");
        let index_idx = lower_expression_for_metal(
            &flatten_strides(&self.out_shape, &self.index_stride),
            "idx",
        );
        let data_idx = lower_expression_for_metal(
            &flatten_strides(&self.out_shape, &self.data_stride),
            "gathered_index",
        );

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                device float *out [[buffer(0)]],
                const device int *indexes [[buffer(1)]],
                const device float *data [[buffer(2)]],
                device uint &n_elements [[buffer(3)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    int gathered_index = indexes[{index_idx}];
                    out[{out_idx}] = data[{data_idx}];
                }}
            }}
            "#,
            dyn_buffer_index = DYN_BUFFER_INDEX
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

    // Gather metrics: read indices + read gathered data + write output
    fn bytes_loaded(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let n = self.output_size().exec(dyn_map).unwrap_or(0);
        // Read n indices (i32 = 4 bytes) + n data elements (f32 = 4 bytes)
        n * std::mem::size_of::<i32>() + n * std::mem::size_of::<f32>()
    }

    fn bytes_stored(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let n = self.output_size().exec(dyn_map).unwrap_or(0);
        // Write n output elements (f32)
        n * std::mem::size_of::<f32>()
    }

    fn flops(&self, _dyn_map: &FxHashMap<char, usize>) -> usize {
        // Gather is memory-bound, no significant FLOPs
        0
    }
}

// MetalScatter: inverse of gather - out = copy(dest); out[indexes[i]] = src[i]
// Uses two sequential dispatches: copy then scatter. Metal guarantees order within an encoder.
#[derive(Debug, Clone)]
pub struct MetalScatter {
    dest_shape: Vec<Expression>,
    dest_strides: Vec<Expression>,
    index_shape: Vec<Expression>,
    index_strides: Vec<Expression>,
    src_strides: Vec<Expression>,
    out_strides: Vec<Expression>,
    copy_pipeline: std::sync::OnceLock<ComputePipelineState>,
}

impl Default for MetalScatter {
    fn default() -> Self {
        Self {
            dest_shape: Vec::new(),
            dest_strides: Vec::new(),
            index_shape: Vec::new(),
            index_strides: Vec::new(),
            src_strides: Vec::new(),
            out_strides: Vec::new(),
            copy_pipeline: std::sync::OnceLock::new(),
        }
    }
}

impl EgglogOp for MetalScatter {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "MetalScatter",
            &[
                ("dest_shape", ELIST),
                ("dest_strides", ELIST),
                ("dest", IR),
                ("indexes", IR),
                ("index_shape", ELIST),
                ("index_strides", ELIST),
                ("src", IR),
                ("src_strides", ELIST),
                ("out_strides", ELIST),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (scatter_args, scatter_match) = Scatter::default().sort().new_call();
        let out_strides = SORTS
            .row_major
            .call([("list".to_string(), scatter_args["dest_shape"].clone())]);
        let dt = v("?__dt");
        let metal_args = [
            ("dest_shape".to_string(), scatter_args["dest_shape"].clone()),
            (
                "dest_strides".to_string(),
                scatter_args["dest_strides"].clone(),
            ),
            ("dest".to_string(), scatter_args["dest"].clone()),
            ("indexes".to_string(), scatter_args["indexes"].clone()),
            (
                "index_shape".to_string(),
                scatter_args["index_shape"].clone(),
            ),
            (
                "index_strides".to_string(),
                scatter_args["index_strides"].clone(),
            ),
            ("src".to_string(), scatter_args["src"].clone()),
            (
                "src_strides".to_string(),
                scatter_args["src_strides"].clone(),
            ),
            ("out_strides".to_string(), out_strides),
        ];
        let metal_op = self.sort().call(metal_args);
        vec![rule([
            union(scatter_match, metal_op.clone()),
            set(dtype(metal_op), dt.clone()),
        ])
        .fact(eq(dt, dtype(scatter_args["src"].clone())))]
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
        use luminal::egglog_utils::extract_expr_list;
        (
            LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
                dest_shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                dest_strides: extract_expr_list(egraph, children[1], list_cache, expr_cache)
                    .unwrap(),
                index_shape: extract_expr_list(egraph, children[4], list_cache, expr_cache)
                    .unwrap(),
                index_strides: extract_expr_list(egraph, children[5], list_cache, expr_cache)
                    .unwrap(),
                src_strides: extract_expr_list(egraph, children[7], list_cache, expr_cache)
                    .unwrap(),
                out_strides: extract_expr_list(egraph, children[8], list_cache, expr_cache)
                    .unwrap(),
                copy_pipeline: std::sync::OnceLock::new(),
            })),
            vec![children[2], children[3], children[6]], // dest, indexes, src
        )
    }
}

impl MetalKernelOp for MetalScatter {
    fn compile(&self, device: &Device) -> ComputePipelineState {
        // Compile the copy kernel and store it
        let dest_idx = lower_expression_for_metal(
            &flatten_strides(&self.dest_shape, &self.dest_strides),
            "idx",
        );
        let out_copy_idx = lower_expression_for_metal(
            &flatten_strides(&self.dest_shape, &self.out_strides),
            "idx",
        );
        let copy_source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void copy_kernel(
                device float *out [[buffer(0)]],
                const device float *dest [[buffer(1)]],
                device uint &n_elements [[buffer(2)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    out[{out_copy_idx}] = dest[{dest_idx}];
                }}
            }}
            "#,
            dyn_buffer_index = DYN_BUFFER_INDEX
        );
        let _ = self
            .copy_pipeline
            .set(compile_shader(device, &copy_source, "copy_kernel"));

        // Compile the scatter kernel (returned as the main pipeline)
        let index_idx = lower_expression_for_metal(
            &flatten_strides(&self.index_shape, &self.index_strides),
            "idx",
        );
        let src_idx = lower_expression_for_metal(
            &flatten_strides(&self.index_shape, &self.src_strides),
            "idx",
        );
        let scatter_source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void scatter_kernel(
                device float *out [[buffer(0)]],
                const device int *indexes [[buffer(1)]],
                const device float *src [[buffer(2)]],
                device uint &n_elements [[buffer(3)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    int scatter_idx = indexes[{index_idx}];
                    out[scatter_idx] = src[{src_idx}];
                }}
            }}
            "#,
            dyn_buffer_index = DYN_BUFFER_INDEX
        );
        compile_shader(device, &scatter_source, "scatter_kernel")
    }

    fn output_size(&self) -> Expression {
        self.dest_shape
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
        let n_dest = self
            .dest_shape
            .iter()
            .cloned()
            .product::<Expression>()
            .exec(dyn_map)
            .unwrap() as u32;
        let n_src = self
            .index_shape
            .iter()
            .cloned()
            .product::<Expression>()
            .exec(dyn_map)
            .unwrap() as u32;

        // Dispatch 1: copy dest → output
        let copy_pipeline = self
            .copy_pipeline
            .get()
            .expect("copy pipeline not compiled");
        encoder.set_compute_pipeline_state(copy_pipeline);
        encoder.set_buffer(0, Some(output), 0);
        encoder.set_buffer(1, Some(inputs[0]), 0); // dest
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &n_dest as *const u32 as *const _,
        );
        let thread_group_size = MTLSize::new(256, 1, 1);
        encoder.dispatch_thread_groups(
            MTLSize::new((n_dest as u64).div_ceil(256), 1, 1),
            thread_group_size,
        );

        // Dispatch 2: scatter src → output[indexes[i]]
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(output), 0);
        encoder.set_buffer(1, Some(inputs[1]), 0); // indexes
        encoder.set_buffer(2, Some(inputs[2]), 0); // src
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &n_src as *const u32 as *const _,
        );
        encoder.dispatch_thread_groups(
            MTLSize::new((n_src as u64).div_ceil(256), 1, 1),
            thread_group_size,
        );
    }

    fn bytes_loaded(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let n_dest = self.output_size().exec(dyn_map).unwrap_or(0);
        let n_src = self
            .index_shape
            .iter()
            .cloned()
            .product::<Expression>()
            .exec(dyn_map)
            .unwrap_or(0);
        n_dest * std::mem::size_of::<f32>()
            + n_src * std::mem::size_of::<i32>()
            + n_src * std::mem::size_of::<f32>()
    }

    fn bytes_stored(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let n = self.output_size().exec(dyn_map).unwrap_or(0);
        n * std::mem::size_of::<f32>()
    }

    fn flops(&self, _dyn_map: &FxHashMap<char, usize>) -> usize {
        0
    }
}

// ============================================================================
// Type Conversion Operations
// ============================================================================

// MetalCast: convert between data types (Int <-> F32, F16, etc.)
// This is a pure element-wise operation with no data movement or reshaping.
#[derive(Debug, Default, Clone)]
pub struct MetalCast {
    size: Expression,
    target_dtype: DType,
}

impl EgglogOp for MetalCast {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "MetalCast",
            &[("inp", IR), ("size", EXPRESSION), ("dtype", DTYPE)],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (args, cast_match) = Cast::default().sort().new_call();
        let metal_op = self.sort().call(&args);
        vec![rule(union(cast_match, metal_op.clone())).set(dtype(metal_op), args["dtype"].clone())]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        _expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::egglog_utils::{extract_dtype, extract_expr};
        (
            LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
                size: extract_expr(egraph, children[1], _expr_cache).unwrap(),
                target_dtype: extract_dtype(egraph, children[2]),
            })),
            vec![children[0]],
        )
    }
}

impl MetalKernelOp for MetalCast {
    fn compile(&self, device: &Device) -> ComputePipelineState {
        let _ = self.target_dtype;
        // MetalRuntime currently allocates all buffers as fp32, so Cast is a no-op copy at runtime.
        // The dtype lives in egglog for correctness / lowering checks, but is not yet reflected in
        // Metal buffer allocation or kernel signatures.
        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                device float *inp [[buffer(0)]],
                device float *out [[buffer(1)]],
                device uint &n_elements [[buffer(2)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    out[idx] = inp[idx];
                }}
            }}
            "#,
            dyn_buffer_index = DYN_BUFFER_INDEX
        );
        compile_shader(device, &source, "mkernel")
    }

    fn output_size(&self) -> Expression {
        self.size
    }

    fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    ) {
        let n_elements = self.size.exec(dyn_map).unwrap_or(0) as u32;

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

    // Cast is memory-bound: 1 read, 1 write, minimal compute
    fn bytes_loaded(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let n = self.size.exec(dyn_map).unwrap_or(0);
        // TODO: input dtype is not encoded; treat as 4B for now (matches current MetalRuntime IO path).
        n * std::mem::size_of::<f32>()
    }

    fn bytes_stored(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let n = self.size.exec(dyn_map).unwrap_or(0);
        // TODO: output dtype sizing should be wired through MetalRuntime allocation.
        n * std::mem::size_of::<f32>()
    }

    fn flops(&self, _dyn_map: &FxHashMap<char, usize>) -> usize {
        0 // Type conversion has negligible compute cost
    }
}

// MetalEmbed: fused embedding lookup - out[i, :] = embed_table[token_ids[i], :]
#[derive(Debug, Clone)]
pub struct MetalEmbed {
    batch_shape: Vec<Expression>,
    token_stride: Vec<Expression>,
    out_stride: Vec<Expression>,
    embed_dim: Expression,
}

impl Default for MetalEmbed {
    fn default() -> Self {
        Self {
            batch_shape: Vec::new(),
            token_stride: Vec::new(),
            out_stride: Vec::new(),
            embed_dim: 1.into(),
        }
    }
}

impl EgglogOp for MetalEmbed {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "MetalEmbed",
            &[
                ("batch_shape", ELIST),
                ("token_ids", IR),
                ("token_stride", ELIST),
                ("embed_table", IR),
                ("out_stride", ELIST),
                ("embed_dim", EXPRESSION),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        // NOTE: Unlike CUDA (which has block ops that pre-create RemoveNthFromEnd nodes),
        // Metal has no such ops. We must create RemoveNthFromEnd in ACTIONS (not conditions)
        // so that (saturate expr) in the next iteration evaluates them into concrete ELists.
        vec![
            // Match Gather with Add(Mul(Cast(token_ids), const), Iota) indices
            Rule::raw("(rule
                (
                    (= ?gather (Gather ?indices ?idx_shape ?idx_stride ?embed_table ?embed_shape ?embed_stride))
                    (= ?indices (Add ?add_shape ?mul_result ?mul_stride ?iota_result ?iota_stride ?add_out_stride))
                    (= ?mul_result (Mul ?mul_shape ?token_ids_cast ?token_cast_stride ?mul_const ?mul_const_stride ?mul_out_stride))
                    (= ?token_ids_cast (Cast ?token_ids ?cast_size ?cast_dtype))
                    (= ?embed_dim (nth_from_end ?embed_shape 0))
                )
                (
                    (let ?batch_shape (RemoveNthFromEnd ?idx_shape 0))
                    (let ?out_stride_batch (RemoveNthFromEnd ?add_out_stride 0))
                    (let ?me (MetalEmbed ?batch_shape ?token_ids ?token_cast_stride ?embed_table ?out_stride_batch ?embed_dim))
                    (union ?gather ?me)
                    (set (dtype ?me) (F32))
                )
                :name \"metal embed with cast mul\"
            )"),
            // Match Gather with Add(Iota, Mul(Cast(token_ids), const)) indices (reversed order)
            Rule::raw("(rule
                (
                    (= ?gather (Gather ?indices ?idx_shape ?idx_stride ?embed_table ?embed_shape ?embed_stride))
                    (= ?indices (Add ?add_shape ?iota_result ?iota_stride ?mul_result ?mul_stride ?add_out_stride))
                    (= ?mul_result (Mul ?mul_shape ?token_ids_cast ?token_cast_stride ?mul_const ?mul_const_stride ?mul_out_stride))
                    (= ?token_ids_cast (Cast ?token_ids ?cast_size ?cast_dtype))
                    (= ?embed_dim (nth_from_end ?embed_shape 0))
                )
                (
                    (let ?batch_shape (RemoveNthFromEnd ?idx_shape 0))
                    (let ?out_stride_batch (RemoveNthFromEnd ?add_out_stride 0))
                    (let ?me (MetalEmbed ?batch_shape ?token_ids ?token_cast_stride ?embed_table ?out_stride_batch ?embed_dim))
                    (union ?gather ?me)
                    (set (dtype ?me) (F32))
                )
                :name \"metal embed with cast mul reversed\"
            )"),
            // Match Gather with Add(Mul(token_ids, const), Iota) indices (no Cast)
            Rule::raw("(rule
                (
                    (= ?gather (Gather ?indices ?idx_shape ?idx_stride ?embed_table ?embed_shape ?embed_stride))
                    (= ?indices (Add ?add_shape ?mul_result ?mul_stride ?iota_result ?iota_stride ?add_out_stride))
                    (= ?mul_result (Mul ?mul_shape ?token_ids ?token_stride ?mul_const ?mul_const_stride ?mul_out_stride))
                    (= ?embed_dim (nth_from_end ?embed_shape 0))
                )
                (
                    (let ?batch_shape (RemoveNthFromEnd ?idx_shape 0))
                    (let ?out_stride_batch (RemoveNthFromEnd ?add_out_stride 0))
                    (let ?me (MetalEmbed ?batch_shape ?token_ids ?token_stride ?embed_table ?out_stride_batch ?embed_dim))
                    (union ?gather ?me)
                    (set (dtype ?me) (F32))
                )
                :name \"metal embed with mul\"
            )"),
            // Match Gather with Add(Iota, Mul(token_ids, const)) indices (reversed, no Cast)
            Rule::raw("(rule
                (
                    (= ?gather (Gather ?indices ?idx_shape ?idx_stride ?embed_table ?embed_shape ?embed_stride))
                    (= ?indices (Add ?add_shape ?iota_result ?iota_stride ?mul_result ?mul_stride ?add_out_stride))
                    (= ?mul_result (Mul ?mul_shape ?token_ids ?token_stride ?mul_const ?mul_const_stride ?mul_out_stride))
                    (= ?embed_dim (nth_from_end ?embed_shape 0))
                )
                (
                    (let ?batch_shape (RemoveNthFromEnd ?idx_shape 0))
                    (let ?out_stride_batch (RemoveNthFromEnd ?add_out_stride 0))
                    (let ?me (MetalEmbed ?batch_shape ?token_ids ?token_stride ?embed_table ?out_stride_batch ?embed_dim))
                    (union ?gather ?me)
                    (set (dtype ?me) (F32))
                )
                :name \"metal embed with mul reversed\"
            )"),
        ]
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
        use luminal::egglog_utils::{extract_expr, extract_expr_list};
        (
            LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
                batch_shape: extract_expr_list(egraph, children[0], list_cache, expr_cache)
                    .unwrap(),
                token_stride: extract_expr_list(egraph, children[2], list_cache, expr_cache)
                    .unwrap(),
                out_stride: extract_expr_list(egraph, children[4], list_cache, expr_cache)
                    .unwrap(),
                embed_dim: extract_expr(egraph, children[5], expr_cache).unwrap(),
            })),
            vec![children[1], children[3]], // token_ids, embed_table
        )
    }
}

impl MetalKernelOp for MetalEmbed {
    fn compile(&self, device: &Device) -> ComputePipelineState {
        let embed_dim_str = lower_expression_for_metal(&self.embed_dim, "");
        let token_offset = lower_expression_for_metal(
            &flatten_strides(&self.batch_shape, &self.token_stride),
            "batch_idx",
        );
        let out_offset = lower_expression_for_metal(
            &flatten_strides(&self.batch_shape, &self.out_stride),
            "batch_idx",
        );

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                device float *out                    [[buffer(0)]],
                const device int *token_ids          [[buffer(1)]],
                const device float *embed_table      [[buffer(2)]],
                device uint &n_elements              [[buffer(3)]],
                constant int *dyn                    [[buffer({dyn_buffer_index})]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    long embed_dim = {embed_dim_str};
                    long batch_idx = idx / embed_dim;
                    long embed_idx = idx % embed_dim;
                    long token_offset = {token_offset};
                    long out_offset   = {out_offset};
                    int token_id = token_ids[token_offset];
                    out[out_offset + embed_idx] = embed_table[(long)token_id * embed_dim + embed_idx];
                }}
            }}
            "#,
            dyn_buffer_index = DYN_BUFFER_INDEX
        );
        compile_shader(device, &source, "mkernel")
    }

    fn output_size(&self) -> Expression {
        self.batch_shape
            .iter()
            .cloned()
            .product::<Expression>()
            .max(Expression::from(1))
            * self.embed_dim
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
        encoder.set_buffer(1, Some(inputs[0]), 0); // token_ids
        encoder.set_buffer(2, Some(inputs[1]), 0); // embed_table
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &n_elements as *const u32 as *const _,
        );

        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new((n_elements as u64).div_ceil(256), 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    }

    fn bytes_loaded(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let n = self.output_size().exec(dyn_map).unwrap_or(0);
        n * std::mem::size_of::<f32>() + n * std::mem::size_of::<i32>()
    }

    fn bytes_stored(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let n = self.output_size().exec(dyn_map).unwrap_or(0);
        n * std::mem::size_of::<f32>()
    }

    fn flops(&self, _dyn_map: &FxHashMap<char, usize>) -> usize {
        0
    }
}
