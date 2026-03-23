use super::{MetalKernelOp, MetalMatmulFamily, MetalMulInfo, MetalSumReduceInfo};
use luminal::{
    egglog_utils::{
        api::{app, eq, rule, sort, union, v, Args, Rule, SortDef, Term as EggTerm},
        base::{dtype, new_op_call, op_term, DTYPE, ELIST, EXPRESSION, F64, IR, SORTS},
        SerializedEGraph,
    },
    hlir::{
        binary_sort, reduce_sort, unary_sort, Add, Cast, Constant, Gather, Iota, LessThan,
        MaxReduce, Mod, Mul, Scatter, SumReduce,
    },
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
    // Matrix ops
    MetalMatmul,
    // Data ops
    MetalConstant,
    MetalIota,
    MetalGather,
    MetalScatter,
    // Type conversion
    MetalCast,
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
            &format!("const_{symbol}"),
            &format!("dyn[{}]", (c - b'a') as usize),
        );
    }
    code
}

pub(crate) fn lower_expression_for_metal(expr: &Expression, index_var: &str) -> String {
    lower_dynamic_consts(expr.to_kernel().replace("const_z", index_var))
}

fn metal_buffer_type(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "float",
        DType::F16 => "half",
        DType::Int => "int",
        _ => panic!("Metal dtype {dtype:?} is not supported yet"),
    }
}

fn metal_numeric_read(dtype: DType, buffer: &str, index: &str) -> String {
    match dtype {
        DType::F32 => format!("{buffer}[{index}]"),
        DType::F16 => format!("float({buffer}[{index}])"),
        DType::Int => format!("float({buffer}[{index}])"),
        _ => panic!("Metal dtype {dtype:?} is not supported yet"),
    }
}

fn metal_numeric_write(dtype: DType, expr: &str) -> String {
    match dtype {
        DType::F32 => expr.to_string(),
        DType::F16 => format!("half({expr})"),
        DType::Int => format!("int({expr})"),
        _ => panic!("Metal dtype {dtype:?} is not supported yet"),
    }
}

fn metal_copy_value(dtype: DType, buffer: &str, index: &str) -> String {
    match dtype {
        DType::F32 | DType::F16 | DType::Int => format!("{buffer}[{index}]"),
        _ => panic!("Metal dtype {dtype:?} is not supported yet"),
    }
}

fn call_sort_from_args(sort: &SortDef, args: &Args) -> EggTerm {
    let mut filtered_args = Args::new();
    for field in &sort.fields {
        filtered_args.add(&field.name, args[field.name.as_str()].clone());
    }
    sort.call(filtered_args)
}

fn unary_dtype_rewrite(hlir_sort: &SortDef, metal_sort: &SortDef) -> Rule {
    let (args, hlir_match) = new_op_call(hlir_sort, &["inp"]);
    let metal_op = op_term(
        call_sort_from_args(metal_sort, &args),
        args["__inputs"].clone(),
    );
    let dt = v("?__dt");
    rule(union(hlir_match, metal_op.clone()))
        .set(dtype(metal_op), dt.clone())
        .fact(eq(dt, dtype(args["inp"].clone())))
}

fn binary_dtype_rewrite(hlir_sort: &SortDef, metal_sort: &SortDef) -> Rule {
    let (args, hlir_match) = new_op_call(hlir_sort, &["inp_a", "inp_b"]);
    let metal_op = op_term(
        call_sort_from_args(metal_sort, &args),
        args["__inputs"].clone(),
    );
    let dt = v("?__dt");
    rule(union(hlir_match, metal_op.clone()))
        .set(dtype(metal_op), dt.clone())
        .fact(eq(dt, dtype(args["inp_a"].clone())))
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
    ($name:ident, $op_name:expr, $expr_builder:expr) => {
        #[derive(Debug, Default, Clone)]
        pub struct $name {
            shape: Vec<Expression>,
            input_strides: Vec<Expression>,
            output_strides: Vec<Expression>,
        }

        impl EgglogOp for $name {
            fn sort(&self) -> SortDef {
                unary_sort($op_name)
            }

            fn rewrites(&self) -> Vec<Rule> {
                let hlir_name = ($op_name).strip_prefix("Metal").unwrap_or($op_name);
                let hlir_sort = unary_sort(hlir_name);
                vec![unary_dtype_rewrite(&hlir_sort, &self.sort())]
            }

            fn cleanup(&self) -> bool {
                false
            }

            fn extract<'a>(
                &'a self,
                egraph: &'a SerializedEGraph,
                kind_children: &[&'a ENodeId],
                input_enodes: Vec<&'a ENodeId>,
                list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
                expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
            ) -> (LLIROp, Vec<&'a ENodeId>) {
                use luminal::egglog_utils::extract_expr_list;
                (
                    LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
                        shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache)
                            .unwrap(),
                        input_strides: extract_expr_list(
                            egraph,
                            kind_children[1],
                            list_cache,
                            expr_cache,
                        )
                        .unwrap(),
                        output_strides: extract_expr_list(
                            egraph,
                            kind_children[2],
                            list_cache,
                            expr_cache,
                        )
                        .unwrap(),
                    })),
                    input_enodes,
                )
            }
        }

        impl MetalKernelOp for $name {
            fn compile(
                &self,
                device: &Device,
                input_dtypes: &[DType],
                output_dtype: DType,
            ) -> ComputePipelineState {
                let input_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
                let input_ty = metal_buffer_type(input_dtype);
                let output_ty = metal_buffer_type(output_dtype);
                // Generate strided index expressions
                let inp_index = flatten_strides(&self.shape, &self.input_strides);
                let out_index = flatten_strides(&self.shape, &self.output_strides);

                // Convert expressions to Metal code
                let inp_idx = lower_expression_for_metal(&inp_index, "idx");
                let out_idx = lower_expression_for_metal(&out_index, "idx");
                let input_expr = metal_numeric_read(input_dtype, "inp", &inp_idx);
                let body_expr = ($expr_builder)(&input_expr);
                let write_expr = metal_numeric_write(output_dtype, &body_expr);

                let source = format!(
                    r#"
                    #include <metal_stdlib>
                    using namespace metal;

                    kernel void mkernel(
                        device {input_ty} *inp [[buffer(0)]],
                        device {output_ty} *out [[buffer(1)]],
                        constant int *dyn [[buffer({dyn_buffer_index})]],
                        device uint &n_elements [[buffer({n_elements_index})]],
                        uint idx [[thread_position_in_grid]]
                    ) {{
                        if (idx < n_elements) {{
                            out[{out_idx}] = {write_expr};
                        }}
                    }}
                    "#,
                    out_idx = out_idx,
                    input_ty = input_ty,
                    output_ty = output_ty,
                    write_expr = write_expr,
                    dyn_buffer_index = 2u64,
                    n_elements_index = 3u64,
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
                    3,
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

metal_unary_op!(MetalExp2, "MetalExp2", |x: &str| format!("exp2({x})"));
metal_unary_op!(MetalLog2, "MetalLog2", |x: &str| format!("log2({x})"));
metal_unary_op!(MetalSin, "MetalSin", |x: &str| format!("sin({x})"));
metal_unary_op!(MetalSqrt, "MetalSqrt", |x: &str| format!("sqrt({x})"));
metal_unary_op!(MetalRecip, "MetalRecip", |x: &str| format!("1.0f / ({x})"));

#[derive(Debug, Default, Clone)]
pub struct MetalAdd {
    shape: Vec<Expression>,
    a_strides: Vec<Expression>,
    b_strides: Vec<Expression>,
    output_strides: Vec<Expression>,
}

impl EgglogOp for MetalAdd {
    fn sort(&self) -> SortDef {
        binary_sort("MetalAdd")
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (args2, hlir_match2) = new_op_call(&Add::default().sort(), &["inp_a", "inp_b"]);
        let metal_op2 = op_term(
            call_sort_from_args(&self.sort(), &args2),
            args2["__inputs"].clone(),
        );

        vec![
            binary_dtype_rewrite(&Add::default().sort(), &self.sort()),
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
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::egglog_utils::extract_expr_list;
        (
            LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
                shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap(),
                a_strides: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                b_strides: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                output_strides: extract_expr_list(egraph, kind_children[3], list_cache, expr_cache)
                    .unwrap(),
            })),
            input_enodes,
        )
    }
}

impl MetalKernelOp for MetalAdd {
    fn compile(
        &self,
        device: &Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> ComputePipelineState {
        let a_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
        let b_dtype = input_dtypes.get(1).copied().unwrap_or(a_dtype);
        let a_ty = metal_buffer_type(a_dtype);
        let b_ty = metal_buffer_type(b_dtype);
        let out_ty = metal_buffer_type(output_dtype);
        // Generate strided index expressions using 'z' = thread index
        let a_index = flatten_strides(&self.shape, &self.a_strides);
        let b_index = flatten_strides(&self.shape, &self.b_strides);
        let out_index = flatten_strides(&self.shape, &self.output_strides);

        // Convert expressions to Metal code, replacing 'const_z' with 'idx'
        let a_idx = lower_expression_for_metal(&a_index, "idx");
        let b_idx = lower_expression_for_metal(&b_index, "idx");
        let out_idx = lower_expression_for_metal(&out_index, "idx");
        let a_val = metal_numeric_read(a_dtype, "a", &a_idx);
        let b_val = metal_numeric_read(b_dtype, "b", &b_idx);
        let out_val = metal_numeric_write(output_dtype, &format!("({a_val}) + ({b_val})"));

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                device {a_ty} *a [[buffer(0)]],
                device {b_ty} *b [[buffer(1)]],
                device {out_ty} *out [[buffer(2)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                device uint &n_elements [[buffer({n_elements_index})]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    out[{out_idx}] = {out_val};
                }}
            }}
            "#,
            a_ty = a_ty,
            b_ty = b_ty,
            out_ty = out_ty,
            out_val = out_val,
            dyn_buffer_index = 3u64,
            n_elements_index = 4u64,
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
            4,
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
        binary_sort("MetalMul")
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![binary_dtype_rewrite(&Mul::default().sort(), &self.sort())]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::egglog_utils::extract_expr_list;
        (
            LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
                shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap(),
                a_strides: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                b_strides: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                output_strides: extract_expr_list(egraph, kind_children[3], list_cache, expr_cache)
                    .unwrap(),
            })),
            input_enodes,
        )
    }
}

impl MetalKernelOp for MetalMul {
    fn compile(
        &self,
        device: &Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> ComputePipelineState {
        let a_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
        let b_dtype = input_dtypes.get(1).copied().unwrap_or(a_dtype);
        let a_ty = metal_buffer_type(a_dtype);
        let b_ty = metal_buffer_type(b_dtype);
        let out_ty = metal_buffer_type(output_dtype);
        let a_index = flatten_strides(&self.shape, &self.a_strides);
        let b_index = flatten_strides(&self.shape, &self.b_strides);
        let out_index = flatten_strides(&self.shape, &self.output_strides);

        let a_idx = lower_expression_for_metal(&a_index, "idx");
        let b_idx = lower_expression_for_metal(&b_index, "idx");
        let out_idx = lower_expression_for_metal(&out_index, "idx");
        let a_val = metal_numeric_read(a_dtype, "a", &a_idx);
        let b_val = metal_numeric_read(b_dtype, "b", &b_idx);
        let out_val = metal_numeric_write(output_dtype, &format!("({a_val}) * ({b_val})"));

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                device {a_ty} *a [[buffer(0)]],
                device {b_ty} *b [[buffer(1)]],
                device {out_ty} *out [[buffer(2)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                device uint &n_elements [[buffer({n_elements_index})]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    out[{out_idx}] = {out_val};
                }}
            }}
            "#,
            a_ty = a_ty,
            b_ty = b_ty,
            out_ty = out_ty,
            out_val = out_val,
            dyn_buffer_index = 3u64,
            n_elements_index = 4u64,
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
            4,
            std::mem::size_of::<u32>() as u64,
            &n_elements as *const u32 as *const _,
        );

        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new((n_elements as u64).div_ceil(256), 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    }

    // Performance metrics (binary: 2 reads, 1 write, 1 flop per element)
    impl_binary_metrics!(self, dyn_map, 1);

    fn mul_info(&self) -> Option<MetalMulInfo> {
        Some(MetalMulInfo {
            shape: self.shape.clone(),
            a_strides: self.a_strides.clone(),
            b_strides: self.b_strides.clone(),
            output_strides: self.output_strides.clone(),
        })
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
    fn sort(&self) -> SortDef {
        binary_sort("MetalMod")
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![binary_dtype_rewrite(&Mod::default().sort(), &self.sort())]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::egglog_utils::extract_expr_list;
        (
            LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
                shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap(),
                a_strides: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                b_strides: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                output_strides: extract_expr_list(egraph, kind_children[3], list_cache, expr_cache)
                    .unwrap(),
            })),
            input_enodes,
        )
    }
}

impl MetalKernelOp for MetalMod {
    fn compile(
        &self,
        device: &Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> ComputePipelineState {
        let a_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
        let b_dtype = input_dtypes.get(1).copied().unwrap_or(a_dtype);
        let a_ty = metal_buffer_type(a_dtype);
        let b_ty = metal_buffer_type(b_dtype);
        let out_ty = metal_buffer_type(output_dtype);
        let a_index = flatten_strides(&self.shape, &self.a_strides);
        let b_index = flatten_strides(&self.shape, &self.b_strides);
        let out_index = flatten_strides(&self.shape, &self.output_strides);

        let a_idx = lower_expression_for_metal(&a_index, "idx");
        let b_idx = lower_expression_for_metal(&b_index, "idx");
        let out_idx = lower_expression_for_metal(&out_index, "idx");
        let a_val = metal_numeric_read(a_dtype, "a", &a_idx);
        let b_val = metal_numeric_read(b_dtype, "b", &b_idx);
        let out_val = metal_numeric_write(output_dtype, &format!("fmod({a_val}, {b_val})"));

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                device {a_ty} *a [[buffer(0)]],
                device {b_ty} *b [[buffer(1)]],
                device {out_ty} *out [[buffer(2)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                device uint &n_elements [[buffer({n_elements_index})]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    out[{out_idx}] = {out_val};
                }}
            }}
            "#,
            a_ty = a_ty,
            b_ty = b_ty,
            out_ty = out_ty,
            out_val = out_val,
            dyn_buffer_index = 3u64,
            n_elements_index = 4u64,
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
            4,
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
        binary_sort("MetalLessThan")
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![binary_dtype_rewrite(
            &LessThan::default().sort(),
            &self.sort(),
        )]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::egglog_utils::extract_expr_list;
        (
            LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
                shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap(),
                a_strides: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                b_strides: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                output_strides: extract_expr_list(egraph, kind_children[3], list_cache, expr_cache)
                    .unwrap(),
            })),
            input_enodes,
        )
    }
}

impl MetalKernelOp for MetalLessThan {
    fn compile(
        &self,
        device: &Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> ComputePipelineState {
        let a_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
        let b_dtype = input_dtypes.get(1).copied().unwrap_or(a_dtype);
        let a_ty = metal_buffer_type(a_dtype);
        let b_ty = metal_buffer_type(b_dtype);
        let out_ty = metal_buffer_type(output_dtype);
        let a_index = flatten_strides(&self.shape, &self.a_strides);
        let b_index = flatten_strides(&self.shape, &self.b_strides);
        let out_index = flatten_strides(&self.shape, &self.output_strides);

        let a_idx = lower_expression_for_metal(&a_index, "idx");
        let b_idx = lower_expression_for_metal(&b_index, "idx");
        let out_idx = lower_expression_for_metal(&out_index, "idx");
        let a_val = metal_numeric_read(a_dtype, "a", &a_idx);
        let b_val = metal_numeric_read(b_dtype, "b", &b_idx);
        let out_val = metal_numeric_write(
            output_dtype,
            &format!("(({a_val}) < ({b_val})) ? 1.0f : 0.0f"),
        );

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                device {a_ty} *a [[buffer(0)]],
                device {b_ty} *b [[buffer(1)]],
                device {out_ty} *out [[buffer(2)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                device uint &n_elements [[buffer({n_elements_index})]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    out[{out_idx}] = {out_val};
                }}
            }}
            "#,
            a_ty = a_ty,
            b_ty = b_ty,
            out_ty = out_ty,
            out_val = out_val,
            dyn_buffer_index = 3u64,
            n_elements_index = 4u64,
        );
        compile_shader(device, &source, "mkernel")
    }

    fn infer_output_dtype(&self, _input_dtypes: &[DType]) -> DType {
        // Metal currently materializes comparisons as numeric 0/1 values.
        DType::F32
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
            4,
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
        reduce_sort("MetalSum")
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![unary_dtype_rewrite(
            &SumReduce::default().sort(),
            &self.sort(),
        )]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::egglog_utils::extract_expr;
        use luminal::egglog_utils::extract_expr_list;
        (
            LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache)
                    .unwrap(),
                iters: extract_expr(egraph, kind_children[1], expr_cache).unwrap(),
                in_stride: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                iter_stride: extract_expr(egraph, kind_children[3], expr_cache).unwrap(),
                out_stride: extract_expr_list(egraph, kind_children[4], list_cache, expr_cache)
                    .unwrap(),
            })),
            input_enodes,
        )
    }
}

impl MetalKernelOp for MetalSumReduce {
    fn compile(
        &self,
        device: &Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> ComputePipelineState {
        let input_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
        let input_ty = metal_buffer_type(input_dtype);
        let output_ty = metal_buffer_type(output_dtype);
        let in_index = flatten_strides(&self.out_shape, &self.in_stride);
        let out_index = flatten_strides(&self.out_shape, &self.out_stride);

        let in_idx = lower_expression_for_metal(&in_index, "gid");
        let out_idx = lower_expression_for_metal(&out_index, "gid");
        let iters = lower_expression_for_metal(&self.iters, "gid");
        // iter_stride is an offset expression over the reduction-loop variable, not a scalar stride.
        let iter_offset = lower_expression_for_metal(&self.iter_stride, "i");
        let in_val = metal_numeric_read(input_dtype, "in", &format!("in_start + {iter_offset}"));
        let out_val = metal_numeric_write(output_dtype, "block_sum");

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            #define THREADS_PER_GROUP 256

            kernel void mkernel(
                const device {input_ty} *in [[buffer(0)]],
                device {output_ty} *out [[buffer(1)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                device uint &n_outputs [[buffer({n_outputs_index})]],
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
                    sum += {in_val};
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
                        out[{out_idx}] = {out_val};
                    }}
                }}
            }}
            "#,
            input_ty = input_ty,
            output_ty = output_ty,
            in_val = in_val,
            out_val = out_val,
            dyn_buffer_index = 2u64,
            n_outputs_index = 3u64,
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
        encoder.set_buffer(0, Some(inputs[0]), 0);
        encoder.set_buffer(1, Some(output), 0);
        encoder.set_bytes(
            3,
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

    fn sum_reduce_info(&self) -> Option<MetalSumReduceInfo> {
        Some(MetalSumReduceInfo {
            shape: self.out_shape.clone(),
            strides: self.out_stride.clone(),
            iters: self.iters,
            iter_stride: self.iter_stride,
        })
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
    fn sort(&self) -> SortDef {
        reduce_sort("MetalMax")
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![unary_dtype_rewrite(
            &MaxReduce::default().sort(),
            &self.sort(),
        )]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::egglog_utils::extract_expr;
        use luminal::egglog_utils::extract_expr_list;
        (
            LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache)
                    .unwrap(),
                iters: extract_expr(egraph, kind_children[1], expr_cache).unwrap(),
                in_stride: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                iter_stride: extract_expr(egraph, kind_children[3], expr_cache).unwrap(),
                out_stride: extract_expr_list(egraph, kind_children[4], list_cache, expr_cache)
                    .unwrap(),
            })),
            input_enodes,
        )
    }
}

impl MetalKernelOp for MetalMaxReduce {
    fn compile(
        &self,
        device: &Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> ComputePipelineState {
        let input_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
        let input_ty = metal_buffer_type(input_dtype);
        let output_ty = metal_buffer_type(output_dtype);
        let in_index = flatten_strides(&self.out_shape, &self.in_stride);
        let out_index = flatten_strides(&self.out_shape, &self.out_stride);

        let in_idx = lower_expression_for_metal(&in_index, "gid");
        let out_idx = lower_expression_for_metal(&out_index, "gid");
        let iters = lower_expression_for_metal(&self.iters, "gid");
        // iter_stride is an offset expression over the reduction-loop variable, not a scalar stride.
        let iter_offset = lower_expression_for_metal(&self.iter_stride, "i");
        let in_val = metal_numeric_read(input_dtype, "in", &format!("in_start + {iter_offset}"));
        let out_val = metal_numeric_write(output_dtype, "block_max");

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            #define THREADS_PER_GROUP 256
            #define NEG_INF_F (-INFINITY)

            kernel void mkernel(
                const device {input_ty} *in [[buffer(0)]],
                device {output_ty} *out [[buffer(1)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                device uint &n_outputs [[buffer({n_outputs_index})]],
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
                    max_val = fmax(max_val, {in_val});
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
                        out[{out_idx}] = {out_val};
                    }}
                }}
            }}
            "#,
            input_ty = input_ty,
            output_ty = output_ty,
            in_val = in_val,
            out_val = out_val,
            dyn_buffer_index = 2u64,
            n_outputs_index = 3u64,
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
        encoder.set_buffer(0, Some(inputs[0]), 0);
        encoder.set_buffer(1, Some(output), 0);
        encoder.set_bytes(
            3,
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

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct MetalGemmParams {
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldb: u32,
    ldd: u32,
    batch_size: u32,
    batch_stride_a: u32,
    batch_stride_b: u32,
    batch_stride_d: u32,
    flags: u32,
}

#[derive(Debug, Default, Clone)]
pub struct MetalMatmul {
    pub m: Expression,
    pub n: Expression,
    pub k: Expression,
    pub lda: Expression,
    pub ldb: Expression,
    pub ldd: Expression,
    pub family: MetalMatmulFamily,
    pub bm: u16,
    pub bn: u16,
    pub bk: u16,
    pub wm: u16,
    pub wn: u16,
    pub batch_size: u32,
    pub batch_stride_a: u32,
    pub batch_stride_b: u32,
    pub batch_stride_d: u32,
}

impl EgglogOp for MetalMatmul {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "MetalMatmul",
            &[
                ("m", EXPRESSION),
                ("n", EXPRESSION),
                ("k", EXPRESSION),
                ("lhs", IR),
                ("lda", EXPRESSION),
                ("rhs", IR),
                ("ldb", EXPRESSION),
                ("ldd", EXPRESSION),
            ],
        )
    }

    fn cleanup(&self) -> bool {
        false
    }
}

impl MetalKernelOp for MetalMatmul {
    fn compile(
        &self,
        device: &Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> ComputePipelineState {
        let lhs_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
        let rhs_dtype = input_dtypes.get(1).copied().unwrap_or(lhs_dtype);
        let lhs_ty = metal_buffer_type(lhs_dtype);
        let rhs_ty = metal_buffer_type(rhs_dtype);
        let out_ty = metal_buffer_type(output_dtype);
        let lhs_read = metal_numeric_read(lhs_dtype, "lhs", "lhs_offset");
        let rhs_read = metal_numeric_read(rhs_dtype, "rhs", "rhs_offset");
        let out_write = metal_numeric_write(output_dtype, "acc");

        let kernel_body = match self.family {
            MetalMatmulFamily::Naive => format!(
                r#"
                kernel void mkernel(
                    const device {lhs_ty} *lhs [[buffer(0)]],
                    const device {rhs_ty} *rhs [[buffer(1)]],
                    device {out_ty} *out [[buffer(2)]],
                    constant int *dyn [[buffer(3)]],
                    constant MetalGemmParams &params [[buffer(4)]],
                    uint2 gid [[thread_position_in_grid]]
                ) {{
                    (void)dyn;
                    if (gid.x >= params.n || gid.y >= params.m) {{
                        return;
                    }}

                    float acc = 0.0f;
                    uint row = gid.y;
                    uint col = gid.x;
                    for (uint kk = 0; kk < params.k; kk++) {{
                        uint lhs_offset = row * params.lda + kk;
                        uint rhs_offset = col + kk * params.ldb;
                        acc += ({lhs_read}) * ({rhs_read});
                    }}

                    out[row * params.ldd + col] = {out_write};
                }}
                "#,
                lhs_ty = lhs_ty,
                rhs_ty = rhs_ty,
                out_ty = out_ty,
                lhs_read = lhs_read,
                rhs_read = rhs_read,
                out_write = out_write,
            ),
            MetalMatmulFamily::RegularTiled => {
                let tile_m = self.bm;
                let tile_n = self.bn;
                let tile_k = self.bk;
                let warp_m = self.wm;
                let warp_n = self.wn;
                format!(
                    r#"
                    #include <metal_simdgroup_matrix>

                    template <typename T>
                    struct Frag8x8 {{
                        using mat_type = simdgroup_matrix<T, 8, 8>;
                        using frag_type = vec<T, 2>;

                        static short2 get_coord(ushort simd_lane_id) {{
                            const short qid = simd_lane_id / 4;
                            const short fm = (qid & 4) + ((simd_lane_id / 2) % 4);
                            const short fn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;
                            return short2{{fn, fm}};
                        }}

                        template <typename U>
                        static void load(
                            thread frag_type& dst,
                            const device U* src,
                            int str_x,
                            int str_y
                        ) {{
                            dst[0] = static_cast<T>(src[0 * str_x + 0 * str_y]);
                            dst[1] = static_cast<T>(src[0 * str_x + 1 * str_y]);
                        }}

                        template <typename U>
                        static void load_safe(
                            thread frag_type& dst,
                            const device U* src,
                            int str_x,
                            int str_y,
                            int lim_x,
                            int lim_y
                        ) {{
                            dst[0] = (lim_x > 0 && lim_y > 0)
                                ? static_cast<T>(src[0 * str_x + 0 * str_y])
                                : T(0);
                            dst[1] = (lim_x > 0 && lim_y > 1)
                                ? static_cast<T>(src[0 * str_x + 1 * str_y])
                                : T(0);
                        }}

                        template <typename U>
                        static void load_tg(
                            thread frag_type& dst,
                            const threadgroup U* src,
                            int str_x,
                            int str_y
                        ) {{
                            dst[0] = static_cast<T>(src[0 * str_x + 0 * str_y]);
                            dst[1] = static_cast<T>(src[0 * str_x + 1 * str_y]);
                        }}

                        template <typename U>
                        static void store_safe(
                            const thread frag_type& src,
                            device U* dst,
                            int str_x,
                            int str_y,
                            int lim_x,
                            int lim_y
                        ) {{
                            if (lim_x > 0 && lim_y > 0) {{
                                dst[0 * str_x + 0 * str_y] = static_cast<U>(src[0]);
                            }}
                            if (lim_x > 0 && lim_y > 1) {{
                                dst[0 * str_x + 1 * str_y] = static_cast<U>(src[1]);
                            }}
                        }}

                        static void mma(
                            thread frag_type& d,
                            thread frag_type& a,
                            thread frag_type& b,
                            thread frag_type& c
                        ) {{
                            mat_type d_mat;
                            mat_type a_mat;
                            mat_type b_mat;
                            mat_type c_mat;

                            reinterpret_cast<thread frag_type&>(a_mat.thread_elements()) = a;
                            reinterpret_cast<thread frag_type&>(b_mat.thread_elements()) = b;
                            reinterpret_cast<thread frag_type&>(c_mat.thread_elements()) = c;

                            simdgroup_multiply_accumulate(d_mat, a_mat, b_mat, c_mat);
                            d = reinterpret_cast<thread frag_type&>(d_mat.thread_elements());
                        }}
                    }};

                    kernel void mkernel(
                        const device {lhs_ty} *lhs [[buffer(0)]],
                        const device {rhs_ty} *rhs [[buffer(1)]],
                        device {out_ty} *out [[buffer(2)]],
                        constant int *dyn [[buffer(3)]],
                        constant MetalGemmParams &params [[buffer(4)]],
                        uint lid [[thread_index_in_threadgroup]],
                        ushort simd_lane_id [[thread_index_in_simdgroup]],
                        ushort simd_group_id [[simdgroup_index_in_threadgroup]],
                        uint3 tgid [[threadgroup_position_in_grid]]
                    ) {{
                        (void)dyn;
                        constexpr uint TILE_M = {tile_m};
                        constexpr uint TILE_N = {tile_n};
                        constexpr uint TILE_K = {tile_k};
                        constexpr uint WM = {warp_m};
                        constexpr uint WN = {warp_n};
                        constexpr uint FRAG = 8;
                        constexpr uint THREADS_PER_TG = WM * WN * 32;

                        threadgroup {lhs_ty} As[TILE_M * TILE_K];
                        threadgroup {rhs_ty} Bs[TILE_K * TILE_N];

                        short2 lane_coord = Frag8x8<float>::get_coord(simd_lane_id);
                        uint sg_row = simd_group_id / WN;
                        uint sg_col = simd_group_id % WN;
                        uint row = tgid.y * TILE_M + sg_row * FRAG;
                        uint col = tgid.x * TILE_N + sg_col * FRAG;

                        thread Frag8x8<float>::frag_type a_frag = Frag8x8<float>::frag_type(0.0f);
                        thread Frag8x8<float>::frag_type b_frag = Frag8x8<float>::frag_type(0.0f);
                        thread Frag8x8<float>::frag_type c_frag = Frag8x8<float>::frag_type(0.0f);
                        thread Frag8x8<float>::frag_type d_frag = Frag8x8<float>::frag_type(0.0f);

                        int row_remain = row < params.m ? int(params.m - row) : 0;
                        int col_remain = col < params.n ? int(params.n - col) : 0;

                        for (uint kk0 = 0; kk0 < params.k; kk0 += TILE_K) {{
                            for (uint idx = lid; idx < TILE_M * TILE_K; idx += THREADS_PER_TG) {{
                                uint local_row = idx / TILE_K;
                                uint local_k = idx % TILE_K;
                                uint global_row = tgid.y * TILE_M + local_row;
                                uint global_k = kk0 + local_k;
                                if (global_row < params.m && global_k < params.k) {{
                                    As[idx] = lhs[global_row * params.lda + global_k];
                                }} else {{
                                    As[idx] = ({lhs_ty})0;
                                }}
                            }}

                            for (uint idx = lid; idx < TILE_K * TILE_N; idx += THREADS_PER_TG) {{
                                uint local_k = idx / TILE_N;
                                uint local_col = idx % TILE_N;
                                uint global_k = kk0 + local_k;
                                uint global_col = tgid.x * TILE_N + local_col;
                                if (global_k < params.k && global_col < params.n) {{
                                    Bs[idx] = rhs[global_k * params.ldb + global_col];
                                }} else {{
                                    Bs[idx] = ({rhs_ty})0;
                                }}
                            }}

                            threadgroup_barrier(mem_flags::mem_threadgroup);

                            for (uint kk = 0; kk < TILE_K; kk += FRAG) {{
                                const threadgroup {lhs_ty}* a_ptr = As
                                    + (sg_row * FRAG + lane_coord.y) * TILE_K
                                    + kk
                                    + lane_coord.x;
                                const threadgroup {rhs_ty}* b_ptr = Bs
                                    + (kk + lane_coord.y) * TILE_N
                                    + (sg_col * FRAG)
                                    + lane_coord.x;

                                Frag8x8<float>::load_tg(a_frag, a_ptr, TILE_K, 1);
                                Frag8x8<float>::load_tg(b_frag, b_ptr, TILE_N, 1);

                                Frag8x8<float>::mma(d_frag, a_frag, b_frag, c_frag);
                                c_frag = d_frag;
                            }}

                            threadgroup_barrier(mem_flags::mem_threadgroup);
                        }}

                        if (row_remain > 0 && col_remain > 0) {{
                            device {out_ty}* out_ptr = out
                                + row * params.ldd
                                + col
                                + lane_coord.y * params.ldd
                                + lane_coord.x;
                            Frag8x8<float>::store_safe(
                                c_frag,
                                out_ptr,
                                params.ldd,
                                1,
                                max(0, row_remain - int(lane_coord.y)),
                                max(0, col_remain - int(lane_coord.x))
                            );
                        }}
                    }}
                    "#,
                    lhs_ty = lhs_ty,
                    rhs_ty = rhs_ty,
                    out_ty = out_ty,
                    tile_m = tile_m,
                    tile_n = tile_n,
                    tile_k = tile_k,
                    warp_m = warp_m,
                    warp_n = warp_n,
                )
            }
        };
        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            struct MetalGemmParams {{
                uint m;
                uint n;
                uint k;
                uint lda;
                uint ldb;
                uint ldd;
                uint batch_size;
                uint batch_stride_a;
                uint batch_stride_b;
                uint batch_stride_d;
                uint flags;
            }};

            {kernel_body}
            "#,
            kernel_body = kernel_body,
        );
        compile_shader(device, &source, "mkernel")
    }

    fn infer_output_dtype(&self, input_dtypes: &[DType]) -> DType {
        input_dtypes.first().copied().unwrap_or(DType::F32)
    }

    fn output_size(&self) -> Expression {
        self.m * self.n
    }

    fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    ) {
        let stride_value = |expr: Expression| {
            expr.substitute('z', Expression::from(1))
                .exec(dyn_map)
                .unwrap() as u32
        };
        let params = MetalGemmParams {
            m: self.m.exec(dyn_map).unwrap() as u32,
            n: self.n.exec(dyn_map).unwrap() as u32,
            k: self.k.exec(dyn_map).unwrap() as u32,
            lda: stride_value(self.lda),
            ldb: stride_value(self.ldb),
            ldd: stride_value(self.ldd),
            batch_size: self.batch_size,
            batch_stride_a: self.batch_stride_a,
            batch_stride_b: self.batch_stride_b,
            batch_stride_d: self.batch_stride_d,
            flags: match self.family {
                MetalMatmulFamily::Naive => 0,
                MetalMatmulFamily::RegularTiled => 1,
            },
        };

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(inputs[0]), 0);
        encoder.set_buffer(1, Some(inputs[1]), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<MetalGemmParams>() as u64,
            &params as *const MetalGemmParams as *const _,
        );

        let thread_group_size = match self.family {
            MetalMatmulFamily::Naive => MTLSize::new(self.bn as u64, self.bm as u64, 1),
            MetalMatmulFamily::RegularTiled => {
                MTLSize::new((self.wm as u64) * (self.wn as u64) * 32, 1, 1)
            }
        };
        let thread_groups = MTLSize::new(
            (params.n as u64).div_ceil(self.bn as u64),
            (params.m as u64).div_ceil(self.bm as u64),
            1,
        );
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    }

    fn bytes_loaded(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let m = self.m.exec(dyn_map).unwrap_or(0);
        let n = self.n.exec(dyn_map).unwrap_or(0);
        let k = self.k.exec(dyn_map).unwrap_or(0);
        2 * m * n * k * std::mem::size_of::<f32>()
    }

    fn bytes_stored(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let m = self.m.exec(dyn_map).unwrap_or(0);
        let n = self.n.exec(dyn_map).unwrap_or(0);
        m * n * std::mem::size_of::<f32>()
    }

    fn flops(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let m = self.m.exec(dyn_map).unwrap_or(0);
        let n = self.n.exec(dyn_map).unwrap_or(0);
        let k = self.k.exec(dyn_map).unwrap_or(0);
        2 * m * n * k
    }

    fn is_matmul(&self) -> bool {
        true
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
    fn sort(&self) -> SortDef {
        sort(IR, "MetalConstant", &[("value", F64)])
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (args, const_match) = new_op_call(&Constant::default().sort(), &[]);
        let metal_op = call_sort_from_args(&self.sort(), &args);
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
        _input_enodes: Vec<&'a ENodeId>,
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
    fn compile(
        &self,
        device: &Device,
        _input_dtypes: &[DType],
        _output_dtype: DType,
    ) -> ComputePipelineState {
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
            dyn_buffer_index = 1u64,
        );
        compile_shader(device, &source, "mkernel")
    }

    fn output_size(&self) -> Expression {
        Expression::from(1)
    }

    fn infer_output_dtype(&self, _input_dtypes: &[DType]) -> DType {
        DType::F32
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
        let (args, iota_match) = new_op_call(&Iota::default().sort(), &[]);
        let metal_op = call_sort_from_args(&self.sort(), &args);
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
        _input_enodes: Vec<&'a ENodeId>,
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
    fn compile(
        &self,
        device: &Device,
        _input_dtypes: &[DType],
        _output_dtype: DType,
    ) -> ComputePipelineState {
        // Generate the expression as Metal code
        let expr_code = lower_expression_for_metal(&self.expr, "idx");

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                device int *out [[buffer(0)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                device uint &n_elements [[buffer({n_elements_index})]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    out[idx] = (int)({expr});
                }}
            }}
            "#,
            expr = expr_code,
            dyn_buffer_index = 1u64,
            n_elements_index = 2u64,
        );
        compile_shader(device, &source, "mkernel")
    }

    fn output_size(&self) -> Expression {
        self.range
    }

    fn infer_output_dtype(&self, _input_dtypes: &[DType]) -> DType {
        DType::Int
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
            2,
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
        let (gather_args, gather_match) =
            new_op_call(&Gather::default().sort(), &["indexes", "data"]);
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
        _input_enodes: Vec<&'a ENodeId>,
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
    fn compile(
        &self,
        device: &Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> ComputePipelineState {
        let data_dtype = input_dtypes.get(1).copied().unwrap_or(DType::F32);
        let data_ty = metal_buffer_type(data_dtype);
        let out_ty = metal_buffer_type(output_dtype);
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
        let gathered_val = metal_copy_value(data_dtype, "data", &data_idx);

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                const device int *indexes [[buffer(0)]],
                const device {data_ty} *data [[buffer(1)]],
                device {out_ty} *out [[buffer(2)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                device uint &n_elements [[buffer({n_elements_index})]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    int gathered_index = indexes[{index_idx}];
                    out[{out_idx}] = {gathered_val};
                }}
            }}
            "#,
            data_ty = data_ty,
            out_ty = out_ty,
            gathered_val = gathered_val,
            dyn_buffer_index = 3u64,
            n_elements_index = 4u64,
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
        encoder.set_buffer(0, Some(inputs[0]), 0); // indexes
        encoder.set_buffer(1, Some(inputs[1]), 0); // data
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(
            4,
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
        let (scatter_args, scatter_match) =
            new_op_call(&Scatter::default().sort(), &["dest", "indexes", "src"]);
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
        vec![rule(union(scatter_match, metal_op.clone()))
            .set(dtype(metal_op), dt.clone())
            .fact(eq(dt, dtype(scatter_args["src"].clone())))]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        _input_enodes: Vec<&'a ENodeId>,
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
    fn compile(
        &self,
        device: &Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> ComputePipelineState {
        let dest_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
        let src_dtype = input_dtypes.get(2).copied().unwrap_or(output_dtype);
        let dest_ty = metal_buffer_type(dest_dtype);
        let src_ty = metal_buffer_type(src_dtype);
        let out_ty = metal_buffer_type(output_dtype);
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
                device {out_ty} *out [[buffer(0)]],
                const device {dest_ty} *dest [[buffer(1)]],
                device uint &n_elements [[buffer(2)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    out[{out_copy_idx}] = dest[{dest_idx}];
                }}
            }}
            "#,
            out_ty = out_ty,
            dest_ty = dest_ty,
            dyn_buffer_index = 4u64
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
                device {out_ty} *out [[buffer(0)]],
                const device int *indexes [[buffer(1)]],
                const device {src_ty} *src [[buffer(2)]],
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
            out_ty = out_ty,
            src_ty = src_ty,
            dyn_buffer_index = 4u64
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
        let (args, cast_match) = new_op_call(&Cast::default().sort(), &["inp"]);
        let metal_op = call_sort_from_args(&self.sort(), &args);
        vec![rule(union(cast_match, metal_op.clone())).set(dtype(metal_op), args["dtype"].clone())]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        _input_enodes: Vec<&'a ENodeId>,
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
    fn compile(
        &self,
        device: &Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> ComputePipelineState {
        let input_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
        let input_ty = metal_buffer_type(input_dtype);
        let output_ty = metal_buffer_type(output_dtype);
        let cast_expr = match (input_dtype, output_dtype) {
            (DType::F32, DType::F32)
            | (DType::F16, DType::F16)
            | (DType::Int, DType::Int)
            | (DType::F32, DType::F16)
            | (DType::F16, DType::F32)
            | (DType::F32, DType::Int)
            | (DType::Int, DType::F32)
            | (DType::F16, DType::Int)
            | (DType::Int, DType::F16) => format!("({output_ty})(inp[idx])"),
            _ => panic!(
                "MetalCast does not support runtime cast from {input_dtype:?} to {output_dtype:?}"
            ),
        };
        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                device {input_ty} *inp [[buffer(0)]],
                device {output_ty} *out [[buffer(1)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                device uint &n_elements [[buffer({n_elements_index})]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    out[idx] = {cast_expr};
                }}
            }}
            "#,
            input_ty = input_ty,
            output_ty = output_ty,
            cast_expr = cast_expr,
            dyn_buffer_index = 2u64,
            n_elements_index = 3u64,
        );
        compile_shader(device, &source, "mkernel")
    }

    fn output_size(&self) -> Expression {
        self.size
    }

    fn infer_output_dtype(&self, _input_dtypes: &[DType]) -> DType {
        self.target_dtype
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
            3,
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
