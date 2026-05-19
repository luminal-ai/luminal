use super::{MPSMatrixLayout, MetalKernelOp, MetalMulInfo, MetalSumReduceInfo};
use luminal::{
    egglog_utils::{
        SerializedEGraph,
        api::{
            Args, Rule, SortDef, Term as EggTerm, app, eq, i64 as lit_i64, rule, sort, union, v,
        },
        base::{
            DTYPE, ELIST, EXPRESSION, F64, I64, IR, OP_KIND, SORTS, add, cons, div, dtype, ilist,
            iter, modd, mul, new_op_call, nil, num, op_term,
        },
    },
    hlir::{
        Add, Cast, Constant, Gather, Iota, LessThan, MaxReduce, Mod, Mul, Scatter, SumReduce,
        binary_sort, reduce_sort, unary_sort,
    },
    op::*,
    prelude::*,
    shape::flatten_strides,
};
use metal::{
    Buffer, CommandBufferRef, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLSize,
    foreign_types::{ForeignType, ForeignTypeRef},
    mps,
};
use objc::runtime::Object;
use objc::{class, msg_send, sel, sel_impl};

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
    MPSMatmul,
    MPSBatchedMatmul,
    GenericMatmul,
    // Data ops
    MetalConstant,
    MetalIota,
    MetalGather,
    MetalScatter,
    MetalScatterNoCopy,
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

fn metal_binary_op_values(
    output_dtype: DType,
    a_dtype: DType,
    b_dtype: DType,
    a_idx: &str,
    b_idx: &str,
) -> (String, String) {
    let read: fn(DType, &str, &str) -> String = if output_dtype == DType::Int {
        metal_copy_value
    } else {
        metal_numeric_read
    };
    (read(a_dtype, "a", a_idx), read(b_dtype, "b", b_idx))
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
    rule(union(hlir_match.clone(), metal_op.clone()))
        .subsume(hlir_match)
        .set(dtype(metal_op), dt.clone())
        .fact(eq(dt, dtype(args["inp"].clone())))
        .ruleset("kernel_lower")
}

fn binary_dtype_rewrite(hlir_sort: &SortDef, metal_sort: &SortDef) -> Rule {
    let (args, hlir_match) = new_op_call(hlir_sort, &["inp_a", "inp_b"]);
    let metal_op = op_term(
        call_sort_from_args(metal_sort, &args),
        args["__inputs"].clone(),
    );
    let dt = v("?__dt");
    rule(union(hlir_match.clone(), metal_op.clone()))
        .subsume(hlir_match)
        .set(dtype(metal_op), dt.clone())
        .fact(eq(dt, dtype(args["inp_a"].clone())))
        .ruleset("kernel_lower")
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
            ) -> Option<ComputePipelineState> {
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
                        constant uint &n_elements [[buffer({n_elements_index})]],
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
                Some(compile_shader(device, &source, "mkernel"))
            }

            fn output_size(&self) -> Expression {
                self.shape
                    .iter()
                    .cloned()
                    .product::<Expression>()
                    .max(Expression::from(1))
            }

            fn encode_compute(
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
            rule(union(hlir_match2.clone(), metal_op2.clone()))
                .subsume(hlir_match2)
                .set(dtype(metal_op2), app(&SORTS.f32_dt, vec![]))
                .ruleset("kernel_lower"),
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
    ) -> Option<ComputePipelineState> {
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
        let (a_val, b_val) = metal_binary_op_values(output_dtype, a_dtype, b_dtype, &a_idx, &b_idx);
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
                constant uint &n_elements [[buffer({n_elements_index})]],
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
        Some(compile_shader(device, &source, "mkernel"))
    }

    fn output_size(&self) -> Expression {
        self.shape
            .iter()
            .cloned()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn encode_compute(
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
    ) -> Option<ComputePipelineState> {
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
        let (a_val, b_val) = metal_binary_op_values(output_dtype, a_dtype, b_dtype, &a_idx, &b_idx);
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
                constant uint &n_elements [[buffer({n_elements_index})]],
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
        Some(compile_shader(device, &source, "mkernel"))
    }

    fn output_size(&self) -> Expression {
        self.shape
            .iter()
            .cloned()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn encode_compute(
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
    ) -> Option<ComputePipelineState> {
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
        let (a_val, b_val) = metal_binary_op_values(output_dtype, a_dtype, b_dtype, &a_idx, &b_idx);
        let out_expr = if output_dtype == DType::Int {
            format!("({a_val}) % ({b_val})")
        } else {
            format!("fmod({a_val}, {b_val})")
        };
        let out_val = metal_numeric_write(output_dtype, &out_expr);

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                device {a_ty} *a [[buffer(0)]],
                device {b_ty} *b [[buffer(1)]],
                device {out_ty} *out [[buffer(2)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                constant uint &n_elements [[buffer({n_elements_index})]],
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
        Some(compile_shader(device, &source, "mkernel"))
    }

    fn output_size(&self) -> Expression {
        self.shape
            .iter()
            .cloned()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn encode_compute(
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
    ) -> Option<ComputePipelineState> {
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
                constant uint &n_elements [[buffer({n_elements_index})]],
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
        Some(compile_shader(device, &source, "mkernel"))
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

    fn encode_compute(
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
    ) -> Option<ComputePipelineState> {
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
                constant uint &n_outputs [[buffer({n_outputs_index})]],
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
        Some(compile_shader(device, &source, "mkernel"))
    }

    fn output_size(&self) -> Expression {
        self.out_shape
            .iter()
            .cloned()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn encode_compute(
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
    ) -> Option<ComputePipelineState> {
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
                constant uint &n_outputs [[buffer({n_outputs_index})]],
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
        Some(compile_shader(device, &source, "mkernel"))
    }

    fn output_size(&self) -> Expression {
        self.out_shape
            .iter()
            .cloned()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn encode_compute(
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

#[derive(Debug, Default, Clone)]
pub struct MPSMatmul {
    pub m: Expression,
    pub n: Expression,
    pub k: Expression,
    pub lhs_row_stride: Expression,
    pub rhs_row_stride: Expression,
    pub out_row_stride: Expression,
    pub transpose_lhs: bool,
    pub transpose_rhs: bool,
}

impl EgglogOp for MPSMatmul {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "MPSMatmul",
            &[
                ("m", EXPRESSION),
                ("n", EXPRESSION),
                ("k", EXPRESSION),
                ("lhs", IR),
                ("lhs_row_stride", EXPRESSION),
                ("rhs", IR),
                ("rhs_row_stride", EXPRESSION),
                ("out_row_stride", EXPRESSION),
                ("transpose_lhs", I64),
                ("transpose_rhs", I64),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        let zero = num(lit_i64(0));
        let z = iter();
        let expr_list = |terms: Vec<EggTerm>| {
            terms
                .into_iter()
                .rev()
                .fold(nil(), |tail, head| cons(head, tail))
        };

        let matmul_rule = |name: &'static str,
                           lhs_layout: MPSMatrixLayout,
                           rhs_layout: MPSMatrixLayout,
                           transpose_lhs: i64,
                           transpose_rhs: i64| {
            let m = v("?m");
            let n = v("?n");
            let k = v("?k");
            let lhs = v("?lhs");
            let rhs = v("?rhs");
            let lhs_row_stride = match lhs_layout {
                MPSMatrixLayout::RowMajor => mul(k.clone(), z.clone()),
                MPSMatrixLayout::TransposedRowMajor => mul(m.clone(), z.clone()),
            };
            let rhs_row_stride = match rhs_layout {
                MPSMatrixLayout::RowMajor => mul(n.clone(), z.clone()),
                MPSMatrixLayout::TransposedRowMajor => mul(k.clone(), z.clone()),
            };
            let lhs_strides = match lhs_layout {
                MPSMatrixLayout::RowMajor => vec![lhs_row_stride.clone(), zero.clone(), z.clone()],
                MPSMatrixLayout::TransposedRowMajor => {
                    vec![z.clone(), zero.clone(), lhs_row_stride.clone()]
                }
            };
            let rhs_strides = match rhs_layout {
                MPSMatrixLayout::RowMajor => vec![zero.clone(), z.clone(), rhs_row_stride.clone()],
                MPSMatrixLayout::TransposedRowMajor => {
                    vec![zero.clone(), rhs_row_stride.clone(), z.clone()]
                }
            };
            let out_row_stride = mul(n.clone(), z.clone());
            let mul_output_strides = v("?mul_output_strides");

            let mul_op = op_term(
                MetalMul {
                    shape: vec![],
                    a_strides: vec![],
                    b_strides: vec![],
                    output_strides: vec![],
                }
                .sort()
                .call([
                    (
                        "shape",
                        cons(m.clone(), cons(n.clone(), cons(k.clone(), nil()))),
                    ),
                    ("a_strides", expr_list(lhs_strides)),
                    ("b_strides", expr_list(rhs_strides)),
                    ("out_strides", mul_output_strides.clone()),
                ]),
                ilist(vec![lhs.clone(), rhs.clone()]),
            );
            let sum_op = op_term(
                MetalSumReduce::default().sort().call([
                    ("shape", cons(m.clone(), cons(n.clone(), nil()))),
                    ("iters", k.clone()),
                    ("strides", v("?sum_in_strides")),
                    ("iter_stride", z.clone()),
                    (
                        "out_strides",
                        cons(out_row_stride.clone(), cons(z.clone(), nil())),
                    ),
                ]),
                ilist(vec![mul_op.clone()]),
            );
            let mps_op = MPSMatmul::default().sort().call([
                ("m", m),
                ("n", n),
                ("k", k),
                ("lhs", lhs),
                ("lhs_row_stride", lhs_row_stride),
                ("rhs", rhs),
                ("rhs_row_stride", rhs_row_stride),
                ("out_row_stride", out_row_stride),
                ("transpose_lhs", lit_i64(transpose_lhs)),
                ("transpose_rhs", lit_i64(transpose_rhs)),
            ]);
            let dt = v(format!("?{}_dt", name.replace('-', "_")));

            rule(union(sum_op.clone(), mps_op.clone()))
                .subsume(sum_op.clone())
                .subsume(mul_op)
                .set(dtype(mps_op), dt.clone())
                .fact(eq(dt, dtype(sum_op)))
                .ruleset("kernel_lower")
                .name(name)
        };

        vec![
            matmul_rule(
                "mps-matmul-row-row",
                MPSMatrixLayout::RowMajor,
                MPSMatrixLayout::RowMajor,
                0,
                0,
            ),
            matmul_rule(
                "mps-matmul-row-transposed-rhs",
                MPSMatrixLayout::RowMajor,
                MPSMatrixLayout::TransposedRowMajor,
                0,
                1,
            ),
            matmul_rule(
                "mps-matmul-transposed-lhs-row",
                MPSMatrixLayout::TransposedRowMajor,
                MPSMatrixLayout::RowMajor,
                1,
                0,
            ),
            matmul_rule(
                "mps-matmul-transposed-lhs-transposed-rhs",
                MPSMatrixLayout::TransposedRowMajor,
                MPSMatrixLayout::TransposedRowMajor,
                1,
                1,
            ),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        _input_enodes: Vec<&'a ENodeId>,
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::egglog_utils::extract_expr;
        let extract_flag = |node: &'a ENodeId| -> bool {
            match egraph.enodes[node].0.as_str() {
                "0" => false,
                "1" => true,
                other => panic!("invalid MPSMatmul transpose flag {other}"),
            }
        };

        (
            LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
                m: extract_expr(egraph, kind_children[0], expr_cache).unwrap(),
                n: extract_expr(egraph, kind_children[1], expr_cache).unwrap(),
                k: extract_expr(egraph, kind_children[2], expr_cache).unwrap(),
                lhs_row_stride: extract_expr(egraph, kind_children[4], expr_cache).unwrap(),
                rhs_row_stride: extract_expr(egraph, kind_children[6], expr_cache).unwrap(),
                out_row_stride: extract_expr(egraph, kind_children[7], expr_cache).unwrap(),
                transpose_lhs: extract_flag(kind_children[8]),
                transpose_rhs: extract_flag(kind_children[9]),
            })),
            vec![kind_children[3], kind_children[5]],
        )
    }
}

impl MPSMatmul {
    fn mps_dtype(dtype: DType) -> mps::MPSDataType {
        match dtype {
            DType::F32 | DType::TF32 => mps::MPSDataType::Float32,
            DType::F16 => mps::MPSDataType::Float16,
            unsupported => panic!("MPSMatmul does not support dtype {unsupported:?}"),
        }
    }

    fn row_bytes(row_stride: Expression, dtype: DType, dyn_map: &FxHashMap<char, usize>) -> u64 {
        let elems = row_stride
            .substitute('z', Expression::from(1))
            .exec(dyn_map)
            .unwrap();
        (elems * dtype.bits().div_ceil(8)) as u64
    }

    fn descriptor(rows: usize, cols: usize, row_bytes: u64, dtype: DType) -> *mut Object {
        let data_type = Self::mps_dtype(dtype) as isize;
        unsafe {
            msg_send![
                class!(MPSMatrixDescriptor),
                matrixDescriptorWithRows: rows
                columns: cols
                rowBytes: row_bytes as usize
                dataType: data_type
            ]
        }
    }

    fn matrix(buffer: &Buffer, descriptor: *mut Object) -> *mut Object {
        unsafe {
            let matrix: *mut Object = msg_send![class!(MPSMatrix), alloc];
            msg_send![matrix, initWithBuffer: buffer.as_ptr() descriptor: descriptor]
        }
    }

    fn matrix_with_offset(
        buffer: &Buffer,
        offset_bytes: u64,
        descriptor: *mut Object,
    ) -> *mut Object {
        unsafe {
            let matrix: *mut Object = msg_send![class!(MPSMatrix), alloc];
            msg_send![
                matrix,
                initWithBuffer: buffer.as_ptr()
                offset: offset_bytes as usize
                descriptor: descriptor
            ]
        }
    }
}

impl MetalKernelOp for MPSMatmul {
    fn compile(
        &self,
        _device: &Device,
        _input_dtypes: &[DType],
        _output_dtype: DType,
    ) -> Option<ComputePipelineState> {
        None
    }

    fn infer_output_dtype(&self, input_dtypes: &[DType]) -> DType {
        input_dtypes.first().copied().unwrap_or(DType::F32)
    }

    fn output_size(&self) -> Expression {
        self.m * self.n
    }

    fn encode_compute(
        &self,
        _encoder: &ComputeCommandEncoderRef,
        _pipeline: &ComputePipelineState,
        _inputs: &[&Buffer],
        _output: &Buffer,
        _dyn_map: &FxHashMap<char, usize>,
    ) {
        panic!("MPSMatmul encodes directly onto the command buffer")
    }

    fn encode(
        &self,
        command_buffer: &CommandBufferRef,
        _pipeline: Option<&ComputePipelineState>,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
        _dyn_buffer: &Buffer,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) {
        assert_eq!(inputs.len(), 2, "MPSMatmul expects lhs and rhs inputs");
        let lhs_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
        let rhs_dtype = input_dtypes.get(1).copied().unwrap_or(lhs_dtype);
        let m = self.m.exec(dyn_map).unwrap();
        let n = self.n.exec(dyn_map).unwrap();
        let k = self.k.exec(dyn_map).unwrap();

        let lhs_rows = if self.transpose_lhs { k } else { m };
        let lhs_cols = if self.transpose_lhs { m } else { k };
        let rhs_rows = if self.transpose_rhs { n } else { k };
        let rhs_cols = if self.transpose_rhs { k } else { n };

        let lhs_desc = Self::descriptor(
            lhs_rows,
            lhs_cols,
            Self::row_bytes(self.lhs_row_stride, lhs_dtype, dyn_map),
            lhs_dtype,
        );
        let rhs_desc = Self::descriptor(
            rhs_rows,
            rhs_cols,
            Self::row_bytes(self.rhs_row_stride, rhs_dtype, dyn_map),
            rhs_dtype,
        );
        let out_desc = Self::descriptor(
            m,
            n,
            Self::row_bytes(self.out_row_stride, output_dtype, dyn_map),
            output_dtype,
        );

        let lhs = Self::matrix(inputs[0], lhs_desc);
        let rhs = Self::matrix(inputs[1], rhs_desc);
        let out = Self::matrix(output, out_desc);

        unsafe {
            let device: *mut Object = msg_send![command_buffer.as_ptr(), device];
            let kernel: *mut Object = msg_send![class!(MPSMatrixMultiplication), alloc];
            let kernel: *mut Object = msg_send![
                kernel,
                initWithDevice: device
                transposeLeft: self.transpose_lhs
                transposeRight: self.transpose_rhs
                resultRows: m
                resultColumns: n
                interiorColumns: k
                alpha: 1.0f64
                beta: 0.0f64
            ];
            let _: () = msg_send![
                kernel,
                encodeToCommandBuffer: command_buffer.as_ptr()
                leftMatrix: lhs
                rightMatrix: rhs
                resultMatrix: out
            ];
            let _: () = msg_send![lhs, release];
            let _: () = msg_send![rhs, release];
            let _: () = msg_send![out, release];
            let _: () = msg_send![kernel, release];
        }
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

#[derive(Debug, Default, Clone)]
pub struct MPSBatchedMatmul {
    pub batch: Expression,
    pub m: Expression,
    pub n: Expression,
    pub k: Expression,
    pub lhs_batch_stride: Expression,
    pub lhs_row_stride: Expression,
    pub rhs_batch_stride: Expression,
    pub rhs_row_stride: Expression,
    pub out_batch_stride: Expression,
    pub out_row_stride: Expression,
    pub transpose_lhs: bool,
    pub transpose_rhs: bool,
}

impl EgglogOp for MPSBatchedMatmul {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "MPSBatchedMatmul",
            &[
                ("batch", EXPRESSION),
                ("m", EXPRESSION),
                ("n", EXPRESSION),
                ("k", EXPRESSION),
                ("lhs", IR),
                ("lhs_batch_stride", EXPRESSION),
                ("lhs_row_stride", EXPRESSION),
                ("rhs", IR),
                ("rhs_batch_stride", EXPRESSION),
                ("rhs_row_stride", EXPRESSION),
                ("out_batch_stride", EXPRESSION),
                ("out_row_stride", EXPRESSION),
                ("transpose_lhs", I64),
                ("transpose_rhs", I64),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        let zero = num(lit_i64(0));
        let z = iter();
        let expr_list = |terms: Vec<EggTerm>| {
            terms
                .into_iter()
                .rev()
                .fold(nil(), |tail, head| cons(head, tail))
        };

        let batched_rule = |name: &'static str,
                            rhs_layout: MPSMatrixLayout,
                            lhs_inner_stride: EggTerm,
                            transpose_rhs: i64| {
            let batch = v("?batch");
            let m = v("?m");
            let n = v("?n");
            let k = v("?k");
            let lhs = v("?lhs");
            let rhs = v("?rhs");
            let lhs_batch_stride = v(format!("?{name}_lhs_batch_stride").replace('-', "_"));
            let lhs_row_stride = v(format!("?{name}_lhs_row_stride").replace('-', "_"));
            let rhs_batch_stride = v(format!("?{name}_rhs_batch_stride").replace('-', "_"));
            let rhs_row_stride = v(format!("?{name}_rhs_row_stride").replace('-', "_"));
            let out_batch_stride = v(format!("?{name}_out_batch_stride").replace('-', "_"));
            let out_row_stride = v(format!("?{name}_out_row_stride").replace('-', "_"));
            let mul_output_strides = v(format!("?{name}_mul_output_strides").replace('-', "_"));

            let rhs_strides = match rhs_layout {
                MPSMatrixLayout::RowMajor => vec![
                    rhs_batch_stride.clone(),
                    zero.clone(),
                    z.clone(),
                    rhs_row_stride.clone(),
                ],
                MPSMatrixLayout::TransposedRowMajor => vec![
                    rhs_batch_stride.clone(),
                    zero.clone(),
                    rhs_row_stride.clone(),
                    z.clone(),
                ],
            };
            let mul_op = op_term(
                MetalMul {
                    shape: vec![],
                    a_strides: vec![],
                    b_strides: vec![],
                    output_strides: vec![],
                }
                .sort()
                .call([
                    (
                        "shape",
                        cons(
                            batch.clone(),
                            cons(m.clone(), cons(n.clone(), cons(k.clone(), nil()))),
                        ),
                    ),
                    (
                        "a_strides",
                        expr_list(vec![
                            lhs_batch_stride.clone(),
                            lhs_row_stride.clone(),
                            zero.clone(),
                            lhs_inner_stride,
                        ]),
                    ),
                    ("b_strides", expr_list(rhs_strides)),
                    ("out_strides", mul_output_strides),
                ]),
                ilist(vec![lhs.clone(), rhs.clone()]),
            );
            let sum_op = op_term(
                MetalSumReduce::default().sort().call([
                    (
                        "shape",
                        cons(batch.clone(), cons(m.clone(), cons(n.clone(), nil()))),
                    ),
                    ("iters", k.clone()),
                    (
                        "strides",
                        v(format!("?{name}_sum_in_strides").replace('-', "_")),
                    ),
                    ("iter_stride", z.clone()),
                    (
                        "out_strides",
                        cons(
                            out_batch_stride.clone(),
                            cons(out_row_stride.clone(), cons(z.clone(), nil())),
                        ),
                    ),
                ]),
                ilist(vec![mul_op.clone()]),
            );
            let mps_op = MPSBatchedMatmul::default().sort().call([
                ("batch", batch),
                ("m", m),
                ("n", n),
                ("k", k),
                ("lhs", lhs),
                ("lhs_batch_stride", lhs_batch_stride),
                ("lhs_row_stride", lhs_row_stride),
                ("rhs", rhs),
                ("rhs_batch_stride", rhs_batch_stride),
                ("rhs_row_stride", rhs_row_stride),
                ("out_batch_stride", out_batch_stride),
                ("out_row_stride", out_row_stride),
                ("transpose_lhs", lit_i64(0)),
                ("transpose_rhs", lit_i64(transpose_rhs)),
            ]);
            let dt = v(format!("?{}_dt", name.replace('-', "_")));

            rule(union(sum_op.clone(), mps_op.clone()))
                .subsume(sum_op.clone())
                .subsume(mul_op)
                .set(dtype(mps_op), dt.clone())
                .fact(eq(dt, dtype(sum_op)))
                .ruleset("kernel_lower")
                .name(name)
        };

        vec![
            batched_rule(
                "mps-batched-matmul-row-row",
                MPSMatrixLayout::RowMajor,
                z.clone(),
                0,
            ),
            batched_rule(
                "mps-batched-matmul-row-transposed-rhs",
                MPSMatrixLayout::TransposedRowMajor,
                z.clone(),
                1,
            ),
            batched_rule(
                "mps-batched-matmul-row-row-wrapped-inner",
                MPSMatrixLayout::RowMajor,
                add(
                    mul(mul(div(z.clone(), v("?k")), v("?k")), v("?m")),
                    modd(z.clone(), v("?k")),
                ),
                0,
            ),
            batched_rule(
                "mps-batched-matmul-row-transposed-rhs-wrapped-inner",
                MPSMatrixLayout::TransposedRowMajor,
                add(
                    mul(mul(div(z.clone(), v("?k")), v("?k")), v("?m")),
                    modd(z.clone(), v("?k")),
                ),
                1,
            ),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        _input_enodes: Vec<&'a ENodeId>,
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::egglog_utils::extract_expr;
        let extract_flag = |node: &'a ENodeId| -> bool {
            match egraph.enodes[node].0.as_str() {
                "0" => false,
                "1" => true,
                other => panic!("invalid MPSBatchedMatmul transpose flag {other}"),
            }
        };

        (
            LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
                batch: extract_expr(egraph, kind_children[0], expr_cache).unwrap(),
                m: extract_expr(egraph, kind_children[1], expr_cache).unwrap(),
                n: extract_expr(egraph, kind_children[2], expr_cache).unwrap(),
                k: extract_expr(egraph, kind_children[3], expr_cache).unwrap(),
                lhs_batch_stride: extract_expr(egraph, kind_children[5], expr_cache).unwrap(),
                lhs_row_stride: extract_expr(egraph, kind_children[6], expr_cache).unwrap(),
                rhs_batch_stride: extract_expr(egraph, kind_children[8], expr_cache).unwrap(),
                rhs_row_stride: extract_expr(egraph, kind_children[9], expr_cache).unwrap(),
                out_batch_stride: extract_expr(egraph, kind_children[10], expr_cache).unwrap(),
                out_row_stride: extract_expr(egraph, kind_children[11], expr_cache).unwrap(),
                transpose_lhs: extract_flag(kind_children[12]),
                transpose_rhs: extract_flag(kind_children[13]),
            })),
            vec![kind_children[4], kind_children[7]],
        )
    }
}

impl MetalKernelOp for MPSBatchedMatmul {
    fn compile(
        &self,
        _device: &Device,
        _input_dtypes: &[DType],
        _output_dtype: DType,
    ) -> Option<ComputePipelineState> {
        None
    }

    fn infer_output_dtype(&self, input_dtypes: &[DType]) -> DType {
        input_dtypes.first().copied().unwrap_or(DType::F32)
    }

    fn output_size(&self) -> Expression {
        self.batch * self.m * self.n
    }

    fn encode_compute(
        &self,
        _encoder: &ComputeCommandEncoderRef,
        _pipeline: &ComputePipelineState,
        _inputs: &[&Buffer],
        _output: &Buffer,
        _dyn_map: &FxHashMap<char, usize>,
    ) {
        panic!("MPSBatchedMatmul encodes directly onto the command buffer")
    }

    fn encode(
        &self,
        command_buffer: &CommandBufferRef,
        _pipeline: Option<&ComputePipelineState>,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
        _dyn_buffer: &Buffer,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) {
        assert_eq!(
            inputs.len(),
            2,
            "MPSBatchedMatmul expects lhs and rhs inputs"
        );
        let lhs_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
        let rhs_dtype = input_dtypes.get(1).copied().unwrap_or(lhs_dtype);
        let batch = self.batch.exec(dyn_map).unwrap();
        let m = self.m.exec(dyn_map).unwrap();
        let n = self.n.exec(dyn_map).unwrap();
        let k = self.k.exec(dyn_map).unwrap();

        let lhs_rows = if self.transpose_lhs { k } else { m };
        let lhs_cols = if self.transpose_lhs { m } else { k };
        let rhs_rows = if self.transpose_rhs { n } else { k };
        let rhs_cols = if self.transpose_rhs { k } else { n };

        let lhs_row_bytes = MPSMatmul::row_bytes(self.lhs_row_stride, lhs_dtype, dyn_map);
        let rhs_row_bytes = MPSMatmul::row_bytes(self.rhs_row_stride, rhs_dtype, dyn_map);
        let out_row_bytes = MPSMatmul::row_bytes(self.out_row_stride, output_dtype, dyn_map);
        let lhs_desc = MPSMatmul::descriptor(lhs_rows, lhs_cols, lhs_row_bytes, lhs_dtype);
        let rhs_desc = MPSMatmul::descriptor(rhs_rows, rhs_cols, rhs_row_bytes, rhs_dtype);
        let out_desc = MPSMatmul::descriptor(m, n, out_row_bytes, output_dtype);

        unsafe {
            let device: *mut Object = msg_send![command_buffer.as_ptr(), device];
            let kernel: *mut Object = msg_send![class!(MPSMatrixMultiplication), alloc];
            let kernel: *mut Object = msg_send![
                kernel,
                initWithDevice: device
                transposeLeft: self.transpose_lhs
                transposeRight: self.transpose_rhs
                resultRows: m
                resultColumns: n
                interiorColumns: k
                alpha: 1.0f64
                beta: 0.0f64
            ];

            for batch_idx in 0..batch {
                let batch_expr = Expression::from(batch_idx as i64);
                let lhs_offset = self
                    .lhs_batch_stride
                    .substitute('z', batch_expr)
                    .exec(dyn_map)
                    .unwrap()
                    * lhs_dtype.bits().div_ceil(8);
                let rhs_offset = self
                    .rhs_batch_stride
                    .substitute('z', batch_expr)
                    .exec(dyn_map)
                    .unwrap()
                    * rhs_dtype.bits().div_ceil(8);
                let out_offset = self
                    .out_batch_stride
                    .substitute('z', batch_expr)
                    .exec(dyn_map)
                    .unwrap()
                    * output_dtype.bits().div_ceil(8);

                let lhs = MPSMatmul::matrix_with_offset(inputs[0], lhs_offset as u64, lhs_desc);
                let rhs = MPSMatmul::matrix_with_offset(inputs[1], rhs_offset as u64, rhs_desc);
                let out = MPSMatmul::matrix_with_offset(output, out_offset as u64, out_desc);
                let _: () = msg_send![
                    kernel,
                    encodeToCommandBuffer: command_buffer.as_ptr()
                    leftMatrix: lhs
                    rightMatrix: rhs
                    resultMatrix: out
                ];
                let _: () = msg_send![lhs, release];
                let _: () = msg_send![rhs, release];
                let _: () = msg_send![out, release];
            }
            let _: () = msg_send![kernel, release];
        }
    }

    fn bytes_loaded(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let batch = self.batch.exec(dyn_map).unwrap_or(0);
        let m = self.m.exec(dyn_map).unwrap_or(0);
        let n = self.n.exec(dyn_map).unwrap_or(0);
        let k = self.k.exec(dyn_map).unwrap_or(0);
        2 * batch * m * n * k * std::mem::size_of::<f32>()
    }

    fn bytes_stored(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let batch = self.batch.exec(dyn_map).unwrap_or(0);
        let m = self.m.exec(dyn_map).unwrap_or(0);
        let n = self.n.exec(dyn_map).unwrap_or(0);
        batch * m * n * std::mem::size_of::<f32>()
    }

    fn flops(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let batch = self.batch.exec(dyn_map).unwrap_or(0);
        let m = self.m.exec(dyn_map).unwrap_or(0);
        let n = self.n.exec(dyn_map).unwrap_or(0);
        let k = self.k.exec(dyn_map).unwrap_or(0);
        2 * batch * m * n * k
    }

    fn is_matmul(&self) -> bool {
        true
    }
}

#[derive(Debug, Default, Clone)]
pub struct GenericMatmul {
    pub out_shape: Vec<Expression>,
    pub mul_shape: Vec<Expression>,
    pub k: Expression,
    pub lhs_strides: Vec<Expression>,
    pub rhs_strides: Vec<Expression>,
    pub sum_input_strides: Vec<Expression>,
    pub sum_iter_stride: Expression,
    pub out_strides: Vec<Expression>,
}

impl EgglogOp for GenericMatmul {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "GenericMatmul",
            &[
                ("out_shape", ELIST),
                ("mul_shape", ELIST),
                ("k", EXPRESSION),
                ("lhs", IR),
                ("lhs_strides", ELIST),
                ("rhs", IR),
                ("rhs_strides", ELIST),
                ("sum_input_strides", ELIST),
                ("sum_iter_stride", EXPRESSION),
                ("out_strides", ELIST),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        let mul_shape = v("?generic_matmul_mul_shape");
        let out_shape = v("?generic_matmul_out_shape");
        let k = v("?generic_matmul_k");
        let lhs = v("?generic_matmul_lhs");
        let rhs = v("?generic_matmul_rhs");
        let lhs_strides = v("?generic_matmul_lhs_strides");
        let rhs_strides = v("?generic_matmul_rhs_strides");
        let mul_output_strides = v("?generic_matmul_mul_output_strides");
        let sum_input_strides = v("?generic_matmul_sum_input_strides");
        let sum_iter_stride = v("?generic_matmul_sum_iter_stride");
        let out_strides = v("?generic_matmul_out_strides");

        let mul_op = op_term(
            MetalMul::default().sort().call([
                ("shape", mul_shape.clone()),
                ("a_strides", lhs_strides.clone()),
                ("b_strides", rhs_strides.clone()),
                ("out_strides", mul_output_strides),
            ]),
            ilist(vec![lhs.clone(), rhs.clone()]),
        );
        let sum_op = op_term(
            MetalSumReduce::default().sort().call([
                ("shape", out_shape.clone()),
                ("iters", k.clone()),
                ("strides", sum_input_strides.clone()),
                ("iter_stride", sum_iter_stride.clone()),
                ("out_strides", out_strides.clone()),
            ]),
            ilist(vec![mul_op.clone()]),
        );
        let generic_op = GenericMatmul::default().sort().call([
            ("out_shape", out_shape),
            ("mul_shape", mul_shape),
            ("k", k),
            ("lhs", lhs),
            ("lhs_strides", lhs_strides),
            ("rhs", rhs),
            ("rhs_strides", rhs_strides),
            ("sum_input_strides", sum_input_strides),
            ("sum_iter_stride", sum_iter_stride),
            ("out_strides", out_strides),
        ]);
        let dt = v("?generic_matmul_dt");

        vec![
            rule(union(sum_op.clone(), generic_op.clone()))
                .set(dtype(generic_op.clone()), dt.clone())
                .fact(eq(dt, dtype(sum_op)))
                .ruleset("matmul_backend")
                .name("generic-matmul-metal-mul-sum"),
            Rule::raw(
                "(rule
                    ((= ?mul (Op (MetalMul ?shape ?as ?bs ?os) ?inputs))
                     (= ?sum (Op (MetalSum ?sshape ?sk ?ssi ?sks ?sso) (ICons ?mul (INil))))
                     (= ?sum (GenericMatmul ?go ?gm ?gk ?gl ?glas ?gr ?grs ?gsis ?gsit ?gos)))
                    ((delete (Op (MetalSum ?sshape ?sk ?ssi ?sks ?sso) (ICons ?mul (INil))))
                     (delete (Op (MetalMul ?shape ?as ?bs ?os) ?inputs)))
                    :ruleset cleanup
                    :name \"delete-broadcast-mul-sum-when-generic-matmul-exists\"
                )",
            ),
            Rule::raw(
                "(rule
                    ((= ?sum (GenericMatmul ?go ?gm ?gk ?gl ?glas ?gr ?grs ?gsis ?gsit ?gos))
                     (= ?sum (MPSMatmul ?mm ?mn ?mk ?ml ?mls ?mr ?mrs ?mos ?mtl ?mtr)))
                    ((delete (GenericMatmul ?go ?gm ?gk ?gl ?glas ?gr ?grs ?gsis ?gsit ?gos)))
                    :ruleset cleanup
                    :name \"prefer-mps-over-generic-matmul\"
                )",
            ),
            Rule::raw(
                "(rule
                    ((= ?sum (GenericMatmul ?go ?gm ?gk ?gl ?glas ?gr ?grs ?gsis ?gsit ?gos))
                     (= ?sum (MPSBatchedMatmul ?bb ?bm ?bn ?bk ?bl ?blbs ?blrs ?br ?brbs ?brrs ?bobs ?bors ?btl ?btr)))
                    ((delete (GenericMatmul ?go ?gm ?gk ?gl ?glas ?gr ?grs ?gsis ?gsit ?gos)))
                    :ruleset cleanup
                    :name \"prefer-mps-batched-over-generic-matmul\"
                )",
            ),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        _input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::egglog_utils::{extract_expr, extract_expr_list};
        (
            LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache)
                    .unwrap(),
                mul_shape: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                k: extract_expr(egraph, kind_children[2], expr_cache).unwrap(),
                lhs_strides: extract_expr_list(egraph, kind_children[4], list_cache, expr_cache)
                    .unwrap(),
                rhs_strides: extract_expr_list(egraph, kind_children[6], list_cache, expr_cache)
                    .unwrap(),
                sum_input_strides: extract_expr_list(
                    egraph,
                    kind_children[7],
                    list_cache,
                    expr_cache,
                )
                .unwrap(),
                sum_iter_stride: extract_expr(egraph, kind_children[8], expr_cache).unwrap(),
                out_strides: extract_expr_list(egraph, kind_children[9], list_cache, expr_cache)
                    .unwrap(),
            })),
            vec![kind_children[3], kind_children[5]],
        )
    }
}

impl MetalKernelOp for GenericMatmul {
    fn compile(
        &self,
        device: &Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<ComputePipelineState> {
        let lhs_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
        let rhs_dtype = input_dtypes.get(1).copied().unwrap_or(lhs_dtype);
        let lhs_ty = metal_buffer_type(lhs_dtype);
        let rhs_ty = metal_buffer_type(rhs_dtype);
        let out_ty = metal_buffer_type(output_dtype);

        let sum_base = flatten_strides(&self.out_shape, &self.sum_input_strides);
        let sum_base_idx = lower_expression_for_metal(&sum_base, "gid");
        let iter_offset = lower_expression_for_metal(&self.sum_iter_stride, "i");
        let lhs_index = flatten_strides(&self.mul_shape, &self.lhs_strides);
        let rhs_index = flatten_strides(&self.mul_shape, &self.rhs_strides);
        let out_index = flatten_strides(&self.out_shape, &self.out_strides);
        let lhs_idx = lower_expression_for_metal(&lhs_index, "mul_idx");
        let rhs_idx = lower_expression_for_metal(&rhs_index, "mul_idx");
        let out_idx = lower_expression_for_metal(&out_index, "gid");
        let iters = lower_expression_for_metal(&self.k, "gid");
        let lhs_val = metal_numeric_read(lhs_dtype, "lhs", &lhs_idx);
        let rhs_val = metal_numeric_read(rhs_dtype, "rhs", &rhs_idx);
        let out_val = metal_numeric_write(output_dtype, "block_sum");

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            #define THREADS_PER_GROUP 256

            kernel void mkernel(
                const device {lhs_ty} *lhs [[buffer(0)]],
                const device {rhs_ty} *rhs [[buffer(1)]],
                device {out_ty} *out [[buffer(2)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                constant uint &n_outputs [[buffer({n_outputs_index})]],
                uint gid [[threadgroup_position_in_grid]],
                uint tid [[thread_index_in_threadgroup]],
                uint simd_lane [[thread_index_in_simdgroup]],
                uint simd_id [[simdgroup_index_in_threadgroup]]
            ) {{
                if (gid >= n_outputs) return;

                threadgroup float warp_sums[THREADS_PER_GROUP / 32];
                int base_idx = {sum_base_idx};
                int iters = {iters};
                (void)dyn;

                float sum = 0.0f;
                for (int i = tid; i < iters; i += THREADS_PER_GROUP) {{
                    int mul_idx = base_idx + {iter_offset};
                    sum += ({lhs_val}) * ({rhs_val});
                }}

                sum = simd_sum(sum);
                if (simd_lane == 0) {{
                    warp_sums[simd_id] = sum;
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);

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
            lhs_ty = lhs_ty,
            rhs_ty = rhs_ty,
            out_ty = out_ty,
            sum_base_idx = sum_base_idx,
            iters = iters,
            iter_offset = iter_offset,
            lhs_val = lhs_val,
            rhs_val = rhs_val,
            out_idx = out_idx,
            out_val = out_val,
            dyn_buffer_index = 3u64,
            n_outputs_index = 4u64,
        );
        Some(compile_shader(device, &source, "mkernel"))
    }

    fn infer_output_dtype(&self, input_dtypes: &[DType]) -> DType {
        input_dtypes.first().copied().unwrap_or(DType::F32)
    }

    fn output_size(&self) -> Expression {
        self.out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn encode_compute(
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
        encoder.set_buffer(1, Some(inputs[1]), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &n_outputs as *const u32 as *const _,
        );

        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new(n_outputs as u64, 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    }

    fn bytes_loaded(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let n_outputs = self.output_size().exec(dyn_map).unwrap_or(0);
        let k = self.k.exec(dyn_map).unwrap_or(0);
        2 * n_outputs * k * std::mem::size_of::<f32>()
    }

    fn bytes_stored(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        self.output_size().exec(dyn_map).unwrap_or(0) * std::mem::size_of::<f32>()
    }

    fn flops(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let n_outputs = self.output_size().exec(dyn_map).unwrap_or(0);
        let k = self.k.exec(dyn_map).unwrap_or(0);
        2 * n_outputs * k
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
        vec![
            rule(union(const_match.clone(), metal_op.clone()))
                .subsume(const_match)
                .set(dtype(metal_op), app(&SORTS.f32_dt, vec![]))
                .ruleset("kernel_lower"),
        ]
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
    ) -> Option<ComputePipelineState> {
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
        Some(compile_shader(device, &source, "mkernel"))
    }

    fn output_size(&self) -> Expression {
        Expression::from(1)
    }

    fn infer_output_dtype(&self, _input_dtypes: &[DType]) -> DType {
        DType::F32
    }

    fn encode_compute(
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
        vec![
            rule(union(iota_match.clone(), metal_op.clone()))
                .subsume(iota_match)
                .set(dtype(metal_op), app(&SORTS.int_dt, vec![]))
                .ruleset("kernel_lower"),
        ]
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
    ) -> Option<ComputePipelineState> {
        // Generate the expression as Metal code
        let expr_code = lower_expression_for_metal(&self.expr, "idx");

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                device int *out [[buffer(0)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                constant uint &n_elements [[buffer({n_elements_index})]],
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
        Some(compile_shader(device, &source, "mkernel"))
    }

    fn output_size(&self) -> Expression {
        self.range
    }

    fn infer_output_dtype(&self, _input_dtypes: &[DType]) -> DType {
        DType::Int
    }

    fn encode_compute(
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
    data_shape: Vec<Expression>,
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
                ("data_shape", ELIST),
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
            ("data_shape".to_string(), gather_args["data_shape"].clone()),
            (
                "data_strides".to_string(),
                gather_args["data_strides"].clone(),
            ),
            ("out_strides".to_string(), out_strides),
        ];
        let metal_op = self.sort().call(metal_args);
        vec![
            rule(union(gather_match.clone(), metal_op.clone()))
                .subsume(gather_match)
                .set(dtype(metal_op), dt.clone())
                .fact(eq(dt, dtype(gather_args["data"].clone())))
                .ruleset("kernel_lower"),
        ]
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
                data_shape: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
                data_stride: extract_expr_list(egraph, children[5], list_cache, expr_cache)
                    .unwrap(),
                out_stride: extract_expr_list(egraph, children[6], list_cache, expr_cache).unwrap(),
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
    ) -> Option<ComputePipelineState> {
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
            &flatten_strides(&self.data_shape, &self.data_stride),
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
                constant uint &n_elements [[buffer({n_elements_index})]],
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
        Some(compile_shader(device, &source, "mkernel"))
    }

    fn output_size(&self) -> Expression {
        self.out_shape
            .iter()
            .cloned()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn infer_output_dtype(&self, input_dtypes: &[DType]) -> DType {
        input_dtypes.get(1).copied().unwrap_or(DType::F32)
    }

    fn encode_compute(
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
        vec![
            rule(union(scatter_match.clone(), metal_op.clone()))
                .subsume(scatter_match)
                .set(dtype(metal_op), dt.clone())
                .fact(eq(dt, dtype(scatter_args["src"].clone())))
                .ruleset("kernel_lower"),
        ]
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
    ) -> Option<ComputePipelineState> {
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
                constant uint &n_elements [[buffer(2)]],
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
                constant uint &n_elements [[buffer(3)]],
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
        Some(compile_shader(device, &scatter_source, "scatter_kernel"))
    }

    fn output_size(&self) -> Expression {
        self.dest_shape
            .iter()
            .cloned()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn encode_compute(
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

#[derive(Debug, Default, Clone)]
#[allow(dead_code)]
pub struct MetalScatterNoCopy {
    dest_shape: Vec<Expression>,
    dest_strides: Vec<Expression>,
    index_shape: Vec<Expression>,
    index_strides: Vec<Expression>,
    src_strides: Vec<Expression>,
    out_strides: Vec<Expression>,
}

impl EgglogOp for MetalScatterNoCopy {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "MetalScatterNoCopy",
            &[
                ("dest_shape", ELIST),
                ("dest_strides", ELIST),
                ("index_shape", ELIST),
                ("index_strides", ELIST),
                ("src_strides", ELIST),
                ("out_strides", ELIST),
            ],
        )
    }

    fn ir_defs(&self) -> Vec<String> {
        vec!["(ConsumedBuffer IR)".to_string()]
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![
            Rule::raw("(relation consumed_buffer_ilist_contains (IList IR))"),
            Rule::raw(
                "(rule
                    ((= ?list (ICons ?head ?tail)))
                    ((consumed_buffer_ilist_contains ?list ?head))
                    :ruleset cleanup
                    :name \"metal-consumed-buffer-ilist-contains-head\"
                )",
            ),
            Rule::raw(
                "(rule
                    ((= ?list (ICons ?head ?tail))
                     (consumed_buffer_ilist_contains ?tail ?item))
                    ((consumed_buffer_ilist_contains ?list ?item))
                    :ruleset cleanup
                    :name \"metal-consumed-buffer-ilist-contains-tail\"
                )",
            ),
            Rule::raw(
                "(rule
                    ((= ?scatter (MetalScatter ?ds ?dst ?dest ?indexes ?is ?istr ?src ?ss ?os))
                     (= ?dst ?os)
                     (= ?dt (dtype ?src)))
                    ((let ?consumed (ConsumedBuffer ?dest))
                     (let ?nocopy (Op (MetalScatterNoCopy ?ds ?dst ?is ?istr ?ss ?os)
                         (ICons ?consumed (ICons ?indexes (ICons ?src (INil))))))
                     (union ?scatter ?nocopy)
                     (set (dtype ?nocopy) ?dt))
                    :ruleset buffer_reuse
                    :name \"metal-scatter-to-scatter-no-copy\"
                )",
            ),
            Rule::raw(
                "(rule
                    ((= ?cb (ConsumedBuffer ?a))
                     (= ?dt (dtype ?a)))
                    ((set (dtype ?cb) ?dt))
                    :ruleset dtype_prop
                    :name \"metal-consumed-buffer-dtype\"
                )",
            ),
            Rule::raw(
                "(rule
                    ((= ?cb (ConsumedBuffer ?a))
                     (= ?op1 (Op ?k1 ?ilist1))
                     (consumed_buffer_ilist_contains ?ilist1 ?cb)
                     (= ?op2 (Op ?k2 ?ilist2))
                     (!= ?op1 ?op2)
                     (consumed_buffer_ilist_contains ?ilist2 ?a))
                    ((delete (ConsumedBuffer ?a)))
                    :ruleset cleanup
                    :name \"metal-consumed-buffer-cleanup-shared-op-use\"
                )",
            ),
            Rule::raw(
                "(rule
                    ((= ?cb (ConsumedBuffer ?dest))
                     (= ?scatter (MetalScatter ?ds ?dst ?dest ?indexes ?is ?istr ?src ?ss ?os))
                     (= ?nocopy (Op (MetalScatterNoCopy ?ds ?dst ?is ?istr ?ss ?os)
                         (ICons ?cb (ICons ?indexes (ICons ?src (INil)))))))
                    ((subsume (MetalScatter ?ds ?dst ?dest ?indexes ?is ?istr ?src ?ss ?os)))
                    :ruleset post_cleanup
                    :name \"metal-scatter-no-copy-dominates-valid-consumed-buffer\"
                )",
            ),
            Rule::raw(
                "(rule
                    ((= ?cb (ConsumedBuffer ?a)))
                    ((union ?cb ?a)
                     (delete (ConsumedBuffer ?a)))
                    :ruleset base_cleanup
                    :name \"metal-consumed-buffer-resolve\"
                )",
            ),
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
                dest_shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache)
                    .unwrap(),
                dest_strides: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                index_shape: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                index_strides: extract_expr_list(egraph, kind_children[3], list_cache, expr_cache)
                    .unwrap(),
                src_strides: extract_expr_list(egraph, kind_children[4], list_cache, expr_cache)
                    .unwrap(),
                out_strides: extract_expr_list(egraph, kind_children[5], list_cache, expr_cache)
                    .unwrap(),
            })),
            input_enodes,
        )
    }
}

impl MetalKernelOp for MetalScatterNoCopy {
    fn compile(
        &self,
        device: &Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<ComputePipelineState> {
        let src_dtype = input_dtypes.get(2).copied().unwrap_or(output_dtype);
        let src_ty = metal_buffer_type(src_dtype);
        let out_ty = metal_buffer_type(output_dtype);
        let index_idx = lower_expression_for_metal(
            &flatten_strides(&self.index_shape, &self.index_strides),
            "idx",
        );
        let src_idx = lower_expression_for_metal(
            &flatten_strides(&self.index_shape, &self.src_strides),
            "idx",
        );
        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void scatter_no_copy_kernel(
                device {out_ty} *out [[buffer(0)]],
                const device int *indexes [[buffer(1)]],
                const device {src_ty} *src [[buffer(2)]],
                constant uint &n_elements [[buffer(3)]],
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
        Some(compile_shader(device, &source, "scatter_no_copy_kernel"))
    }

    fn infer_output_dtype(&self, input_dtypes: &[DType]) -> DType {
        input_dtypes.first().copied().unwrap_or(DType::F32)
    }

    fn output_size(&self) -> Expression {
        self.dest_shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn encode_compute(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    ) {
        let n_src = self
            .index_shape
            .iter()
            .copied()
            .product::<Expression>()
            .exec(dyn_map)
            .unwrap() as u32;

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(output), 0);
        encoder.set_buffer(1, Some(inputs[1]), 0);
        encoder.set_buffer(2, Some(inputs[2]), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &n_src as *const u32 as *const _,
        );
        let thread_group_size = MTLSize::new(256, 1, 1);
        encoder.dispatch_thread_groups(
            MTLSize::new((n_src as u64).div_ceil(256), 1, 1),
            thread_group_size,
        );
    }

    fn bytes_loaded(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let n_src = self
            .index_shape
            .iter()
            .copied()
            .product::<Expression>()
            .exec(dyn_map)
            .unwrap_or(0);
        n_src * std::mem::size_of::<i32>() + n_src * std::mem::size_of::<f32>()
    }

    fn bytes_stored(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let n_src = self
            .index_shape
            .iter()
            .copied()
            .product::<Expression>()
            .exec(dyn_map)
            .unwrap_or(0);
        n_src * std::mem::size_of::<f32>()
    }

    fn flops(&self, _dyn_map: &FxHashMap<char, usize>) -> usize {
        0
    }

    fn output_aliases_input(&self) -> Option<usize> {
        Some(0)
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
        vec![
            rule(union(cast_match.clone(), metal_op.clone()))
                .subsume(cast_match)
                .set(dtype(metal_op), args["dtype"].clone())
                .ruleset("kernel_lower"),
        ]
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
    ) -> Option<ComputePipelineState> {
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
                constant uint &n_elements [[buffer({n_elements_index})]],
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
        Some(compile_shader(device, &source, "mkernel"))
    }

    fn output_size(&self) -> Expression {
        self.size
    }

    fn infer_output_dtype(&self, _input_dtypes: &[DType]) -> DType {
        self.target_dtype
    }

    fn encode_compute(
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
