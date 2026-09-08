use super::{
    COUNT_PARAM_SLOT, OFFSET_PARAM_SLOT, PARAM_SLOT_COUNT, WORKGROUP_SIZE, WebGpuEncodeContext,
    WebGpuKernelOp, WebGpuMatrixLayout, WebGpuPipeline,
};
use crate::runtime::WebGpuBuffer;
use luminal::{
    egglog_utils::{
        SerializedEGraph,
        api::{
            Args, Rule, SortDef, Term as EggTerm, app, eq, i64 as lit_i64, rule, sort, union, v,
        },
        base::{
            DTYPE, ELIST, EXPRESSION, F64, I64, IR, SORTS, cons, dtype, ilist, iter, mul,
            new_op_call, nil, num, op_term,
        },
    },
    hlir::{
        Add, Cast, Constant, Gather, Iota, LessThan, MaxReduce, Mod, Mul, Scatter, SumReduce,
        binary_sort, reduce_sort, unary_sort,
    },
    op::*,
    prelude::*,
    shape::{Term, flatten_strides},
};
use wgpu::util::DeviceExt;

pub type WebGpuOps = (
    WebGpuExp2,
    WebGpuLog2,
    WebGpuSin,
    WebGpuSqrt,
    WebGpuRecip,
    WebGpuAdd,
    WebGpuMul,
    WebGpuMod,
    WebGpuLessThan,
    WebGpuSumReduce,
    WebGpuMaxReduce,
    WebGpuMatmul,
    GenericMatmul,
    WebGpuConstant,
    WebGpuIota,
    WebGpuGather,
    WebGpuScatter,
    WebGpuScatterNoCopy,
    WebGpuCast,
);

fn compile_shader_raw(
    device: &wgpu::Device,
    source: &str,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point,
    })
}

fn compile_shader(device: &wgpu::Device, source: &str, entry_point: &str) -> WebGpuPipeline {
    WebGpuPipeline::single(compile_shader_raw(device, source, entry_point))
}

fn shader_prelude() -> String {
    format!(
        r#"
struct Params {{
    dims: array<i32, {dyn_slots}>,
    n: i32,
    offset: i32,
}}
"#,
        dyn_slots = super::DYN_SLOT_COUNT
    )
}

fn params_buffer(
    device: &wgpu::Device,
    dyn_map: &FxHashMap<char, usize>,
    count: u32,
    offset: u32,
) -> wgpu::Buffer {
    let mut params = [0i32; PARAM_SLOT_COUNT];
    for (&symbol, &value) in dyn_map {
        if symbol.is_ascii_lowercase() {
            let slot = (symbol as u8 - b'a') as usize;
            if slot < super::DYN_SLOT_COUNT {
                params[slot] = value as i32;
            }
        }
    }
    params[COUNT_PARAM_SLOT] = count as i32;
    params[OFFSET_PARAM_SLOT] = offset as i32;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn dispatch_chunk(
    context: &mut WebGpuEncodeContext<'_>,
    pipeline: &wgpu::ComputePipeline,
    buffers: &[&WebGpuBuffer],
    count: u32,
    offset: u32,
    workgroups: u32,
) {
    if workgroups == 0 {
        return;
    }
    let params = params_buffer(context.device, context.dyn_map, count, offset);
    let mut entries = buffers
        .iter()
        .enumerate()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: buffer.raw().as_entire_binding(),
        })
        .collect::<Vec<_>>();
    entries.push(wgpu::BindGroupEntry {
        binding: buffers.len() as u32,
        resource: params.as_entire_binding(),
    });
    let bind_group = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
    let mut pass = context
        .encoder
        .begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(workgroups, 1, 1);
}

fn dispatch_1d(
    context: &mut WebGpuEncodeContext<'_>,
    pipeline: &wgpu::ComputePipeline,
    buffers: &[&WebGpuBuffer],
    count: u32,
) {
    if count == 0 {
        return;
    }
    assert!(
        count <= i32::MAX as u32,
        "WebGPU kernels currently index buffers with i32 and cannot dispatch {count} elements"
    );
    let max_workgroups = context.max_compute_workgroups_per_dimension.max(1);
    let max_elements_per_dispatch = max_workgroups.saturating_mul(WORKGROUP_SIZE).max(1);
    let mut offset = 0u32;
    while offset < count {
        let chunk_elements = (count - offset).min(max_elements_per_dispatch);
        let workgroups = chunk_elements.div_ceil(WORKGROUP_SIZE);
        dispatch_chunk(context, pipeline, buffers, count, offset, workgroups);
        offset += chunk_elements;
    }
}

fn dispatch_workgroups(
    context: &mut WebGpuEncodeContext<'_>,
    pipeline: &wgpu::ComputePipeline,
    buffers: &[&WebGpuBuffer],
    count: u32,
    workgroups: u32,
) {
    if workgroups == 0 {
        return;
    }
    assert!(
        count <= i32::MAX as u32 && workgroups <= i32::MAX as u32,
        "WebGPU kernels currently index workgroups with i32 and cannot dispatch count={count}, workgroups={workgroups}"
    );
    let max_workgroups = context.max_compute_workgroups_per_dimension.max(1);
    let mut offset = 0u32;
    while offset < workgroups {
        let chunk_workgroups = (workgroups - offset).min(max_workgroups);
        dispatch_chunk(context, pipeline, buffers, count, offset, chunk_workgroups);
        offset += chunk_workgroups;
    }
}

fn dispatch_2d(
    context: &mut WebGpuEncodeContext<'_>,
    pipeline: &wgpu::ComputePipeline,
    buffers: &[&WebGpuBuffer],
    count: u32,
    workgroups_x: u32,
    workgroups_y: u32,
) {
    if workgroups_x == 0 || workgroups_y == 0 {
        return;
    }
    let max_workgroups = context.max_compute_workgroups_per_dimension.max(1);
    assert!(
        workgroups_x <= max_workgroups && workgroups_y <= max_workgroups,
        "WebGPU 2D dispatch dimensions [{workgroups_x}, {workgroups_y}] exceed adapter limit {max_workgroups}"
    );
    let params = params_buffer(context.device, context.dyn_map, count, 0);
    let mut entries = buffers
        .iter()
        .enumerate()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: buffer.raw().as_entire_binding(),
        })
        .collect::<Vec<_>>();
    entries.push(wgpu::BindGroupEntry {
        binding: buffers.len() as u32,
        resource: params.as_entire_binding(),
    });
    let bind_group = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
    let mut pass = context
        .encoder
        .begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}

pub(crate) fn lower_expression_for_webgpu(expr: &Expression, index_var: &str) -> String {
    let mut stack = Vec::new();
    for term in expr.terms.read().iter().copied() {
        let value = match term {
            Term::Num(n) => n.to_string(),
            Term::Var('z') => index_var.to_string(),
            Term::Var(c) => {
                assert!(c.is_ascii_lowercase(), "unsupported dynamic symbol {c:?}");
                format!("params.dims[{}]", (c as u8 - b'a') as usize)
            }
            Term::Add => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                format!("(({a}) + ({b}))")
            }
            Term::Sub => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                format!("(({a}) - ({b}))")
            }
            Term::Mul => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                format!("(({a}) * ({b}))")
            }
            Term::Div => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                format!("(({a}) / ({b}))")
            }
            Term::CeilDiv => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                format!("((({a}) + ({b}) - 1) / ({b}))")
            }
            Term::Mod => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                format!("(({a}) % ({b}))")
            }
            Term::Min => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                format!("min({a}, {b})")
            }
            Term::Max => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                format!("max({a}, {b})")
            }
            Term::And => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                format!("select(0, 1, (({a}) != 0 && ({b}) != 0))")
            }
            Term::Or => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                format!("select(0, 1, (({a}) != 0 || ({b}) != 0))")
            }
            Term::Gte => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                format!("select(0, 1, ({a}) >= ({b}))")
            }
            Term::Lt => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                format!("select(0, 1, ({a}) < ({b}))")
            }
        };
        stack.push(value);
    }
    stack.pop().unwrap_or_else(|| "0".to_string())
}

pub(crate) fn lower_scalar_expression_for_webgpu(expr: &Expression) -> String {
    lower_expression_for_webgpu(&expr.substitute('z', Expression::from(1)), "idx")
}

fn webgpu_buffer_type(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 | DType::TF32 => "f32",
        DType::Int => "i32",
        _ => panic!("WebGPU dtype {dtype:?} is not supported yet"),
    }
}

fn buffer_index(index: &str) -> String {
    format!("u32({index})")
}

fn webgpu_numeric_read(dtype: DType, buffer: &str, index: &str) -> String {
    match dtype {
        DType::F32 | DType::TF32 => format!("{buffer}[{}]", buffer_index(index)),
        DType::Int => format!("f32({buffer}[{}])", buffer_index(index)),
        _ => panic!("WebGPU dtype {dtype:?} is not supported yet"),
    }
}

fn webgpu_numeric_write(dtype: DType, expr: &str) -> String {
    match dtype {
        DType::F32 | DType::TF32 => expr.to_string(),
        DType::Int => format!("i32({expr})"),
        _ => panic!("WebGPU dtype {dtype:?} is not supported yet"),
    }
}

fn webgpu_copy_value(dtype: DType, buffer: &str, index: &str) -> String {
    match dtype {
        DType::F32 | DType::TF32 | DType::Int => {
            format!("{buffer}[{}]", buffer_index(index))
        }
        _ => panic!("WebGPU dtype {dtype:?} is not supported yet"),
    }
}

fn webgpu_binary_op_values(
    output_dtype: DType,
    a_dtype: DType,
    b_dtype: DType,
    a_idx: &str,
    b_idx: &str,
) -> (String, String) {
    let read: fn(DType, &str, &str) -> String = if output_dtype == DType::Int {
        webgpu_copy_value
    } else {
        webgpu_numeric_read
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

fn unary_dtype_rewrite(hlir_sort: &SortDef, webgpu_sort: &SortDef) -> Rule {
    let (args, hlir_match) = new_op_call(hlir_sort, &["inp"]);
    let webgpu_op = op_term(
        call_sort_from_args(webgpu_sort, &args),
        args["__inputs"].clone(),
    );
    let dt = v("?__dt");
    rule(union(hlir_match.clone(), webgpu_op.clone()))
        .subsume(hlir_match)
        .set(dtype(webgpu_op), dt.clone())
        .fact(eq(dt, dtype(args["inp"].clone())))
        .ruleset("kernel_lower")
}

fn binary_dtype_rewrite(hlir_sort: &SortDef, webgpu_sort: &SortDef) -> Rule {
    let (args, hlir_match) = new_op_call(hlir_sort, &["inp_a", "inp_b"]);
    let webgpu_op = op_term(
        call_sort_from_args(webgpu_sort, &args),
        args["__inputs"].clone(),
    );
    let dt = v("?__dt");
    rule(union(hlir_match.clone(), webgpu_op.clone()))
        .subsume(hlir_match)
        .set(dtype(webgpu_op), dt.clone())
        .fact(eq(dt, dtype(args["inp_a"].clone())))
        .ruleset("kernel_lower")
}

macro_rules! impl_unary_metrics {
    ($self:ident, $dyn_map:ident) => {
        fn bytes_loaded(&$self, $dyn_map: &FxHashMap<char, usize>) -> usize {
            $self.output_size().exec($dyn_map).unwrap_or(0) * std::mem::size_of::<f32>()
        }

        fn bytes_stored(&$self, $dyn_map: &FxHashMap<char, usize>) -> usize {
            $self.output_size().exec($dyn_map).unwrap_or(0) * std::mem::size_of::<f32>()
        }

        fn flops(&$self, $dyn_map: &FxHashMap<char, usize>) -> usize {
            $self.output_size().exec($dyn_map).unwrap_or(0)
        }
    };
}

macro_rules! impl_binary_metrics {
    ($self:ident, $dyn_map:ident, $flops_per_elem:expr) => {
        fn bytes_loaded(&$self, $dyn_map: &FxHashMap<char, usize>) -> usize {
            $self.output_size().exec($dyn_map).unwrap_or(0) * 2 * std::mem::size_of::<f32>()
        }

        fn bytes_stored(&$self, $dyn_map: &FxHashMap<char, usize>) -> usize {
            $self.output_size().exec($dyn_map).unwrap_or(0) * std::mem::size_of::<f32>()
        }

        fn flops(&$self, $dyn_map: &FxHashMap<char, usize>) -> usize {
            $self.output_size().exec($dyn_map).unwrap_or(0) * $flops_per_elem
        }
    };
}

macro_rules! impl_reduce_metrics {
    ($self:ident, $dyn_map:ident) => {
        fn bytes_loaded(&$self, $dyn_map: &FxHashMap<char, usize>) -> usize {
            let n_outputs = $self.output_size().exec($dyn_map).unwrap_or(0);
            let iters = $self.iters.exec($dyn_map).unwrap_or(0);
            n_outputs * iters * std::mem::size_of::<f32>()
        }

        fn bytes_stored(&$self, $dyn_map: &FxHashMap<char, usize>) -> usize {
            $self.output_size().exec($dyn_map).unwrap_or(0) * std::mem::size_of::<f32>()
        }

        fn flops(&$self, $dyn_map: &FxHashMap<char, usize>) -> usize {
            let n_outputs = $self.output_size().exec($dyn_map).unwrap_or(0);
            let iters = $self.iters.exec($dyn_map).unwrap_or(0);
            n_outputs * iters
        }
    };
}

macro_rules! webgpu_unary_op {
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
                let hlir_name = ($op_name).strip_prefix("WebGpu").unwrap_or($op_name);
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
                    LLIROp::new::<dyn WebGpuKernelOp>(Box::new(Self {
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

        impl WebGpuKernelOp for $name {
            fn compile(
                &self,
                device: &wgpu::Device,
                input_dtypes: &[DType],
                output_dtype: DType,
            ) -> Option<WebGpuPipeline> {
                let input_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
                let input_ty = webgpu_buffer_type(input_dtype);
                let output_ty = webgpu_buffer_type(output_dtype);
                let inp_index = flatten_strides(&self.shape, &self.input_strides);
                let out_index = flatten_strides(&self.shape, &self.output_strides);
                let inp_idx = lower_expression_for_webgpu(&inp_index, "idx");
                let out_idx = lower_expression_for_webgpu(&out_index, "idx");
                let input_expr = webgpu_numeric_read(input_dtype, "inp", &inp_idx);
                let body_expr = ($expr_builder)(&input_expr);
                let write_expr = webgpu_numeric_write(output_dtype, &body_expr);
                let prelude = shader_prelude();
                let source = format!(
                    r#"
{prelude}
@group(0) @binding(0) var<storage, read> inp: array<{input_ty}>;
@group(0) @binding(1) var<storage, read_write> out: array<{output_ty}>;
@group(0) @binding(2) var<storage, read> params: Params;

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = i32(global_id.x) + params.offset;
    if (idx < params.n) {{
        out[{out_index}] = {write_expr};
    }}
}}
"#,
                    out_index = buffer_index(&out_idx),
                    workgroup_size = WORKGROUP_SIZE,
                );
                Some(compile_shader(device, &source, "main"))
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
                context: &mut WebGpuEncodeContext<'_>,
                pipeline: &WebGpuPipeline,
                inputs: &[&WebGpuBuffer],
                output: &WebGpuBuffer,
                _input_dtypes: &[DType],
                _output_dtype: DType,
            ) {
                let n_elements = self.output_size().exec(context.dyn_map).unwrap() as u32;
                dispatch_1d(context, pipeline.get(0), &[inputs[0], output], n_elements);
            }

            impl_unary_metrics!(self, dyn_map);
        }
    };
}

webgpu_unary_op!(WebGpuExp2, "WebGpuExp2", |x: &str| format!("exp2({x})"));
webgpu_unary_op!(WebGpuLog2, "WebGpuLog2", |x: &str| format!("log2({x})"));
webgpu_unary_op!(WebGpuSin, "WebGpuSin", |x: &str| format!("sin({x})"));
webgpu_unary_op!(WebGpuSqrt, "WebGpuSqrt", |x: &str| format!("sqrt({x})"));
webgpu_unary_op!(WebGpuRecip, "WebGpuRecip", |x: &str| format!("1.0 / ({x})"));

macro_rules! webgpu_binary_op {
    ($name:ident, $sort_name:expr, $hlir:ty, $expr_builder:expr, $flops:expr) => {
        #[derive(Debug, Default, Clone)]
        pub struct $name {
            shape: Vec<Expression>,
            a_strides: Vec<Expression>,
            b_strides: Vec<Expression>,
            output_strides: Vec<Expression>,
        }

        impl EgglogOp for $name {
            fn sort(&self) -> SortDef {
                binary_sort($sort_name)
            }

            fn rewrites(&self) -> Vec<Rule> {
                vec![binary_dtype_rewrite(
                    &<$hlir>::default().sort(),
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
                    LLIROp::new::<dyn WebGpuKernelOp>(Box::new(Self {
                        shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache)
                            .unwrap(),
                        a_strides: extract_expr_list(
                            egraph,
                            kind_children[1],
                            list_cache,
                            expr_cache,
                        )
                        .unwrap(),
                        b_strides: extract_expr_list(
                            egraph,
                            kind_children[2],
                            list_cache,
                            expr_cache,
                        )
                        .unwrap(),
                        output_strides: extract_expr_list(
                            egraph,
                            kind_children[3],
                            list_cache,
                            expr_cache,
                        )
                        .unwrap(),
                    })),
                    input_enodes,
                )
            }
        }

        impl WebGpuKernelOp for $name {
            fn compile(
                &self,
                device: &wgpu::Device,
                input_dtypes: &[DType],
                output_dtype: DType,
            ) -> Option<WebGpuPipeline> {
                let a_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
                let b_dtype = input_dtypes.get(1).copied().unwrap_or(a_dtype);
                let a_ty = webgpu_buffer_type(a_dtype);
                let b_ty = webgpu_buffer_type(b_dtype);
                let out_ty = webgpu_buffer_type(output_dtype);
                let a_index = flatten_strides(&self.shape, &self.a_strides);
                let b_index = flatten_strides(&self.shape, &self.b_strides);
                let out_index = flatten_strides(&self.shape, &self.output_strides);
                let a_idx = lower_expression_for_webgpu(&a_index, "idx");
                let b_idx = lower_expression_for_webgpu(&b_index, "idx");
                let out_idx = lower_expression_for_webgpu(&out_index, "idx");
                let (a_val, b_val) =
                    webgpu_binary_op_values(output_dtype, a_dtype, b_dtype, &a_idx, &b_idx);
                let out_expr = ($expr_builder)(&a_val, &b_val, output_dtype);
                let out_val = webgpu_numeric_write(output_dtype, &out_expr);
                let prelude = shader_prelude();
                let source = format!(
                    r#"
{prelude}
@group(0) @binding(0) var<storage, read> a: array<{a_ty}>;
@group(0) @binding(1) var<storage, read> b: array<{b_ty}>;
@group(0) @binding(2) var<storage, read_write> out: array<{out_ty}>;
@group(0) @binding(3) var<storage, read> params: Params;

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = i32(global_id.x) + params.offset;
    if (idx < params.n) {{
        out[{out_index}] = {out_val};
    }}
}}
"#,
                    out_index = buffer_index(&out_idx),
                    workgroup_size = WORKGROUP_SIZE,
                );
                Some(compile_shader(device, &source, "main"))
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
                context: &mut WebGpuEncodeContext<'_>,
                pipeline: &WebGpuPipeline,
                inputs: &[&WebGpuBuffer],
                output: &WebGpuBuffer,
                _input_dtypes: &[DType],
                _output_dtype: DType,
            ) {
                let n_elements = self.output_size().exec(context.dyn_map).unwrap() as u32;
                dispatch_1d(
                    context,
                    pipeline.get(0),
                    &[inputs[0], inputs[1], output],
                    n_elements,
                );
            }

            impl_binary_metrics!(self, dyn_map, $flops);
        }
    };
}

#[derive(Debug, Default, Clone)]
pub struct WebGpuAdd {
    shape: Vec<Expression>,
    a_strides: Vec<Expression>,
    b_strides: Vec<Expression>,
    output_strides: Vec<Expression>,
}

impl EgglogOp for WebGpuAdd {
    fn sort(&self) -> SortDef {
        binary_sort("WebGpuAdd")
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (args2, hlir_match2) = new_op_call(&Add::default().sort(), &["inp_a", "inp_b"]);
        let webgpu_op2 = op_term(
            call_sort_from_args(&self.sort(), &args2),
            args2["__inputs"].clone(),
        );

        vec![
            binary_dtype_rewrite(&Add::default().sort(), &self.sort()),
            rule(union(hlir_match2.clone(), webgpu_op2.clone()))
                .subsume(hlir_match2)
                .set(dtype(webgpu_op2), app(&SORTS.f32_dt, vec![]))
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
            LLIROp::new::<dyn WebGpuKernelOp>(Box::new(Self {
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

impl WebGpuKernelOp for WebGpuAdd {
    fn compile(
        &self,
        device: &wgpu::Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<WebGpuPipeline> {
        compile_binary_kernel(
            device,
            &self.shape,
            &self.a_strides,
            &self.b_strides,
            &self.output_strides,
            input_dtypes,
            output_dtype,
            |a, b, _| format!("({a}) + ({b})"),
        )
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
        context: &mut WebGpuEncodeContext<'_>,
        pipeline: &WebGpuPipeline,
        inputs: &[&WebGpuBuffer],
        output: &WebGpuBuffer,
        _input_dtypes: &[DType],
        _output_dtype: DType,
    ) {
        let n_elements = self.output_size().exec(context.dyn_map).unwrap() as u32;
        dispatch_1d(
            context,
            pipeline.get(0),
            &[inputs[0], inputs[1], output],
            n_elements,
        );
    }

    impl_binary_metrics!(self, dyn_map, 1);
}

#[allow(clippy::too_many_arguments)]
fn compile_binary_kernel(
    device: &wgpu::Device,
    shape: &[Expression],
    a_strides: &[Expression],
    b_strides: &[Expression],
    output_strides: &[Expression],
    input_dtypes: &[DType],
    output_dtype: DType,
    expr_builder: impl Fn(&str, &str, DType) -> String,
) -> Option<WebGpuPipeline> {
    let a_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
    let b_dtype = input_dtypes.get(1).copied().unwrap_or(a_dtype);
    let a_ty = webgpu_buffer_type(a_dtype);
    let b_ty = webgpu_buffer_type(b_dtype);
    let out_ty = webgpu_buffer_type(output_dtype);
    let a_idx = lower_expression_for_webgpu(&flatten_strides(shape, a_strides), "idx");
    let b_idx = lower_expression_for_webgpu(&flatten_strides(shape, b_strides), "idx");
    let out_idx = lower_expression_for_webgpu(&flatten_strides(shape, output_strides), "idx");
    let (a_val, b_val) = webgpu_binary_op_values(output_dtype, a_dtype, b_dtype, &a_idx, &b_idx);
    let out_expr = expr_builder(&a_val, &b_val, output_dtype);
    let out_val = webgpu_numeric_write(output_dtype, &out_expr);
    let prelude = shader_prelude();
    let source = format!(
        r#"
{prelude}
@group(0) @binding(0) var<storage, read> a: array<{a_ty}>;
@group(0) @binding(1) var<storage, read> b: array<{b_ty}>;
@group(0) @binding(2) var<storage, read_write> out: array<{out_ty}>;
@group(0) @binding(3) var<storage, read> params: Params;

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = i32(global_id.x) + params.offset;
    if (idx < params.n) {{
        out[{out_index}] = {out_val};
    }}
}}
"#,
        out_index = buffer_index(&out_idx),
        workgroup_size = WORKGROUP_SIZE,
    );
    Some(compile_shader(device, &source, "main"))
}

webgpu_binary_op!(
    WebGpuMul,
    "WebGpuMul",
    Mul,
    |a: &str, b: &str, _dtype: DType| format!("({a}) * ({b})"),
    1
);

webgpu_binary_op!(
    WebGpuMod,
    "WebGpuMod",
    Mod,
    |a: &str, b: &str, dtype: DType| {
        if dtype == DType::Int {
            format!("({a}) % ({b})")
        } else {
            format!("({a}) - ({b}) * floor(({a}) / ({b}))")
        }
    },
    10
);

#[derive(Debug, Default, Clone)]
pub struct WebGpuLessThan {
    shape: Vec<Expression>,
    a_strides: Vec<Expression>,
    b_strides: Vec<Expression>,
    output_strides: Vec<Expression>,
}

impl EgglogOp for WebGpuLessThan {
    fn sort(&self) -> SortDef {
        binary_sort("WebGpuLessThan")
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
            LLIROp::new::<dyn WebGpuKernelOp>(Box::new(Self {
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

impl WebGpuKernelOp for WebGpuLessThan {
    fn compile(
        &self,
        device: &wgpu::Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<WebGpuPipeline> {
        compile_binary_kernel(
            device,
            &self.shape,
            &self.a_strides,
            &self.b_strides,
            &self.output_strides,
            input_dtypes,
            output_dtype,
            |a, b, _| format!("select(0.0, 1.0, ({a}) < ({b}))"),
        )
    }

    fn infer_output_dtype(&self, _input_dtypes: &[DType]) -> DType {
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
        context: &mut WebGpuEncodeContext<'_>,
        pipeline: &WebGpuPipeline,
        inputs: &[&WebGpuBuffer],
        output: &WebGpuBuffer,
        _input_dtypes: &[DType],
        _output_dtype: DType,
    ) {
        let n_elements = self.output_size().exec(context.dyn_map).unwrap() as u32;
        dispatch_1d(
            context,
            pipeline.get(0),
            &[inputs[0], inputs[1], output],
            n_elements,
        );
    }

    impl_binary_metrics!(self, dyn_map, 1);
}

macro_rules! webgpu_reduce_op {
    ($name:ident, $sort_name:expr, $hlir:ty, $identity:expr, $combine:expr, $metrics_flops:expr) => {
        #[derive(Debug, Default, Clone)]
        pub struct $name {
            out_shape: Vec<Expression>,
            iters: Expression,
            in_stride: Vec<Expression>,
            iter_stride: Expression,
            out_stride: Vec<Expression>,
        }

        impl EgglogOp for $name {
            fn sort(&self) -> SortDef {
                reduce_sort($sort_name)
            }

            fn rewrites(&self) -> Vec<Rule> {
                vec![unary_dtype_rewrite(
                    &<$hlir>::default().sort(),
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
                use luminal::egglog_utils::{extract_expr, extract_expr_list};
                (
                    LLIROp::new::<dyn WebGpuKernelOp>(Box::new(Self {
                        out_shape: extract_expr_list(
                            egraph,
                            kind_children[0],
                            list_cache,
                            expr_cache,
                        )
                        .unwrap(),
                        iters: extract_expr(egraph, kind_children[1], expr_cache).unwrap(),
                        in_stride: extract_expr_list(
                            egraph,
                            kind_children[2],
                            list_cache,
                            expr_cache,
                        )
                        .unwrap(),
                        iter_stride: extract_expr(egraph, kind_children[3], expr_cache).unwrap(),
                        out_stride: extract_expr_list(
                            egraph,
                            kind_children[4],
                            list_cache,
                            expr_cache,
                        )
                        .unwrap(),
                    })),
                    input_enodes,
                )
            }
        }

        impl WebGpuKernelOp for $name {
            fn compile(
                &self,
                device: &wgpu::Device,
                input_dtypes: &[DType],
                output_dtype: DType,
            ) -> Option<WebGpuPipeline> {
                let input_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
                let input_ty = webgpu_buffer_type(input_dtype);
                let output_ty = webgpu_buffer_type(output_dtype);
                let in_idx = lower_expression_for_webgpu(
                    &flatten_strides(&self.out_shape, &self.in_stride),
                    "gid",
                );
                let out_idx = lower_expression_for_webgpu(
                    &flatten_strides(&self.out_shape, &self.out_stride),
                    "gid",
                );
                let iters = lower_expression_for_webgpu(&self.iters, "gid");
                let iter_offset = lower_expression_for_webgpu(&self.iter_stride, "i");
                let in_val =
                    webgpu_numeric_read(input_dtype, "input", &format!("in_start + {iter_offset}"));
                let out_val = webgpu_numeric_write(output_dtype, "block_value");
                let prelude = shader_prelude();
                let source = format!(
                    r#"
{prelude}
@group(0) @binding(0) var<storage, read> input: array<{input_ty}>;
@group(0) @binding(1) var<storage, read_write> out: array<{output_ty}>;
@group(0) @binding(2) var<storage, read> params: Params;

var<workgroup> partials: array<f32, {workgroup_size}>;

@compute @workgroup_size({workgroup_size})
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {{
    let gid = i32(workgroup_id.x) + params.offset;
    if (gid >= params.n) {{
        return;
    }}
    let tid = i32(local_id.x);
    let in_start = {in_idx};
    let iters = {iters};

    var value: f32 = {identity};
    var i = tid;
    loop {{
        if (i >= iters) {{
            break;
        }}
        value = {combine_loop};
        i = i + i32({workgroup_size});
    }}

    partials[local_id.x] = value;
    workgroupBarrier();

    var stride = {half_workgroup}u;
    loop {{
        if (stride == 0u) {{
            break;
        }}
        if (local_id.x < stride) {{
            partials[local_id.x] = {combine_reduce};
        }}
        workgroupBarrier();
        stride = stride / 2u;
    }}

    if (local_id.x == 0u) {{
        let block_value = partials[0u];
        out[{out_idx}] = {out_val};
    }}
}}
"#,
                    workgroup_size = WORKGROUP_SIZE,
                    half_workgroup = WORKGROUP_SIZE / 2,
                    identity = $identity,
                    combine_loop = ($combine)("value", &in_val),
                    combine_reduce =
                        ($combine)("partials[local_id.x]", "partials[local_id.x + stride]"),
                    out_idx = buffer_index(&out_idx),
                );
                Some(compile_shader(device, &source, "main"))
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
                context: &mut WebGpuEncodeContext<'_>,
                pipeline: &WebGpuPipeline,
                inputs: &[&WebGpuBuffer],
                output: &WebGpuBuffer,
                _input_dtypes: &[DType],
                _output_dtype: DType,
            ) {
                let n_outputs = self.output_size().exec(context.dyn_map).unwrap() as u32;
                dispatch_workgroups(
                    context,
                    pipeline.get(0),
                    &[inputs[0], output],
                    n_outputs,
                    n_outputs,
                );
            }

            impl_reduce_metrics!(self, dyn_map);
        }
    };
}

webgpu_reduce_op!(
    WebGpuSumReduce,
    "WebGpuSum",
    SumReduce,
    "0.0",
    |a: &str, b: &str| format!("({a}) + ({b})"),
    1
);

webgpu_reduce_op!(
    WebGpuMaxReduce,
    "WebGpuMax",
    MaxReduce,
    "-3.4028234663852886e38",
    |a: &str, b: &str| format!("max({a}, {b})"),
    1
);

#[derive(Debug, Default, Clone)]
pub struct WebGpuMatmul {
    pub m: Expression,
    pub n: Expression,
    pub k: Expression,
    pub lhs_row_stride: Expression,
    pub rhs_row_stride: Expression,
    pub out_row_stride: Expression,
    pub transpose_lhs: bool,
    pub transpose_rhs: bool,
}

impl EgglogOp for WebGpuMatmul {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "WebGpuMatmul",
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
                           lhs_layout: WebGpuMatrixLayout,
                           rhs_layout: WebGpuMatrixLayout,
                           transpose_lhs: i64,
                           transpose_rhs: i64| {
            let m = v("?m");
            let n = v("?n");
            let k = v("?k");
            let lhs = v("?lhs");
            let rhs = v("?rhs");
            let lhs_row_stride = match lhs_layout {
                WebGpuMatrixLayout::RowMajor => mul(k.clone(), z.clone()),
                WebGpuMatrixLayout::TransposedRowMajor => mul(m.clone(), z.clone()),
            };
            let rhs_row_stride = match rhs_layout {
                WebGpuMatrixLayout::RowMajor => mul(n.clone(), z.clone()),
                WebGpuMatrixLayout::TransposedRowMajor => mul(k.clone(), z.clone()),
            };
            let lhs_strides = match lhs_layout {
                WebGpuMatrixLayout::RowMajor => {
                    vec![lhs_row_stride.clone(), zero.clone(), z.clone()]
                }
                WebGpuMatrixLayout::TransposedRowMajor => {
                    vec![z.clone(), zero.clone(), lhs_row_stride.clone()]
                }
            };
            let rhs_strides = match rhs_layout {
                WebGpuMatrixLayout::RowMajor => {
                    vec![zero.clone(), z.clone(), rhs_row_stride.clone()]
                }
                WebGpuMatrixLayout::TransposedRowMajor => {
                    vec![zero.clone(), rhs_row_stride.clone(), z.clone()]
                }
            };
            let out_row_stride = mul(n.clone(), z.clone());
            let mul_output_strides = v("?webgpu_matmul_mul_output_strides");

            let mul_op = op_term(
                WebGpuMul::default().sort().call([
                    (
                        "shape",
                        cons(m.clone(), cons(n.clone(), cons(k.clone(), nil()))),
                    ),
                    ("a_strides", expr_list(lhs_strides)),
                    ("b_strides", expr_list(rhs_strides)),
                    ("out_strides", mul_output_strides),
                ]),
                ilist(vec![lhs.clone(), rhs.clone()]),
            );
            let sum_op = op_term(
                WebGpuSumReduce::default().sort().call([
                    ("shape", cons(m.clone(), cons(n.clone(), nil()))),
                    ("iters", k.clone()),
                    ("strides", v("?webgpu_matmul_sum_in_strides")),
                    ("iter_stride", z.clone()),
                    (
                        "out_strides",
                        cons(out_row_stride.clone(), cons(z.clone(), nil())),
                    ),
                ]),
                ilist(vec![mul_op.clone()]),
            );
            let matmul_op = WebGpuMatmul::default().sort().call([
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

            rule(union(sum_op.clone(), matmul_op.clone()))
                .set(dtype(matmul_op), dt.clone())
                .fact(eq(dt, dtype(sum_op)))
                .ruleset("kernel_lower")
                .name(name)
        };

        vec![
            matmul_rule(
                "webgpu-matmul-row-row",
                WebGpuMatrixLayout::RowMajor,
                WebGpuMatrixLayout::RowMajor,
                0,
                0,
            ),
            matmul_rule(
                "webgpu-matmul-row-transposed-rhs",
                WebGpuMatrixLayout::RowMajor,
                WebGpuMatrixLayout::TransposedRowMajor,
                0,
                1,
            ),
            matmul_rule(
                "webgpu-matmul-transposed-lhs-row",
                WebGpuMatrixLayout::TransposedRowMajor,
                WebGpuMatrixLayout::RowMajor,
                1,
                0,
            ),
            matmul_rule(
                "webgpu-matmul-transposed-lhs-transposed-rhs",
                WebGpuMatrixLayout::TransposedRowMajor,
                WebGpuMatrixLayout::TransposedRowMajor,
                1,
                1,
            ),
            Rule::raw(
                "(rule
                    ((= ?mul (Op (WebGpuMul ?shape ?as ?bs ?os) ?inputs))
                     (= ?sum (Op (WebGpuSum ?sshape ?sk ?ssi ?sks ?sso) (ICons ?mul (INil))))
                     (= ?sum (WebGpuMatmul ?m ?n ?k ?lhs ?lhsrs ?rhs ?rhsrs ?ors ?tl ?tr)))
                    ((delete (Op (WebGpuSum ?sshape ?sk ?ssi ?sks ?sso) (ICons ?mul (INil))))
                     (delete (Op (WebGpuMul ?shape ?as ?bs ?os) ?inputs)))
                    :ruleset cleanup
                    :name \"delete-broadcast-mul-sum-when-webgpu-matmul-exists\"
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
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::egglog_utils::extract_expr;
        let extract_flag = |node: &'a ENodeId| -> bool {
            match egraph.enodes[node].0.as_str() {
                "0" => false,
                "1" => true,
                other => panic!("invalid WebGpuMatmul transpose flag {other}"),
            }
        };

        (
            LLIROp::new::<dyn WebGpuKernelOp>(Box::new(Self {
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

impl WebGpuKernelOp for WebGpuMatmul {
    fn compile(
        &self,
        device: &wgpu::Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<WebGpuPipeline> {
        let lhs_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
        let rhs_dtype = input_dtypes.get(1).copied().unwrap_or(lhs_dtype);
        let lhs_ty = webgpu_buffer_type(lhs_dtype);
        let rhs_ty = webgpu_buffer_type(rhs_dtype);
        let out_ty = webgpu_buffer_type(output_dtype);
        let m = lower_scalar_expression_for_webgpu(&self.m);
        let n = lower_scalar_expression_for_webgpu(&self.n);
        let k = lower_scalar_expression_for_webgpu(&self.k);
        let lhs_row_stride = lower_scalar_expression_for_webgpu(&self.lhs_row_stride);
        let rhs_row_stride = lower_scalar_expression_for_webgpu(&self.rhs_row_stride);
        let out_row_stride = lower_scalar_expression_for_webgpu(&self.out_row_stride);
        let lhs_idx = if self.transpose_lhs {
            format!("lhs_col * ({lhs_row_stride}) + row")
        } else {
            format!("row * ({lhs_row_stride}) + lhs_col")
        };
        let rhs_idx = if self.transpose_rhs {
            format!("col * ({rhs_row_stride}) + rhs_row")
        } else {
            format!("rhs_row * ({rhs_row_stride}) + col")
        };
        let lhs_val = webgpu_numeric_read(lhs_dtype, "lhs", &lhs_idx);
        let rhs_val = webgpu_numeric_read(rhs_dtype, "rhs", &rhs_idx);
        let out_val = webgpu_numeric_write(output_dtype, "sum");
        let prelude = shader_prelude();
        let source = format!(
            r#"
{prelude}
@group(0) @binding(0) var<storage, read> lhs: array<{lhs_ty}>;
@group(0) @binding(1) var<storage, read> rhs: array<{rhs_ty}>;
@group(0) @binding(2) var<storage, read_write> out: array<{out_ty}>;
@group(0) @binding(3) var<storage, read> params: Params;

var<workgroup> lhs_tile: array<array<f32, 16>, 16>;
var<workgroup> rhs_tile: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {{
    let col = i32(global_id.x);
    let row = i32(global_id.y);
    let lx = local_id.x;
    let ly = local_id.y;
    let m = {m};
    let n = {n};
    let k = {k};

    var sum = 0.0;
    var tile_start = 0;
    loop {{
        if (tile_start >= k) {{
            break;
        }}

        let lhs_col = tile_start + i32(lx);
        let rhs_row = tile_start + i32(ly);
        var lhs_load = 0.0;
        var rhs_load = 0.0;
        if (row < m && lhs_col < k) {{
            lhs_load = {lhs_val};
        }}
        if (rhs_row < k && col < n) {{
            rhs_load = {rhs_val};
        }}
        lhs_tile[ly][lx] = lhs_load;
        rhs_tile[ly][lx] = rhs_load;
        workgroupBarrier();

        var inner = 0u;
        loop {{
            if (inner >= 16u) {{
                break;
            }}
            sum = sum + lhs_tile[ly][inner] * rhs_tile[inner][lx];
            inner = inner + 1u;
        }}
        workgroupBarrier();
        tile_start = tile_start + 16;
    }}

    if (row < m && col < n) {{
        out[u32(row * ({out_row_stride}) + col)] = {out_val};
    }}
}}
"#
        );
        Some(compile_shader(device, &source, "main"))
    }

    fn infer_output_dtype(&self, input_dtypes: &[DType]) -> DType {
        input_dtypes.first().copied().unwrap_or(DType::F32)
    }

    fn output_size(&self) -> Expression {
        self.m * self.n
    }

    fn encode_compute(
        &self,
        context: &mut WebGpuEncodeContext<'_>,
        pipeline: &WebGpuPipeline,
        inputs: &[&WebGpuBuffer],
        output: &WebGpuBuffer,
        _input_dtypes: &[DType],
        _output_dtype: DType,
    ) {
        let m = self.m.exec(context.dyn_map).unwrap() as u32;
        let n = self.n.exec(context.dyn_map).unwrap() as u32;
        dispatch_2d(
            context,
            pipeline.get(0),
            &[inputs[0], inputs[1], output],
            m.saturating_mul(n),
            n.div_ceil(16),
            m.div_ceil(16),
        );
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
            WebGpuMul::default().sort().call([
                ("shape", mul_shape.clone()),
                ("a_strides", lhs_strides.clone()),
                ("b_strides", rhs_strides.clone()),
                ("out_strides", mul_output_strides),
            ]),
            ilist(vec![lhs.clone(), rhs.clone()]),
        );
        let sum_op = op_term(
            WebGpuSumReduce::default().sort().call([
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
                .name("generic-matmul-webgpu-mul-sum"),
            Rule::raw(
                "(rule
                    ((= ?mul (Op (WebGpuMul ?shape ?as ?bs ?os) ?inputs))
                     (= ?sum (Op (WebGpuSum ?sshape ?sk ?ssi ?sks ?sso) (ICons ?mul (INil))))
                     (= ?sum (GenericMatmul ?go ?gm ?gk ?gl ?glas ?gr ?grs ?gsis ?gsit ?gos)))
                    ((delete (Op (WebGpuSum ?sshape ?sk ?ssi ?sks ?sso) (ICons ?mul (INil))))
                     (delete (Op (WebGpuMul ?shape ?as ?bs ?os) ?inputs)))
                    :ruleset cleanup
                    :name \"delete-broadcast-mul-sum-when-webgpu-generic-matmul-exists\"
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
            LLIROp::new::<dyn WebGpuKernelOp>(Box::new(Self {
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

impl WebGpuKernelOp for GenericMatmul {
    fn compile(
        &self,
        device: &wgpu::Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<WebGpuPipeline> {
        let lhs_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
        let rhs_dtype = input_dtypes.get(1).copied().unwrap_or(lhs_dtype);
        let lhs_ty = webgpu_buffer_type(lhs_dtype);
        let rhs_ty = webgpu_buffer_type(rhs_dtype);
        let out_ty = webgpu_buffer_type(output_dtype);
        let sum_base_idx = lower_expression_for_webgpu(
            &flatten_strides(&self.out_shape, &self.sum_input_strides),
            "gid",
        );
        let iter_offset = lower_expression_for_webgpu(&self.sum_iter_stride, "i");
        let lhs_idx = lower_expression_for_webgpu(
            &flatten_strides(&self.mul_shape, &self.lhs_strides),
            "mul_idx",
        );
        let rhs_idx = lower_expression_for_webgpu(
            &flatten_strides(&self.mul_shape, &self.rhs_strides),
            "mul_idx",
        );
        let out_idx = lower_expression_for_webgpu(
            &flatten_strides(&self.out_shape, &self.out_strides),
            "gid",
        );
        let iters = lower_expression_for_webgpu(&self.k, "gid");
        let lhs_val = webgpu_numeric_read(lhs_dtype, "lhs", &lhs_idx);
        let rhs_val = webgpu_numeric_read(rhs_dtype, "rhs", &rhs_idx);
        let out_val = webgpu_numeric_write(output_dtype, "block_sum");
        let prelude = shader_prelude();
        let source = format!(
            r#"
{prelude}
@group(0) @binding(0) var<storage, read> lhs: array<{lhs_ty}>;
@group(0) @binding(1) var<storage, read> rhs: array<{rhs_ty}>;
@group(0) @binding(2) var<storage, read_write> out: array<{out_ty}>;
@group(0) @binding(3) var<storage, read> params: Params;

var<workgroup> partials: array<f32, {workgroup_size}>;

@compute @workgroup_size({workgroup_size})
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {{
    let gid = i32(workgroup_id.x) + params.offset;
    if (gid >= params.n) {{
        return;
    }}
    let tid = i32(local_id.x);
    let base_idx = {sum_base_idx};
    let iters = {iters};

    var sum = 0.0;
    var i = tid;
    loop {{
        if (i >= iters) {{
            break;
        }}
        let mul_idx = base_idx + {iter_offset};
        sum = sum + ({lhs_val}) * ({rhs_val});
        i = i + i32({workgroup_size});
    }}

    partials[local_id.x] = sum;
    workgroupBarrier();

    var stride = {half_workgroup}u;
    loop {{
        if (stride == 0u) {{
            break;
        }}
        if (local_id.x < stride) {{
            partials[local_id.x] = partials[local_id.x] + partials[local_id.x + stride];
        }}
        workgroupBarrier();
        stride = stride / 2u;
    }}

    if (local_id.x == 0u) {{
        let block_sum = partials[0u];
        out[{out_idx}] = {out_val};
    }}
}}
"#,
            workgroup_size = WORKGROUP_SIZE,
            half_workgroup = WORKGROUP_SIZE / 2,
            out_idx = buffer_index(&out_idx),
        );
        Some(compile_shader(device, &source, "main"))
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
        context: &mut WebGpuEncodeContext<'_>,
        pipeline: &WebGpuPipeline,
        inputs: &[&WebGpuBuffer],
        output: &WebGpuBuffer,
        _input_dtypes: &[DType],
        _output_dtype: DType,
    ) {
        let n_outputs = self.output_size().exec(context.dyn_map).unwrap() as u32;
        dispatch_workgroups(
            context,
            pipeline.get(0),
            &[inputs[0], inputs[1], output],
            n_outputs,
            n_outputs,
        );
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

#[derive(Debug, Default, Clone)]
pub struct WebGpuConstant {
    value: f32,
}

impl EgglogOp for WebGpuConstant {
    fn sort(&self) -> SortDef {
        sort(IR, "WebGpuConstant", &[("value", F64)])
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (args, const_match) = new_op_call(&Constant::default().sort(), &[]);
        let webgpu_op = call_sort_from_args(&self.sort(), &args);
        vec![
            rule(union(const_match.clone(), webgpu_op.clone()))
                .subsume(const_match)
                .set(dtype(webgpu_op), app(&SORTS.f32_dt, vec![]))
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
            LLIROp::new::<dyn WebGpuKernelOp>(Box::new(Self {
                value: egraph.enodes[children[0]]
                    .0
                    .replace('"', "")
                    .parse::<f32>()
                    .unwrap(),
            })),
            vec![],
        )
    }
}

impl WebGpuKernelOp for WebGpuConstant {
    fn compile(
        &self,
        device: &wgpu::Device,
        _input_dtypes: &[DType],
        _output_dtype: DType,
    ) -> Option<WebGpuPipeline> {
        let value = if self.value.fract() == 0.0 {
            format!("{:.1}", self.value)
        } else {
            self.value.to_string()
        };
        let prelude = shader_prelude();
        let source = format!(
            r#"
{prelude}
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    if (global_id.x == 0u && params.n > 0) {{
        out[0u] = {value};
    }}
}}
"#
        );
        Some(compile_shader(device, &source, "main"))
    }

    fn output_size(&self) -> Expression {
        Expression::from(1)
    }

    fn infer_output_dtype(&self, _input_dtypes: &[DType]) -> DType {
        DType::F32
    }

    fn encode_compute(
        &self,
        context: &mut WebGpuEncodeContext<'_>,
        pipeline: &WebGpuPipeline,
        _inputs: &[&WebGpuBuffer],
        output: &WebGpuBuffer,
        _input_dtypes: &[DType],
        _output_dtype: DType,
    ) {
        dispatch_workgroups(context, pipeline.get(0), &[output], 1, 1);
    }
}

#[derive(Debug, Default, Clone)]
pub struct WebGpuIota {
    expr: Expression,
    range: Expression,
}

impl EgglogOp for WebGpuIota {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "WebGpuIota",
            &[("expr", EXPRESSION), ("range", EXPRESSION)],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (args, iota_match) = new_op_call(&Iota::default().sort(), &[]);
        let webgpu_op = call_sort_from_args(&self.sort(), &args);
        vec![
            rule(union(iota_match.clone(), webgpu_op.clone()))
                .subsume(iota_match)
                .set(dtype(webgpu_op), app(&SORTS.int_dt, vec![]))
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
            LLIROp::new::<dyn WebGpuKernelOp>(Box::new(Self {
                expr: extract_expr(egraph, children[0], expr_cache).unwrap(),
                range: extract_expr(egraph, children[1], expr_cache).unwrap(),
            })),
            vec![],
        )
    }
}

impl WebGpuKernelOp for WebGpuIota {
    fn compile(
        &self,
        device: &wgpu::Device,
        _input_dtypes: &[DType],
        _output_dtype: DType,
    ) -> Option<WebGpuPipeline> {
        let expr_code = lower_expression_for_webgpu(&self.expr, "idx");
        let prelude = shader_prelude();
        let source = format!(
            r#"
{prelude}
@group(0) @binding(0) var<storage, read_write> out: array<i32>;
@group(0) @binding(1) var<storage, read> params: Params;

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = i32(global_id.x) + params.offset;
    if (idx < params.n) {{
        out[u32(idx)] = {expr_code};
    }}
}}
"#,
            workgroup_size = WORKGROUP_SIZE,
        );
        Some(compile_shader(device, &source, "main"))
    }

    fn output_size(&self) -> Expression {
        self.range
    }

    fn infer_output_dtype(&self, _input_dtypes: &[DType]) -> DType {
        DType::Int
    }

    fn encode_compute(
        &self,
        context: &mut WebGpuEncodeContext<'_>,
        pipeline: &WebGpuPipeline,
        _inputs: &[&WebGpuBuffer],
        output: &WebGpuBuffer,
        _input_dtypes: &[DType],
        _output_dtype: DType,
    ) {
        let n_elements = self.range.exec(context.dyn_map).unwrap() as u32;
        dispatch_1d(context, pipeline.get(0), &[output], n_elements);
    }
}

#[derive(Debug, Default, Clone)]
pub struct WebGpuGather {
    out_shape: Vec<Expression>,
    index_stride: Vec<Expression>,
    data_shape: Vec<Expression>,
    data_stride: Vec<Expression>,
    out_stride: Vec<Expression>,
}

impl EgglogOp for WebGpuGather {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "WebGpuGather",
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
        let webgpu_args = [
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
        let webgpu_op = self.sort().call(webgpu_args);
        vec![
            rule(union(gather_match.clone(), webgpu_op.clone()))
                .subsume(gather_match)
                .set(dtype(webgpu_op), dt.clone())
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
            LLIROp::new::<dyn WebGpuKernelOp>(Box::new(Self {
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

impl WebGpuKernelOp for WebGpuGather {
    fn compile(
        &self,
        device: &wgpu::Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<WebGpuPipeline> {
        let data_dtype = input_dtypes.get(1).copied().unwrap_or(DType::F32);
        let data_ty = webgpu_buffer_type(data_dtype);
        let out_ty = webgpu_buffer_type(output_dtype);
        let out_idx =
            lower_expression_for_webgpu(&flatten_strides(&self.out_shape, &self.out_stride), "idx");
        let index_idx = lower_expression_for_webgpu(
            &flatten_strides(&self.out_shape, &self.index_stride),
            "idx",
        );
        let data_idx = lower_expression_for_webgpu(
            &flatten_strides(&self.data_shape, &self.data_stride),
            "gathered_index",
        );
        let gathered_val = webgpu_copy_value(data_dtype, "data", &data_idx);
        let prelude = shader_prelude();
        let source = format!(
            r#"
{prelude}
@group(0) @binding(0) var<storage, read> indexes: array<i32>;
@group(0) @binding(1) var<storage, read> data: array<{data_ty}>;
@group(0) @binding(2) var<storage, read_write> out: array<{out_ty}>;
@group(0) @binding(3) var<storage, read> params: Params;

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = i32(global_id.x) + params.offset;
    if (idx < params.n) {{
        let gathered_index = indexes[{index_idx}];
        out[{out_idx}] = {gathered_val};
    }}
}}
"#,
            index_idx = buffer_index(&index_idx),
            out_idx = buffer_index(&out_idx),
            workgroup_size = WORKGROUP_SIZE,
        );
        Some(compile_shader(device, &source, "main"))
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
        context: &mut WebGpuEncodeContext<'_>,
        pipeline: &WebGpuPipeline,
        inputs: &[&WebGpuBuffer],
        output: &WebGpuBuffer,
        _input_dtypes: &[DType],
        _output_dtype: DType,
    ) {
        let n_elements = self.output_size().exec(context.dyn_map).unwrap() as u32;
        dispatch_1d(
            context,
            pipeline.get(0),
            &[inputs[0], inputs[1], output],
            n_elements,
        );
    }

    fn bytes_loaded(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let n = self.output_size().exec(dyn_map).unwrap_or(0);
        n * std::mem::size_of::<i32>() + n * std::mem::size_of::<f32>()
    }

    fn bytes_stored(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        self.output_size().exec(dyn_map).unwrap_or(0) * std::mem::size_of::<f32>()
    }
}

#[derive(Debug, Default, Clone)]
pub struct WebGpuScatter {
    dest_shape: Vec<Expression>,
    dest_strides: Vec<Expression>,
    index_shape: Vec<Expression>,
    index_strides: Vec<Expression>,
    src_strides: Vec<Expression>,
    out_strides: Vec<Expression>,
}

impl EgglogOp for WebGpuScatter {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "WebGpuScatter",
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
        let webgpu_args = [
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
        let webgpu_op = self.sort().call(webgpu_args);
        vec![
            rule(union(scatter_match.clone(), webgpu_op.clone()))
                .subsume(scatter_match)
                .set(dtype(webgpu_op), dt.clone())
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
            LLIROp::new::<dyn WebGpuKernelOp>(Box::new(Self {
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
            })),
            vec![children[2], children[3], children[6]],
        )
    }
}

impl WebGpuKernelOp for WebGpuScatter {
    fn compile(
        &self,
        device: &wgpu::Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<WebGpuPipeline> {
        let dest_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
        let src_dtype = input_dtypes.get(2).copied().unwrap_or(output_dtype);
        let dest_ty = webgpu_buffer_type(dest_dtype);
        let src_ty = webgpu_buffer_type(src_dtype);
        let out_ty = webgpu_buffer_type(output_dtype);
        let dest_idx = lower_expression_for_webgpu(
            &flatten_strides(&self.dest_shape, &self.dest_strides),
            "idx",
        );
        let out_copy_idx = lower_expression_for_webgpu(
            &flatten_strides(&self.dest_shape, &self.out_strides),
            "idx",
        );
        let index_idx = lower_expression_for_webgpu(
            &flatten_strides(&self.index_shape, &self.index_strides),
            "idx",
        );
        let src_idx = lower_expression_for_webgpu(
            &flatten_strides(&self.index_shape, &self.src_strides),
            "idx",
        );
        let prelude = shader_prelude();
        let copy_source = format!(
            r#"
{prelude}
@group(0) @binding(0) var<storage, read_write> out: array<{out_ty}>;
@group(0) @binding(1) var<storage, read> dest: array<{dest_ty}>;
@group(0) @binding(2) var<storage, read> params: Params;

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = i32(global_id.x) + params.offset;
    if (idx < params.n) {{
        out[{out_copy_idx}] = dest[{dest_idx}];
    }}
}}
"#,
            out_copy_idx = buffer_index(&out_copy_idx),
            dest_idx = buffer_index(&dest_idx),
            workgroup_size = WORKGROUP_SIZE,
        );
        let prelude = shader_prelude();
        let scatter_source = format!(
            r#"
{prelude}
@group(0) @binding(0) var<storage, read_write> out: array<{out_ty}>;
@group(0) @binding(1) var<storage, read> indexes: array<i32>;
@group(0) @binding(2) var<storage, read> src: array<{src_ty}>;
@group(0) @binding(3) var<storage, read> params: Params;

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = i32(global_id.x) + params.offset;
    if (idx < params.n) {{
        let scatter_idx = indexes[{index_idx}];
        out[u32(scatter_idx)] = src[{src_idx}];
    }}
}}
"#,
            index_idx = buffer_index(&index_idx),
            src_idx = buffer_index(&src_idx),
            workgroup_size = WORKGROUP_SIZE,
        );
        Some(WebGpuPipeline::new(vec![
            compile_shader_raw(device, &copy_source, "main"),
            compile_shader_raw(device, &scatter_source, "main"),
        ]))
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
        context: &mut WebGpuEncodeContext<'_>,
        pipeline: &WebGpuPipeline,
        inputs: &[&WebGpuBuffer],
        output: &WebGpuBuffer,
        _input_dtypes: &[DType],
        _output_dtype: DType,
    ) {
        let n_dest = self
            .dest_shape
            .iter()
            .cloned()
            .product::<Expression>()
            .exec(context.dyn_map)
            .unwrap() as u32;
        let n_src = self
            .index_shape
            .iter()
            .cloned()
            .product::<Expression>()
            .exec(context.dyn_map)
            .unwrap() as u32;
        dispatch_1d(context, pipeline.get(0), &[output, inputs[0]], n_dest);
        dispatch_1d(
            context,
            pipeline.get(1),
            &[output, inputs[1], inputs[2]],
            n_src,
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
        self.output_size().exec(dyn_map).unwrap_or(0) * std::mem::size_of::<f32>()
    }
}

#[derive(Debug, Default, Clone)]
#[allow(dead_code)]
pub struct WebGpuScatterNoCopy {
    dest_shape: Vec<Expression>,
    dest_strides: Vec<Expression>,
    index_shape: Vec<Expression>,
    index_strides: Vec<Expression>,
    src_strides: Vec<Expression>,
    out_strides: Vec<Expression>,
}

impl EgglogOp for WebGpuScatterNoCopy {
    fn sort(&self) -> SortDef {
        sort(
            luminal::egglog_utils::base::OP_KIND,
            "WebGpuScatterNoCopy",
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
            Rule::raw("(relation webgpu_consumed_buffer_ilist_contains (IList IR))"),
            Rule::raw(
                "(rule
                    ((= ?list (ICons ?head ?tail)))
                    ((webgpu_consumed_buffer_ilist_contains ?list ?head))
                    :ruleset cleanup
                    :name \"webgpu-consumed-buffer-ilist-contains-head\"
                )",
            ),
            Rule::raw(
                "(rule
                    ((= ?list (ICons ?head ?tail))
                     (webgpu_consumed_buffer_ilist_contains ?tail ?item))
                    ((webgpu_consumed_buffer_ilist_contains ?list ?item))
                    :ruleset cleanup
                    :name \"webgpu-consumed-buffer-ilist-contains-tail\"
                )",
            ),
            Rule::raw(
                "(rule
                    ((= ?scatter (WebGpuScatter ?ds ?dst ?dest ?indexes ?is ?istr ?src ?ss ?os))
                     (= ?dst ?os)
                     (= ?dt (dtype ?src)))
                    ((let ?consumed (ConsumedBuffer ?dest))
                     (let ?nocopy (Op (WebGpuScatterNoCopy ?ds ?dst ?is ?istr ?ss ?os)
                         (ICons ?consumed (ICons ?indexes (ICons ?src (INil))))))
                     (union ?scatter ?nocopy)
                     (set (dtype ?nocopy) ?dt))
                    :ruleset buffer_reuse
                    :name \"webgpu-scatter-to-scatter-no-copy\"
                )",
            ),
            Rule::raw(
                "(rule
                    ((= ?cb (ConsumedBuffer ?a))
                     (= ?dt (dtype ?a)))
                    ((set (dtype ?cb) ?dt))
                    :ruleset dtype_prop
                    :name \"webgpu-consumed-buffer-dtype\"
                )",
            ),
            Rule::raw(
                "(rule
                    ((= ?cb (ConsumedBuffer ?a))
                     (= ?op1 (Op ?k1 ?ilist1))
                     (webgpu_consumed_buffer_ilist_contains ?ilist1 ?cb)
                     (= ?op2 (Op ?k2 ?ilist2))
                     (!= ?op1 ?op2)
                     (webgpu_consumed_buffer_ilist_contains ?ilist2 ?a))
                    ((delete (ConsumedBuffer ?a)))
                    :ruleset cleanup
                    :name \"webgpu-consumed-buffer-cleanup-shared-op-use\"
                )",
            ),
            Rule::raw(
                "(rule
                    ((= ?cb (ConsumedBuffer ?dest))
                     (= ?scatter (WebGpuScatter ?ds ?dst ?dest ?indexes ?is ?istr ?src ?ss ?os))
                     (= ?nocopy (Op (WebGpuScatterNoCopy ?ds ?dst ?is ?istr ?ss ?os)
                         (ICons ?cb (ICons ?indexes (ICons ?src (INil)))))))
                    ((subsume (WebGpuScatter ?ds ?dst ?dest ?indexes ?is ?istr ?src ?ss ?os)))
                    :ruleset post_cleanup
                    :name \"webgpu-scatter-no-copy-dominates-valid-consumed-buffer\"
                )",
            ),
            Rule::raw(
                "(rule
                    ((= ?cb (ConsumedBuffer ?a)))
                    ((union ?cb ?a)
                     (delete (ConsumedBuffer ?a)))
                    :ruleset base_cleanup
                    :name \"webgpu-consumed-buffer-resolve\"
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
            LLIROp::new::<dyn WebGpuKernelOp>(Box::new(Self {
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

impl WebGpuKernelOp for WebGpuScatterNoCopy {
    fn compile(
        &self,
        device: &wgpu::Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<WebGpuPipeline> {
        let src_dtype = input_dtypes.get(2).copied().unwrap_or(output_dtype);
        let src_ty = webgpu_buffer_type(src_dtype);
        let out_ty = webgpu_buffer_type(output_dtype);
        let index_idx = lower_expression_for_webgpu(
            &flatten_strides(&self.index_shape, &self.index_strides),
            "idx",
        );
        let src_idx = lower_expression_for_webgpu(
            &flatten_strides(&self.index_shape, &self.src_strides),
            "idx",
        );
        let prelude = shader_prelude();
        let source = format!(
            r#"
{prelude}
@group(0) @binding(0) var<storage, read_write> out: array<{out_ty}>;
@group(0) @binding(1) var<storage, read> indexes: array<i32>;
@group(0) @binding(2) var<storage, read> src: array<{src_ty}>;
@group(0) @binding(3) var<storage, read> params: Params;

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = i32(global_id.x) + params.offset;
    if (idx < params.n) {{
        let scatter_idx = indexes[{index_idx}];
        out[u32(scatter_idx)] = src[{src_idx}];
    }}
}}
"#,
            index_idx = buffer_index(&index_idx),
            src_idx = buffer_index(&src_idx),
            workgroup_size = WORKGROUP_SIZE,
        );
        Some(compile_shader(device, &source, "main"))
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
        context: &mut WebGpuEncodeContext<'_>,
        pipeline: &WebGpuPipeline,
        inputs: &[&WebGpuBuffer],
        output: &WebGpuBuffer,
        _input_dtypes: &[DType],
        _output_dtype: DType,
    ) {
        let n_src = self
            .index_shape
            .iter()
            .copied()
            .product::<Expression>()
            .exec(context.dyn_map)
            .unwrap() as u32;
        dispatch_1d(
            context,
            pipeline.get(0),
            &[output, inputs[1], inputs[2]],
            n_src,
        );
    }

    fn output_aliases_input(&self) -> Option<usize> {
        Some(0)
    }
}

#[derive(Debug, Default, Clone)]
pub struct WebGpuCast {
    size: Expression,
    target_dtype: DType,
}

impl EgglogOp for WebGpuCast {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "WebGpuCast",
            &[("inp", IR), ("size", EXPRESSION), ("dtype", DTYPE)],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (args, cast_match) = new_op_call(&Cast::default().sort(), &["inp"]);
        let webgpu_op = call_sort_from_args(&self.sort(), &args);
        vec![
            rule(union(cast_match.clone(), webgpu_op.clone()))
                .subsume(cast_match)
                .set(dtype(webgpu_op), args["dtype"].clone())
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
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::egglog_utils::{extract_dtype, extract_expr};
        (
            LLIROp::new::<dyn WebGpuKernelOp>(Box::new(Self {
                size: extract_expr(egraph, children[1], expr_cache).unwrap(),
                target_dtype: extract_dtype(egraph, children[2]),
            })),
            vec![children[0]],
        )
    }
}

impl WebGpuKernelOp for WebGpuCast {
    fn compile(
        &self,
        device: &wgpu::Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<WebGpuPipeline> {
        let input_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
        let input_ty = webgpu_buffer_type(input_dtype);
        let output_ty = webgpu_buffer_type(output_dtype);
        let cast_expr = match (input_dtype, output_dtype) {
            (DType::F32 | DType::TF32, DType::F32 | DType::TF32) | (DType::Int, DType::Int) => {
                "inp[u32(idx)]".to_string()
            }
            (DType::F32 | DType::TF32, DType::Int) => "i32(inp[u32(idx)])".to_string(),
            (DType::Int, DType::F32 | DType::TF32) => "f32(inp[u32(idx)])".to_string(),
            _ => panic!(
                "WebGpuCast does not support runtime cast from {input_dtype:?} to {output_dtype:?}"
            ),
        };
        let prelude = shader_prelude();
        let source = format!(
            r#"
{prelude}
@group(0) @binding(0) var<storage, read> inp: array<{input_ty}>;
@group(0) @binding(1) var<storage, read_write> out: array<{output_ty}>;
@group(0) @binding(2) var<storage, read> params: Params;

@compute @workgroup_size({workgroup_size})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = i32(global_id.x) + params.offset;
    if (idx < params.n) {{
        out[u32(idx)] = {cast_expr};
    }}
}}
"#,
            workgroup_size = WORKGROUP_SIZE,
        );
        Some(compile_shader(device, &source, "main"))
    }

    fn output_size(&self) -> Expression {
        self.size
    }

    fn infer_output_dtype(&self, _input_dtypes: &[DType]) -> DType {
        self.target_dtype
    }

    fn encode_compute(
        &self,
        context: &mut WebGpuEncodeContext<'_>,
        pipeline: &WebGpuPipeline,
        inputs: &[&WebGpuBuffer],
        output: &WebGpuBuffer,
        _input_dtypes: &[DType],
        _output_dtype: DType,
    ) {
        let n_elements = self.size.exec(context.dyn_map).unwrap_or(0) as u32;
        dispatch_1d(context, pipeline.get(0), &[inputs[0], output], n_elements);
    }

    fn bytes_loaded(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        self.size.exec(dyn_map).unwrap_or(0) * std::mem::size_of::<f32>()
    }

    fn bytes_stored(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        self.size.exec(dyn_map).unwrap_or(0) * std::mem::size_of::<f32>()
    }
}
