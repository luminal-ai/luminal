use std::{f32, usize};

use super::{CpuKernelOp, CpuSumReduceInfo, CpuMulInfo};
use luminal::{
    dtype::DType,
    egglog_utils::{
        SerializedEGraph, api::{Args, Rule, SortDef, Term as EggTerm, app, eq, rule, sort, union, v}, base::{ELIST, EXPRESSION, F64, IR, SORTS, dtype, new_op_call, op_term}
    },
    hlir::{
        Constant, Gather, Iota, MaxReduce, SumReduce, binary_sort, reduce_sort, unary_sort
    },
    op::*,
    prelude::*, shape::flatten_strides,
};

// ---------------------------------------------------------------------------------------------------
// Shared Helper
// ---------------------------------------------------------------------------------------------------
fn call_sort_from_args(sort_def: &SortDef, args: &Args) -> EggTerm {
    let mut filtered = Args::new();
    for field in &sort_def.fields {
        filtered.add(&field.name, args[field.name.as_str()].clone());
    }
    sort_def.call(filtered)
}

fn unary_dtype_rewrite(hlir_sort: &SortDef, cpu_sort: &SortDef) -> Rule {
    let (args, hlir_match) = new_op_call(hlir_sort, &["inp"]);
    let cpu_op = op_term(call_sort_from_args(cpu_sort, &args), args["__inputs"].clone());
    let dt = v("?__dt");
    rule(union(hlir_match, cpu_op.clone()))
        .set(dtype(cpu_op), dt.clone())
        .fact(eq(dt, dtype(args["inp"].clone())))
        .ruleset("kernel_lower")
}

fn binary_dtype_rewrite(hlir_sort: &SortDef, cpu_sort: &SortDef) -> Rule {
    let (args, hlir_match) = new_op_call(hlir_sort, &["inp_a", "inp_b"]);
    let cpu_op = op_term(call_sort_from_args(cpu_sort, &args), args["__inputs"].clone());
    let dt = v("?__dt");
    rule(union(hlir_match, cpu_op.clone()))
        .set(dtype(cpu_op), dt.clone())
        .fact(eq(dt, dtype(args["inp_a"].clone())))
        .ruleset("kernel_lower")
}

fn resolve(expr: &Expression, dyn_map: &FxHashMap<char, usize>) -> usize {
    let mut map = dyn_map.clone();
    map.entry('z').or_insert(1);
    expr.exec(&map).unwrap_or(0)
}

fn eval_at(expr: &Expression, i: usize, dyn_map: &FxHashMap<char, usize>) -> usize {
    let mut map = dyn_map.clone();
    map.insert('z', i);
    expr.exec(&map).unwrap_or(0)
}

// ---------------------------------------------------------------------------------------------------
// Macro: generate a unary element-wise op
// 
// $name - struct name, e.g. CpuExp2
// $op_name - string used in the sort, e.g. "CpuExp2"
// $compute - a closure |x: f32| -> f32
// ---------------------------------------------------------------------------------------------------
macro_rules! cpu_unary_ops {
    ($name:ident, $op_name:expr, $compute:expr) => {
        #[derive(Debug, Default, Clone)]
        pub struct $name {
            shape: Vec<Expression>,
            input_strides: Vec<Expression>,
            output_strides: Vec<Expression>,
        }

        impl EgglogOp for $name {
            fn sort(&self) -> SortDef { unary_sort($op_name) }

            fn rewrites(&self) -> Vec<Rule> {
                let hlir_name = $op_name.strip_prefix("Cpu").unwrap_or($op_name);
                vec![unary_dtype_rewrite(&unary_sort(hlir_name), &self.sort())]
            }

            fn cleanup(&self) -> bool { false }

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
                    LLIROp::new::<dyn CpuKernelOp>(Box::new(Self {
                        shape:          extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap(),
                        input_strides:  extract_expr_list(egraph, kind_children[1], list_cache, expr_cache).unwrap(),
                        output_strides: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache).unwrap(),
                    })),
                    input_enodes,
                )
            }
        }

        impl CpuKernelOp for $name {
            fn output_size(&self) -> Expression {
                self.shape.iter().cloned().product::<Expression>().max(Expression::from(1))
            }

            fn process(
                &self,
                inputs: &[(&[f32], DType)],
                dyn_map: &FxHashMap<char, usize>,
            ) -> Vec<f32> {
                let input = inputs[0].0;
                let n = resolve(&self.output_size(), dyn_map);

                let index_expr = flatten_strides(&self.shape, &self.input_strides);
                let f: fn(f32) -> f32 = $compute;
                (0..n).map(|i| f(input[eval_at(&index_expr, i, dyn_map)])).collect()
            }
        }
    };
}

// ---------------------------------------------------------------------------------------------------
// Macro: generate a binary element-wise op
// ---------------------------------------------------------------------------------------------------
macro_rules! cpu_binary_op {
    ($name: ident, $op_name: expr, $compute: expr) => {
        #[derive(Debug, Default, Clone)]
        pub struct $name {
            shape: Vec<Expression>,
            a_strides: Vec<Expression>,
            b_strides: Vec<Expression>,
            output_strides: Vec<Expression>,
        }

        impl EgglogOp for $name {
            fn sort(&self) -> SortDef { binary_sort($op_name) }

            fn rewrites(&self) -> Vec<Rule> {
                let hlir_name = $op_name.strip_prefix("Cpu").unwrap_or($op_name);
                vec![binary_dtype_rewrite(&binary_sort(hlir_name), &self.sort())]
            }

            fn cleanup(&self) -> bool { false }

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
                    LLIROp::new::<dyn CpuKernelOp>(Box::new(Self {
                        shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap(),
                        a_strides: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache).unwrap(),
                        b_strides: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache).unwrap(),
                        output_strides: extract_expr_list(egraph, kind_children[3], list_cache, expr_cache).unwrap(),
                    })),
                    input_enodes,
                )
            }
        }

        impl CpuKernelOp for $name {
            fn output_size(&self) -> Expression {
                self.shape.iter().cloned().product::<Expression>().max(Expression::from(1))
            }

            fn process(
                &self,
                inputs: &[(&[f32], DType)],
                dyn_map: &FxHashMap<char, usize>,
            ) -> Vec<f32> {
                let a = inputs[0].0;
                let b = inputs[1].0;
                let n = resolve(&self.output_size(), dyn_map);

                let a_expr = flatten_strides(&self.shape, &self.a_strides);
                let b_expr = flatten_strides(&self.shape, &self.b_strides);

                let f: fn(f32, f32) -> f32 = $compute;
                (0..n).map(|i| {
                    f(a[eval_at(&a_expr, i, dyn_map)], b[eval_at(&b_expr, i, dyn_map)])
                }).collect()
            }
        }
    };
}

// ---------------------------------------------------------------------------------------------------
// Unary ops (macro instantiations)
// ---------------------------------------------------------------------------------------------------
cpu_unary_ops!(CpuExp2, "CpuExp2", |x: f32| x.exp2());
cpu_unary_ops!(CpuLog2, "CpuLog2", |x: f32| x.log2());
cpu_unary_ops!(CpuSin, "CpuSin", |x: f32| x.sin());
cpu_unary_ops!(CpuSqrt, "CpuSqrt", |x: f32| x.sqrt());
cpu_unary_ops!(CpuRecip, "CpuRecip", |x: f32| 1.0 / x);

// ---------------------------------------------------------------------------------------------------
// Binary ops (macro instantiations)
// ---------------------------------------------------------------------------------------------------
cpu_binary_op!(CpuAdd, "CpuAdd", |a: f32, b: f32| a + b);
cpu_binary_op!(CpuMod, "CpuMod", |a: f32, b: f32| a % b);
cpu_binary_op!(CpuLessThan, "CpuLessThan", |a: f32, b: f32| if a < b { 1.0 } else { 0.0 });


// ---------------------------------------------------------------------------------------------------
// CpuSumReduce
// ---------------------------------------------------------------------------------------------------
#[derive(Debug, Default, Clone)]
pub struct CpuSumReduce {
    out_shape: Vec<Expression>,
    pub iters: Expression,
    in_strides: Vec<Expression>,
    iter_stride: Expression,
    out_stride: Vec<Expression>,
}

impl EgglogOp for CpuSumReduce {
    fn sort(&self) -> SortDef{ reduce_sort("CpuSumReduce") }

    fn rewrites(&self) -> Vec<Rule> {
        let (args, hlir_match) = new_op_call(&SumReduce::default().sort(), &["inp"]);
        let cpu_op = op_term(call_sort_from_args(&self.sort(), &args), args["__inputs"].clone());
        let dt = v("?__dt");
        vec![
            rule(union(hlir_match, cpu_op.clone()))
                .set(dtype(cpu_op), dt.clone())
                .fact(eq(dt, dtype(args["inp"].clone())))
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
        use luminal::egglog_utils::{extract_expr, extract_expr_list};
        (
            LLIROp::new::<dyn CpuKernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap(),
                iters: extract_expr(egraph, kind_children[1], expr_cache).unwrap(),
                in_strides: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache).unwrap(),
                iter_stride: extract_expr(egraph, kind_children[3], expr_cache).unwrap(),
                out_stride: extract_expr_list(egraph, kind_children[4], list_cache, expr_cache).unwrap(),
            })),
            input_enodes,
        )
    }
}

impl CpuKernelOp for CpuSumReduce {
    fn output_size(&self) -> Expression {
        self.out_shape.iter().cloned().product::<Expression>().max(Expression::from(1))
    }

    fn process(&self, inputs: &[(&[f32], DType)], dyn_map: &FxHashMap<char, usize>) -> Vec<f32> {
        let input = inputs[0].0;
        let n_out = resolve(&self.output_size(), dyn_map);
        let iters = resolve(&self.iters, dyn_map);

        let in_strides_expr = flatten_strides(&self.out_shape, &self.in_strides);

        let iter_stride_expr = self.iter_stride.clone();

        let mut out = vec![0.0f32; n_out];
        for gid in 0..n_out {
            let in_start = eval_at(&in_strides_expr, gid, dyn_map);
            let mut acc = 0.0f32;
            for i in 0..iters {
                let step = eval_at(&iter_stride_expr, i, dyn_map);
                acc += input[in_start + step];
            }
            out[gid] = acc;
        }
        out
    }

    fn sum_reduce_info(&self) -> Option<CpuSumReduceInfo> {
        Some(CpuSumReduceInfo {
            shape: self.out_shape.clone(),
            strides: self.out_stride.clone(),
            iters: self.iters,
            iter_stride: self.iter_stride,
        })
    }
}


// ---------------------------------------------------------------------------------------------------
// CpuMaxReduce
// ---------------------------------------------------------------------------------------------------
#[derive(Debug, Default, Clone)]
pub struct CpuMaxReduce {
    out_shape: Vec<Expression>,
    iters: Expression,
    in_strides: Vec<Expression>,
    iter_stride: Expression,
    out_stride: Vec<Expression>,
}

impl EgglogOp for CpuMaxReduce {
    fn sort(&self) -> SortDef { reduce_sort("CpuMaxReduce") }

    fn rewrites(&self) -> Vec<luminal::egglog_utils::api::Rule> {
        let (args, hlir_match) = new_op_call(&MaxReduce::default().sort(), &["inp"]);
        let cpu_op = op_term(call_sort_from_args(&self.sort(), &args), args["__inputs"].clone());
        let dt = v("?__dt");
        vec![
            rule(union(hlir_match, cpu_op.clone()))
                .set(dtype(cpu_op), dt.clone())
                .fact(eq(dt, dtype(args["inp"].clone())))
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
        use luminal::egglog_utils::{extract_expr, extract_expr_list};
        (
            LLIROp::new::<dyn CpuKernelOp>(Box::new(Self {
                out_shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap(),
                iters: extract_expr(egraph, kind_children[1], expr_cache).unwrap(),
                in_strides: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache).unwrap(),
                iter_stride: extract_expr(egraph, kind_children[3], expr_cache).unwrap(),
                out_stride: extract_expr_list(egraph, kind_children[4], list_cache, expr_cache).unwrap(),
            })),
            input_enodes,
        )
    }
}

impl CpuKernelOp for CpuMaxReduce {
    fn output_size(&self) -> Expression {
        self.out_shape.iter().cloned().product::<Expression>().max(Expression::from(1))
    }

    fn process(&self, inputs: &[(&[f32], DType)], dyn_map: &FxHashMap<char, usize>) -> Vec<f32> {
        let input = inputs[0].0;
        let n_out = resolve(&self.output_size(), dyn_map);
        let iters = resolve(&self.iters, dyn_map);

        let in_start_expr = flatten_strides(&self.out_shape, &self.in_strides);
        let iter_stride_expr = self.iter_stride.clone();

        (0..n_out).map(|gid| {
            let in_start = eval_at(&in_start_expr, gid, dyn_map);
            (0..iters).map(|i| {
                let step = eval_at(&iter_stride_expr, i, dyn_map);
                input[in_start + step]
            }).fold(f32::NEG_INFINITY, f32::max)
        }).collect()
    }
}


// ---------------------------------------------------------------------------------------------------
// CpuConstant - fills the out put with single scaler value
// ---------------------------------------------------------------------------------------------------
#[derive(Debug, Default, Clone)]
pub struct CpuConstant {
    value: f32,
}

impl EgglogOp for CpuConstant {
    fn sort(&self) -> SortDef {
        sort(IR, "CpuConstant", &[("value", F64)])
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (args, hlir_match) = new_op_call(&Constant::default().sort(), &[]);
        let cpu_op = call_sort_from_args(&self.sort(), &args);
        vec![
            rule(union(hlir_match, cpu_op.clone()))
                .set(dtype(cpu_op), app(&SORTS.f32_dt, vec![]))
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
            _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
            expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
        ) -> (LLIROp, Vec<&'a ENodeId>) {
            let value = egraph.enodes[kind_children[0]].0.replace('"', "").parse::<f32>().unwrap_or(0.0);
            (
                LLIROp::new::<dyn CpuKernelOp>(Box::new(Self { value })),
                vec![],
            )
    }
}

impl CpuKernelOp for CpuConstant {
    fn output_size(&self) -> Expression {
        Expression::from(1)
    }

    fn process(&self, _inputs: &[(&[f32], DType)], _dyn_map: &FxHashMap<char, usize>) -> Vec<f32> {
        vec![self.value]
    }
}


// ---------------------------------------------------------------------------------------------------
// CpuIota
// ---------------------------------------------------------------------------------------------------
#[derive(Debug, Default, Clone)]
pub struct CpuIota {
    expr: Expression,
    range: Expression,
}

impl EgglogOp for CpuIota {
    fn sort(&self) -> SortDef {
        sort(IR, "CpuIota", &[("expr", EXPRESSION), ("range", EXPRESSION)])
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (args, hlir_match) = new_op_call(&Iota::default().sort(), &[]);
        let cpu_op = call_sort_from_args(&self.sort(), &args);
        vec![
            rule(union(hlir_match, cpu_op.clone()))
                .set(dtype(cpu_op), app(&SORTS.int_dt, vec![]))
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
            _input_enodes: Vec<&'a ENodeId>,
            _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
            expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
        ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::egglog_utils::extract_expr;
        (
            LLIROp::new::<dyn CpuKernelOp>(Box::new(Self {
                expr: extract_expr(egraph, kind_children[0], expr_cache).unwrap(),
                range: extract_expr(egraph, kind_children[1], expr_cache).unwrap(),
            })),
            vec![],
        )
    }
}

impl CpuKernelOp for CpuIota {
    fn output_size(&self) -> Expression {
        self.range.clone()
    }

    fn infer_output_dtype(&self, _: &[DType]) -> DType {
        DType::Int
    }

    fn process(&self, _inputs: &[(&[f32], DType)], dyn_map: &FxHashMap<char, usize>) -> Vec<f32> {
        let n = resolve(&self.range, dyn_map);
        (0..n).map(|i| eval_at(&self.expr, i, dyn_map) as f32).collect()
    }
}


// ---------------------------------------------------------------------------------------------------
// CpuGather
// ---------------------------------------------------------------------------------------------------
#[derive(Debug, Default, Clone)]
pub struct CpuGather {
    out_shape: Vec<Expression>,
    index_stride: Vec<Expression>,
    data_stride: Vec<Expression>,
    out_stride: Vec<Expression>
}

impl EgglogOp for CpuGather {
    fn sort(&self) -> SortDef {
        sort(IR, "CpuGather", &[
            ("out_shape", ELIST),
            ("indexes", IR),
            ("index_strides", ELIST),
            ("data", IR),
            ("data_strides", ELIST),
            ("out_strides", ELIST),
        ])
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (gather_args, gather_match) = new_op_call(&Gather::default().sort(), &["indexes", "data"]);

        let out_strides = SORTS.row_major.call([("list".to_string(), gather_args["index_shape"].clone())]);
        let dt = v("?__dt");

        let cpu_args = [
            ("out_shape".to_string(), gather_args["index_shape"].clone()),
            ("indexes".to_string(),       gather_args["indexes"].clone()),
            ("index_strides".to_string(), gather_args["index_strides"].clone()),
            ("data".to_string(),          gather_args["data"].clone()),
            ("data_strides".to_string(),  gather_args["data_strides"].clone()),
            ("out_strides".to_string(),   out_strides),
        ];
        let cpu_op = self.sort().call(cpu_args);
        vec![
            rule(union(gather_match, cpu_op.clone()))
                .set(dtype(cpu_op), dt.clone())
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
            kind_children: &[&'a ENodeId],
            _input_enodes: Vec<&'a ENodeId>,
            list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
            expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
        ) -> (LLIROp, Vec<&'a ENodeId>) {
        use luminal::egglog_utils::extract_expr_list;
        (
            LLIROp::new::<dyn CpuKernelOp>(Box::new(Self {
                out_shape:    extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap(),
                index_stride: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache).unwrap(),
                data_stride:  extract_expr_list(egraph, kind_children[4], list_cache, expr_cache).unwrap(),
                out_stride:   extract_expr_list(egraph, kind_children[5], list_cache, expr_cache).unwrap(),
            })),
            vec![kind_children[1], kind_children[3]],
        )
    }
}

impl CpuKernelOp for CpuGather {
    fn output_size(&self) -> Expression {
        let n_queries = self.out_shape.iter().cloned().product::<Expression>().max(Expression::from(1));
        let dim_expr = self.data_stride.first().cloned().unwrap_or(Expression::from(1));
        n_queries * dim_expr
    }

    fn process(&self, inputs: &[(&[f32], DType)], dyn_map: &FxHashMap<char, usize>) -> Vec<f32> {
        let indexes = inputs[0].0;
        let data = inputs[1].0;
        let n_queries = resolve(self.out_shape.first().unwrap_or(&Expression::from(1)), dyn_map);

        let dim = self.data_stride.first().map(|e| eval_at(e, 1, dyn_map)).unwrap_or(1);

        let index_expr = flatten_strides(&self.out_shape, &self.index_stride);

        let mut out = vec![0.0f32; n_queries * dim];
        for q in 0..n_queries {
            let index_ops = eval_at(&index_expr, q, dyn_map);
            let gathered_row = indexes[index_ops] as usize;
            let data_base = self.data_stride.first().map(|e| eval_at(e, gathered_row, dyn_map)).unwrap_or(gathered_row * dim);

            for col in 0..dim {
                out[q * dim + col] = data[data_base + col];
            }
        }
        out
    }
}


// ---------------------------------------------------------------------------------------------------
// CpuMul
// ---------------------------------------------------------------------------------------------------
#[derive(Debug, Default, Clone)]
pub struct CpuMul {
    pub shape: Vec<Expression>,
    pub a_strides: Vec<Expression>,
    pub b_strides: Vec<Expression>,
    pub output_strides: Vec<Expression>,
}

impl EgglogOp for CpuMul {
    fn sort(&self) -> SortDef {
        binary_sort("CpuMul")
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![binary_dtype_rewrite(&binary_sort("Mul"), &self.sort())]
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
            LLIROp::new::<dyn  CpuKernelOp>(Box::new(Self {
                shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap(),
                a_strides: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache).unwrap(),
                b_strides: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache).unwrap(),
                output_strides: extract_expr_list(egraph, kind_children[3], list_cache, expr_cache).unwrap(),
            })),
            input_enodes,
        )
    }
}

impl CpuKernelOp for CpuMul {
    // Re-declare a required methods do Rust doesn't complain about a partial override

    fn output_size(&self) -> Expression {
        self.shape.iter().cloned().product::<Expression>().max(Expression::from(1))
    }

    fn process(&self, inputs: &[(&[f32], DType)], dyn_map: &FxHashMap<char, usize>) -> Vec<f32> {
        let a = inputs[0].0;
        let b = inputs[1].0;
        let n = resolve(&self.output_size(), dyn_map);
        
        let a_expr = flatten_strides(&self.shape, &self.a_strides);
        let b_expr = flatten_strides(&self.shape, &self.b_strides);

        (0..n).map(|i| {
            a[eval_at(&a_expr, i, dyn_map)] * b[eval_at(&b_expr, i, dyn_map)]
        }).collect()
    }

    fn mul_info(&self) -> Option<CpuMulInfo> {
        Some(CpuMulInfo {
            shape: self.shape.clone(),
            a_strides: self.a_strides.clone(),
            b_strides: self.b_strides.clone(),
            output_strides: self.output_strides.clone(),
        })
    }
}


