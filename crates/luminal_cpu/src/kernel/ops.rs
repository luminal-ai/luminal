use std::{f32, usize};

use super::{CpuKernelOp, CpuSumReduceInfo};
use luminal::{
    dtype::DType,
    egglog_utils::{
        SerializedEGraph, api::{Args, Rule, SortDef, Term as EggTerm, eq, rule, sort, union, v}, base::{ELIST, EXPRESSION, F64, IR, dtype, new_op_call, op_term}
    },
    hlir::{
        Constant, Gather, Iota, MaxReduce, SumReduce, binary_sort, reduce_sort, unary_sort
    },
    op::*,
    prelude::*,
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
    rule(union(hlir_match, cpu_op.clone())).set(dtype(cpu_op), dt.clone()).fact(eq(dt, dtype(args["inp"].clone())))
}

fn binary_dtype_rewrite(hlir_sort: &SortDef, cpu_sort: &SortDef) -> Rule {
    let (args, hlir_match) = new_op_call(hlir_sort, &["inp_a", "inp_b"]);
    let cpu_op = op_term(call_sort_from_args(cpu_sort, &args), args["__inputs"].clone());
    let dt = v("?__dt");
    rule(union(hlir_match, cpu_op.clone())).set(dtype(cpu_op), dt.clone()).fact(eq(dt, dtype(args["inp_a"].clone())))
}

fn resolve(expr: &Expression, dyn_map: &FxHashMap<char, usize>) -> usize {
    let mut map = dyn_map.clone();
    map.entry('z').or_insert(1);
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
                let f: fn(f32) -> f32 = $compute;
                (0..n).map(|i| f(input[i])).collect()
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
                let f: fn(f32, f32) -> f32 = $compute;
                (0..n).map(|i| f(a[i], b[i])).collect()
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
cpu_binary_op!(CpuMul, "CpuMul", |a: f32, b: f32| a * b);
cpu_binary_op!(CpuMod, "CpuMod", |a: f32, b: f32| a % b);
cpu_binary_op!(CpuLessThan, "CpuLessThan", |a: f32, b: f32| if a < b { 1.0 } else { 0.0 });

// ---------------------------------------------------------------------------------------------------
// CpuSumReduce
// ---------------------------------------------------------------------------------------------------
#[derive(Debug, Default, Clone)]
pub struct CpuSumReduce {
    shape: Vec<Expression>,
    strides: Vec<Expression>,
    pub iters: Expression,
    iter_strides: Expression,
}

impl EgglogOp for CpuSumReduce {
    fn sort(&self) -> SortDef{ reduce_sort("CpuSumReduce") }

    fn rewrites(&self) -> Vec<Rule> {
        let (args, hlir_match) = new_op_call(&SumReduce::default().sort(), &["inp"]);
        let cpu_op = op_term(call_sort_from_args(&self.sort(), &args), args["__inputs"].clone());
        let dt = v("?__dt");
        vec![rule(union(hlir_match, cpu_op.clone())).set(dtype(cpu_op), dt.clone()).fact(eq(dt, dtype(args["inp"].clone())))]
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
                shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap(),
                strides: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache).unwrap(),
                iters: extract_expr(egraph, kind_children[2], expr_cache).unwrap(),
                iter_strides: extract_expr(egraph, kind_children[3], expr_cache).unwrap(),
            })),
            input_enodes,
        )
    }
}

impl CpuKernelOp for CpuSumReduce {
    fn output_size(&self) -> Expression {
        self.shape.iter().cloned().product::<Expression>().max(Expression::from(1))
    }

    fn process(&self, inputs: &[(&[f32], DType)], dyn_map: &FxHashMap<char, usize>) -> Vec<f32> {
        let input = inputs[0].0;
        let n_out = resolve(&self.output_size(), dyn_map);
        let iters = resolve(&self.iters, dyn_map);

        let mut out = vec![0.0f32; n_out];
        for out_idx in 0..n_out {
            let mut acc = 0.0f32;
            for k in 0..iters {
                acc += input[out_idx * iters + k];
            }
            out[out_idx] = acc;
        }
        out
    }

    fn sum_reduce_info(&self) -> Option<CpuSumReduceInfo> {
        Some(CpuSumReduceInfo {
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            iters: self.iters.clone(),
            iter_strides: self.iter_strides.clone(),
        })
    }
}


// ---------------------------------------------------------------------------------------------------
// CpuMaxReduce
// ---------------------------------------------------------------------------------------------------
#[derive(Debug, Default, Clone)]
pub struct CpuMaxReduce {
    shape: Vec<Expression>,
    strides: Vec<Expression>,
    iters: Expression,
    iter_strides: Expression,
}

impl EgglogOp for CpuMaxReduce {
    fn sort(&self) -> SortDef { reduce_sort("CpuMaxReduce") }

    fn rewrites(&self) -> Vec<luminal::egglog_utils::api::Rule> {
        let (args, hlir_match) = new_op_call(&MaxReduce::default().sort(), &["inp"]);
        let cpu_op = op_term(call_sort_from_args(&self.sort(), &args), args["__inputs"].clone());
        let dt = v("?__dt");
        vec![rule(union(hlir_match, cpu_op.clone())).set(dtype(cpu_op), dt.clone()).fact(eq(dt, dtype(args["inp"].clone())))    ]
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
                shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap(),
                strides: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache).unwrap(),
                iters: extract_expr(egraph, kind_children[2], expr_cache).unwrap(),
                iter_strides: extract_expr(egraph, kind_children[3], expr_cache).unwrap()
            })),
            input_enodes,
        )
    }
}

impl CpuKernelOp for CpuMaxReduce {
    fn output_size(&self) -> Expression {
        self.shape.iter().cloned().product::<Expression>().max(Expression::from(1))
    }

    fn process(&self, inputs: &[(&[f32], DType)], dyn_map: &FxHashMap<char, usize>) -> Vec<f32> {
        let input = inputs[0].0;
        let n_out = resolve(&self.output_size(), dyn_map);
        let iters = resolve(&self.iters, dyn_map);

        (0..n_out).map(|out_idx| {
            (0..iters).map(|k| input[out_idx * iters + k]).fold(f32::NEG_INFINITY, f32::max)
        }).collect()
    }
}


// ---------------------------------------------------------------------------------------------------
// CpuConstant - fills the out put with single scaler value
// ---------------------------------------------------------------------------------------------------
#[derive(Debug, Default, Clone)]
pub struct CpuConstant {
    value: f32,
    size: Expression,
}

impl EgglogOp for CpuConstant {
    fn sort(&self) -> SortDef {
        sort(IR, "CpuConstant", &[("value", F64), ("size", EXPRESSION)])
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (args, hlir_match) = new_op_call(&Constant::default().sort(), &[]);
        let cpu_op = call_sort_from_args(&self.sort(), &args);
        vec![rule(union(hlir_match, cpu_op.clone()))]
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
        use luminal::egglog_utils::{extract_expr};
        (
            LLIROp::new::<dyn CpuKernelOp>(Box::new(Self {
                value: egraph.enodes[kind_children[0]]
                .0
                .replace("\"", "")
                .parse::<f32>()
                .unwrap(),
                size: extract_expr(egraph, kind_children[1], expr_cache).unwrap()
            })),
            input_enodes,
        )
    }
}

impl CpuKernelOp for CpuConstant {
    fn output_size(&self) -> Expression {
        self.size.clone()
    }

    fn process(&self, _inputs: &[(&[f32], DType)], dyn_map: &FxHashMap<char, usize>) -> Vec<f32> {
        let n = resolve(&self.size, dyn_map);
        vec![self.value; n]
    }
}


// ---------------------------------------------------------------------------------------------------
// CpuIota
// ---------------------------------------------------------------------------------------------------
#[derive(Debug, Default, Clone)]
pub struct CpuIota {
    size: Expression,
}

impl EgglogOp for CpuIota {
    fn sort(&self) -> SortDef {
        sort(IR, "CpuIota", &[("size", EXPRESSION)])
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (args, hlir_match) = new_op_call(&Iota::default().sort(), &[]);
        let cpu_op = call_sort_from_args(&self.sort(), &args);
        vec![rule(union(hlir_match, cpu_op.clone()))]
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
        use luminal::egglog_utils::extract_expr;
        (
            LLIROp::new::<dyn CpuKernelOp>(Box::new(Self {
                size: extract_expr(egraph, kind_children[0], expr_cache).unwrap()
            })),
            input_enodes,
        )
    }
}

impl CpuKernelOp for CpuIota {
    fn output_size(&self) -> Expression {
        self.size.clone()
    }

    fn process(&self, _inputs: &[(&[f32], DType)], dyn_map: &FxHashMap<char, usize>) -> Vec<f32> {
        let n = resolve(&self.size, dyn_map);
        (0..n).map(|i| i as f32).collect()
    }
}


// ---------------------------------------------------------------------------------------------------
// CpuGather
// ---------------------------------------------------------------------------------------------------
#[derive(Debug, Default, Clone)]
pub struct CpuGather {
    index_shape: Vec<Expression>,
    src_shape: Vec<Expression>,
}

impl EgglogOp for CpuGather {
    fn sort(&self) -> SortDef {
        sort(IR, "CpuGather", &[
            ("inp", IR), ("indexes", IR),
            ("index_shape", ELIST), ("src_shape", ELIST),
        ])
    }

    fn rewrites(&self) -> Vec<Rule> {
        let (args, hlir_match) = new_op_call(&Gather::default().sort(), &["inp", "indexes"]);
        let cpu_op = op_term(call_sort_from_args(&self.sort(), &args), args["__inputs"].clone());
        let dt = v("?__dt");
        vec![rule(union(hlir_match, cpu_op.clone())).set(dtype(cpu_op), dt.clone()).fact(eq(dt, dtype(args["inp"].clone())))]
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
            LLIROp::new::<dyn CpuKernelOp>(Box::new(Self {
                index_shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap(),
                src_shape: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache).unwrap(),
            })),
            input_enodes,
        )
    }
}

impl CpuKernelOp for CpuGather {
    fn output_size(&self) -> Expression {
        self.index_shape.iter().cloned().product::<Expression>().max(Expression::from(1))
    }

    fn process(&self, inputs: &[(&[f32], DType)], dyn_map: &FxHashMap<char, usize>) -> Vec<f32> {
        let src = inputs[0].0;
        let indexes = inputs[1].0;
        let n = resolve(&self.output_size(), dyn_map);
        (0..n).map(|i| src[indexes[i] as usize]).collect()
    }
}


// ---------------------------------------------------------------------------------------------------
// mul_info() override on CpuMul
// ---------------------------------------------------------------------------------------------------
// impl CpuKernelOp for CpuMul {
//     // Re-declare a required methods do Rust doesn't complain about a partial override

//     fn output_size(&self) -> Expression {
//         self.shape.iter().cloned().product::<Expression>().max(Expression::from(1))
//     }

//     fn process(&self, inputs: &[(&[f32], DType)], dyn_map: &FxHashMap<char, usize>) -> Vec<f32> {
//         let a = inputs[0].0;
//         let b = inputs[1].0;
//         let n = resolve(&self.output_size(), dyn_map);
//         (0..n).map(|i| a[i] * b [i]).collect()
//     }

//     fn mul_info(&self) -> Option<CpuMulInfo> {
//         Some(CpuMulInfo {
//             shape: self.shape.clone(),
//             a_strides: self.a_strides.clone(),
//             b_strides: self.b_strides.clone(),
//             output_strides: self.output_strides.clone(),
//         })
//     }
// }


