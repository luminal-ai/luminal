use std::fmt::Display;
use std::{fmt::Debug, sync::Arc};

use crate::egglog_utils::api::{Term, eq, v};
use crate::egglog_utils::{
    api::{Action, Rule, SortDef, sort},
    base::*,
    *,
};
use crate::op::*;
use crate::prelude::*;

use as_any::AsAny;
use itertools::Itertools;

// --- Dtype helpers for direct IR variants (Input, Output) ---

/// Helper: build a dtype propagation rule for a direct IR op.
/// Matches the op, reads dtype from the named IR source field, and sets it on the op.
fn dtype_propagation_rule(sort: &SortDef, dtype_source: &str) -> Rule {
    let (args, op_match) = sort.new_call();
    let e = v("__e");
    let dty = v("__dty");
    Rule::new()
        .fact(eq(e.clone(), op_match))
        .fact(eq(dty.clone(), dtype(args[dtype_source].clone())))
        .action(Action::Set(dtype(e), dty))
        .ruleset("dtype_prop")
}

/// Helper: build a dtype-from-field rule for a direct IR op.
fn dtype_from_field_rule(sort: &SortDef, dtype_field: &str) -> Rule {
    let (args, op_match) = sort.new_call();
    let e = v("__e");
    Rule::new()
        .fact(eq(e.clone(), op_match))
        .action(Action::Set(dtype(e), args[dtype_field].clone()))
        .ruleset("dtype_prop")
}

// --- Dtype helpers for normalized ops (Op OpKind IList) ---

/// Dtype propagation for a normalized op: inherits from first IList input.
fn dtype_propagation_op(kind_sort: &SortDef) -> Rule {
    let (_, kind_term) = kind_sort.new_call();
    let e = v("__e");
    let first_inp = v("__first_inp");
    let tail = v("__tail");
    let dty = v("__dty");
    Rule::new()
        .fact(eq(
            e.clone(),
            op_term(
                kind_term,
                Term::App {
                    variant: "ICons".to_string(),
                    args: vec![first_inp.clone(), tail],
                },
            ),
        ))
        .fact(eq(dty.clone(), dtype(first_inp)))
        .action(Action::Set(dtype(e), dty))
        .ruleset("dtype_prop")
}

/// Dtype from a field on the OpKind (e.g., Cast's dtype field).
fn dtype_from_kind_field(kind_sort: &SortDef, field_name: &str) -> Rule {
    let (args, kind_term) = kind_sort.new_call();
    let e = v("__e");
    let inputs = v("__inputs");
    Rule::new()
        .fact(eq(e.clone(), op_term(kind_term, inputs)))
        .action(Action::Set(dtype(e), args[field_name].clone()))
        .ruleset("dtype_prop")
}

/// Fixed dtype for a normalized op (e.g., Iota always Int).
fn dtype_fixed_op(kind_sort: &SortDef, dtype_sort: &SortDef) -> Rule {
    let (_, kind_term) = kind_sort.new_call();
    let e = v("__e");
    let inputs = v("__inputs");
    Rule::new()
        .fact(eq(e.clone(), op_term(kind_term, inputs)))
        .action(Action::Set(dtype(e), dtype_sort.call(())))
        .ruleset("dtype_prop")
}

/// Build an IList egglog string from input variable names.
fn ilist_egglog(inputs: &[&str]) -> String {
    list_to_egglog(inputs, "ICons", "INil")
}
use num_traits::Float;
use petgraph::{Direction, algo::toposort, prelude::StableGraph, visit::EdgeRef};
use rustc_hash::{FxHashMap, FxHashSet};
use tracing::info_span;

// --- Convenience sort builders for common op patterns ---

/// Unary op kind: (shape: EList, strides: EList, out_strides: EList), IList: [inp]
pub fn unary_sort(name: &str) -> SortDef {
    sort(
        OP_KIND,
        name,
        &[("shape", ELIST), ("strides", ELIST), ("out_strides", ELIST)],
    )
}

/// Binary op kind: (shape: EList, a_strides: EList, b_strides: EList, out_strides: EList), IList: [inp_a, inp_b]
pub fn binary_sort(name: &str) -> SortDef {
    sort(
        OP_KIND,
        name,
        &[
            ("shape", ELIST),
            ("a_strides", ELIST),
            ("b_strides", ELIST),
            ("out_strides", ELIST),
        ],
    )
}

/// Reduce op kind: (shape: EList, iters: Expression, strides: EList, iter_stride: Expression, out_strides: EList), IList: [inp]
pub fn reduce_sort(name: &str) -> SortDef {
    sort(
        OP_KIND,
        name,
        &[
            ("shape", ELIST),
            ("iters", EXPRESSION),
            ("strides", ELIST),
            ("iter_stride", EXPRESSION),
            ("out_strides", ELIST),
        ],
    )
}

pub type HLIROps = (
    Input,
    Output,
    CustomOpKind,
    Constant,
    Cast,
    Iota,
    Exp2,
    Log2,
    Sin,
    Recip,
    Sqrt,
    Add,
    Mul,
    Mod,
    LessThan,
    Gather,
    Scatter,
    SumReduce,
    MaxReduce,
    Softmax,
);

#[derive(Default, Debug, Clone)]
pub struct Input {
    pub node: usize,
    pub label: String,
    pub dtype: DType,
}

impl Display for Input {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Input({}{})",
            if self.label.is_empty() {
                "".to_string()
            } else {
                format!("{}; ", self.label)
            },
            self.dtype
        )
    }
}

impl EgglogOp for Input {
    fn sort(&self) -> SortDef {
        sort(
            IR,
            "Input",
            &[("node", I64), ("label", STRING), ("dtype", DTYPE)],
        )
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![dtype_from_field_rule(&self.sort(), "dtype")]
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        _input_enodes: Vec<&'a ENodeId>,
        _: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        _: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let node = egraph.enodes[kind_children[0]]
            .0
            .replace("\"", "")
            .parse::<usize>()
            .unwrap();
        let label = egraph.enodes[kind_children[1]].0.replace("\"", "");
        (
            LLIROp::new::<Input>(Box::new(Self {
                node,
                label,
                dtype: extract_dtype(egraph, kind_children[2]),
            })),
            vec![],
        )
    }
}

impl HLIROp for Input {
    fn to_egglog(&self, _: &[(NodeIndex, String)]) -> String {
        format!(
            "(Input {} \"{}\" ({:?}))",
            self.node, self.label, self.dtype
        )
    }
}

impl NativeOp for Input {
    fn execute(&self, _: Vec<&NativeData>, _: &FxHashMap<char, usize>) -> NativeData {
        unimplemented!()
    }
}

#[derive(Default, Debug, Clone)]
pub struct Output {
    pub node: usize,
}

impl Display for Output {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Output")
    }
}

impl EgglogOp for Output {
    fn sort(&self) -> SortDef {
        sort(IR, "Output", &[("inp", IR), ("node", I64)])
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![dtype_propagation_rule(&self.sort(), "inp")]
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        _input_enodes: Vec<&'a ENodeId>,
        _: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        _: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<Output>(Box::new(Self {
                node: egraph.enodes[kind_children[1]]
                    .0
                    .replace("\"", "")
                    .parse::<usize>()
                    .unwrap(),
            })),
            vec![kind_children[0]],
        )
    }
}

impl HLIROp for Output {
    fn to_egglog(&self, inp: &[(NodeIndex, String)]) -> String {
        format!("(Output {} {})", inp[0].1, self.node)
    }
}

impl NativeOp for Output {
    fn execute(&self, _: Vec<&NativeData>, _: &FxHashMap<char, usize>) -> NativeData {
        unimplemented!()
    }
}

#[derive(Default, Debug, Clone)]
pub struct CustomOpKind {
    pub id: usize,
    pub dtype: DType,
}

impl Display for CustomOpKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CustomOp({})", self.dtype)
    }
}

impl EgglogOp for CustomOpKind {
    fn sort(&self) -> SortDef {
        sort(OP_KIND, "CustomOpKind", &[("id", I64), ("dtype", DTYPE)])
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![dtype_from_kind_field(&self.sort(), "dtype")]
    }

    fn cleanup(&self) -> bool {
        false
    }
}

impl HLIROp for CustomOpKind {
    fn to_egglog(&self, inp: &[(NodeIndex, String)]) -> String {
        format!(
            "(Op (CustomOpKind {} ({:?})) {})",
            self.id,
            self.dtype,
            list_to_egglog(&inp.iter().map(|i| &i.1).collect_vec(), "ICons", "INil"),
        )
    }
}

impl NativeOp for CustomOpKind {
    fn execute(&self, _: Vec<&NativeData>, _: &FxHashMap<char, usize>) -> NativeData {
        unimplemented!()
    }
}

/// Produces a single number constant from an expression or a float
#[derive(Clone, PartialEq, Default)]
pub struct Constant(pub f32);
impl Debug for Constant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Constant({:?})", self.0)
    }
}

impl Display for Constant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl HLIROp for Constant {
    fn to_egglog(&self, _: &[(NodeIndex, String)]) -> String {
        format!("(Op (Constant {:.6}) (INil))", self.0)
    }
}

impl EgglogOp for Constant {
    fn sort(&self) -> SortDef {
        sort(OP_KIND, "Constant", &[("value", F64)])
    }
    fn cleanup(&self) -> bool {
        true
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![dtype_fixed_op(&self.sort(), &SORTS.f32_dt)]
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        _input_enodes: Vec<&'a ENodeId>,
        _: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        _: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn NativeOp>(Box::new(Self(
                egraph.enodes[kind_children[0]]
                    .0
                    .replace("\"", "")
                    .parse::<f32>()
                    .unwrap(),
            ))),
            vec![],
        )
    }
}

impl NativeOp for Constant {
    fn execute(&self, _: Vec<&NativeData>, _: &FxHashMap<char, usize>) -> NativeData {
        NativeData::F32(vec![self.0])
    }
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct Iota(pub Expression, pub Expression);
impl Display for Iota {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Iota({}; {})", self.0, self.1)
    }
}
impl HLIROp for Iota {
    fn to_egglog(&self, _: &[(NodeIndex, String)]) -> String {
        format!(
            "(Op (Iota {} {}) (INil))",
            self.0.to_egglog(),
            self.1.to_egglog()
        )
    }
}
impl EgglogOp for Iota {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "Iota",
            &[("expr", EXPRESSION), ("range", EXPRESSION)],
        )
    }

    fn cleanup(&self) -> bool {
        true
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![dtype_fixed_op(&self.sort(), &SORTS.int_dt)]
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        _input_enodes: Vec<&'a ENodeId>,
        _: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn NativeOp>(Box::new(Self(
                extract_expr(egraph, kind_children[0], expr_cache).unwrap(),
                extract_expr(egraph, kind_children[1], expr_cache).unwrap(),
            ))),
            vec![],
        )
    }
}
impl NativeOp for Iota {
    fn execute(&self, _: Vec<&NativeData>, dyn_map: &FxHashMap<char, usize>) -> NativeData {
        let length = self.1.exec(dyn_map).unwrap();
        let expr = self.0.resolve_vars(dyn_map);
        NativeData::Int(
            (0..length)
                .map(|i| expr.exec_single_var(i) as i32)
                .collect(),
        )
    }
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct Cast(pub Expression, pub DType);
impl Display for Cast {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cast({})", self.1)
    }
}
impl HLIROp for Cast {
    fn to_egglog(&self, inp: &[(NodeIndex, String)]) -> String {
        format!(
            "(Op (Cast {} ({:?})) (ICons {} (INil)))",
            self.0.to_egglog(),
            self.1,
            inp[0].1,
        )
    }
}
impl EgglogOp for Cast {
    fn sort(&self) -> SortDef {
        sort(OP_KIND, "Cast", &[("size", EXPRESSION), ("dtype", DTYPE)])
    }

    fn cleanup(&self) -> bool {
        true
    }

    fn n_inputs(&self) -> usize {
        1
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![dtype_from_kind_field(&self.sort(), "dtype")]
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        _: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        ec: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn NativeOp>(Box::new(Self(
                extract_expr(egraph, kind_children[0], ec).unwrap(),
                extract_dtype(egraph, kind_children[1]),
            ))),
            input_enodes,
        )
    }
}
impl NativeOp for Cast {
    fn execute(&self, input: Vec<&NativeData>, _: &FxHashMap<char, usize>) -> NativeData {
        match self.1 {
            DType::F32 => NativeData::F32(match &input[0] {
                NativeData::F32(f) => f.clone(),
                NativeData::F16(f) => f.iter().map(|f| f.to_f32()).collect(),
                NativeData::Bf16(f) => f.iter().map(|f| f.to_f32()).collect(),
                NativeData::Int(i) => i.iter().map(|i| *i as f32).collect(),
                NativeData::Bool(b) => b.iter().map(|b| if *b { 1.0 } else { 0.0 }).collect(),
            }),
            DType::Int => NativeData::Int(match &input[0] {
                NativeData::F32(f) => f.iter().map(|f| *f as i32).collect(),
                NativeData::F16(f) => f.iter().map(|f| f.to_f32() as i32).collect(),
                NativeData::Bf16(f) => f.iter().map(|f| f.to_f32() as i32).collect(),
                NativeData::Int(i) => i.clone(),
                NativeData::Bool(b) => b.iter().map(|b| if *b { 1 } else { 0 }).collect(),
            }),
            DType::F16 => NativeData::F16(match &input[0] {
                NativeData::F32(f) => f.iter().copied().map(f16::from_f32).collect(),
                NativeData::F16(f) => f.clone(),
                NativeData::Bf16(f) => f.iter().map(|f| f16::from_f32(f.to_f32())).collect(),
                NativeData::Int(i) => i.iter().map(|i| f16::from_f32(*i as f32)).collect(),
                NativeData::Bool(b) => b
                    .iter()
                    .map(|b| f16::from_f32(if *b { 1.0 } else { 0.0 }))
                    .collect(),
            }),
            DType::Bf16 => NativeData::Bf16(match &input[0] {
                NativeData::F32(f) => f.iter().copied().map(bf16::from_f32).collect(),
                NativeData::F16(f) => f.iter().map(|f| bf16::from_f32(f.to_f32())).collect(),
                NativeData::Bf16(f) => f.clone(),
                NativeData::Int(i) => i.iter().map(|i| bf16::from_f32(*i as f32)).collect(),
                NativeData::Bool(b) => b
                    .iter()
                    .map(|b| bf16::from_f32(if *b { 1.0 } else { 0.0 }))
                    .collect(),
            }),
            DType::Bool => NativeData::Bool(match &input[0] {
                NativeData::F32(f) => f.iter().map(|f| *f != 0.0).collect(),
                NativeData::F16(f) => f.iter().map(|f| f.to_f32() != 0.0).collect(),
                NativeData::Bf16(f) => f.iter().map(|f| f.to_f32() != 0.0).collect(),
                NativeData::Int(i) => i.iter().map(|i| *i != 0).collect(),
                NativeData::Bool(b) => b.clone(),
            }),
            other => unimplemented!("Cast to {other} is not yet supported in native interpreter"),
        }
    }
}

/// Graph break for chunking search graphs
#[derive(Clone, PartialEq, Default)]
pub struct GraphBreak {
    pub input_shape: ShapeTracker,
}
impl Debug for GraphBreak {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GraphBreak")
    }
}
impl Display for GraphBreak {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GraphBreak")
    }
}

impl HLIROp for GraphBreak {
    fn to_egglog(&self, _: &[(NodeIndex, String)]) -> String {
        panic!("Cannot turn GraphBreak into egglog op!");
    }
}

// Unary Op (A -> A)

fn unary_impl(
    inp: &NativeData,
    shape: &[Expression],
    strides: &[Expression],
    dyn_map: &FxHashMap<char, usize>,
    f32_fn: fn(f32) -> f32,
    f16_fn: fn(f16) -> f16,
    bf16_fn: fn(bf16) -> bf16,
) -> NativeData {
    let ind = StridedIterator::new(shape, strides, dyn_map);
    match &inp {
        NativeData::F32(f) => NativeData::F32(ind.map(|i| f32_fn(f[i])).collect()),
        NativeData::F16(f) => NativeData::F16(ind.map(|i| f16_fn(f[i])).collect()),
        NativeData::Bf16(f) => NativeData::Bf16(ind.map(|i| bf16_fn(f[i])).collect()),
        NativeData::Int(_) => panic!("not implemented for int"),
        NativeData::Bool(_) => panic!("not implemented for bool"),
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Log2 {
    pub shape: Vec<Expression>,
    pub strides: Vec<Expression>,
    pub input_shape: ShapeTracker,
}
impl Display for Log2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Log2")
    }
}
impl HLIROp for Log2 {
    fn to_egglog(&self, inputs: &[(NodeIndex, String)]) -> String {
        format!(
            "(Op (Log2 {} {} {}) {})",
            elist_to_egglog(&self.input_shape.dims),
            elist_to_egglog(&self.input_shape.strides),
            elist_to_egglog(&self.input_shape.contiguous().strides),
            ilist_egglog(&[&inputs[0].1]),
        )
    }
}
impl EgglogOp for Log2 {
    fn sort(&self) -> SortDef {
        unary_sort("Log2")
    }
    fn cleanup(&self) -> bool {
        true
    }
    fn n_inputs(&self) -> usize {
        1
    }
    fn rewrites(&self) -> Vec<Rule> {
        vec![dtype_propagation_op(&self.sort())]
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn NativeOp>(Box::new(Self {
                shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap(),
                strides: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                ..Default::default()
            })),
            input_enodes,
        )
    }
}
impl NativeOp for Log2 {
    fn execute(&self, inputs: Vec<&NativeData>, dyn_map: &FxHashMap<char, usize>) -> NativeData {
        unary_impl(
            inputs[0],
            &self.shape,
            &self.strides,
            dyn_map,
            |f| f.log2(),
            |f| f.log2(),
            |f| f.log2(),
        )
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Exp2 {
    pub shape: Vec<Expression>,
    pub strides: Vec<Expression>,
    pub input_shape: ShapeTracker,
}
impl Display for Exp2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Exp2")
    }
}
impl HLIROp for Exp2 {
    fn to_egglog(&self, inputs: &[(NodeIndex, String)]) -> String {
        format!(
            "(Op (Exp2 {} {} {}) {})",
            elist_to_egglog(&self.input_shape.dims),
            elist_to_egglog(&self.input_shape.strides),
            elist_to_egglog(&self.input_shape.contiguous().strides),
            ilist_egglog(&[&inputs[0].1]),
        )
    }
}
impl EgglogOp for Exp2 {
    fn sort(&self) -> SortDef {
        unary_sort("Exp2")
    }
    fn cleanup(&self) -> bool {
        true
    }
    fn n_inputs(&self) -> usize {
        1
    }
    fn rewrites(&self) -> Vec<Rule> {
        vec![dtype_propagation_op(&self.sort())]
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn NativeOp>(Box::new(Self {
                shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap(),
                strides: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                ..Default::default()
            })),
            input_enodes,
        )
    }
}
impl NativeOp for Exp2 {
    fn execute(&self, inputs: Vec<&NativeData>, dyn_map: &FxHashMap<char, usize>) -> NativeData {
        unary_impl(
            inputs[0],
            &self.shape,
            &self.strides,
            dyn_map,
            |f| f.exp2(),
            |f| f.exp2(),
            |f| f.exp2(),
        )
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sin {
    pub shape: Vec<Expression>,
    pub strides: Vec<Expression>,
    pub input_shape: ShapeTracker,
}
impl Display for Sin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Sin")
    }
}
impl HLIROp for Sin {
    fn to_egglog(&self, inputs: &[(NodeIndex, String)]) -> String {
        format!(
            "(Op (Sin {} {} {}) {})",
            elist_to_egglog(&self.input_shape.dims),
            elist_to_egglog(&self.input_shape.strides),
            elist_to_egglog(&self.input_shape.contiguous().strides),
            ilist_egglog(&[&inputs[0].1]),
        )
    }
}

impl EgglogOp for Sin {
    fn sort(&self) -> SortDef {
        unary_sort("Sin")
    }
    fn cleanup(&self) -> bool {
        true
    }
    fn n_inputs(&self) -> usize {
        1
    }
    fn rewrites(&self) -> Vec<Rule> {
        vec![dtype_propagation_op(&self.sort())]
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn NativeOp>(Box::new(Self {
                shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap(),
                strides: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                ..Default::default()
            })),
            input_enodes,
        )
    }
}
impl NativeOp for Sin {
    fn execute(&self, inputs: Vec<&NativeData>, dyn_map: &FxHashMap<char, usize>) -> NativeData {
        unary_impl(
            inputs[0],
            &self.shape,
            &self.strides,
            dyn_map,
            |f| f.sin(),
            |f| f.sin(),
            |f| f.sin(),
        )
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Recip {
    pub shape: Vec<Expression>,
    pub strides: Vec<Expression>,
    pub input_shape: ShapeTracker,
}
impl Display for Recip {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Recip")
    }
}
impl HLIROp for Recip {
    fn to_egglog(&self, inputs: &[(NodeIndex, String)]) -> String {
        format!(
            "(Op (Recip {} {} {}) {})",
            elist_to_egglog(&self.input_shape.dims),
            elist_to_egglog(&self.input_shape.strides),
            elist_to_egglog(&self.input_shape.contiguous().strides),
            ilist_egglog(&[&inputs[0].1]),
        )
    }
}

impl EgglogOp for Recip {
    fn sort(&self) -> SortDef {
        unary_sort("Recip")
    }
    fn cleanup(&self) -> bool {
        true
    }
    fn n_inputs(&self) -> usize {
        1
    }
    fn rewrites(&self) -> Vec<Rule> {
        vec![dtype_propagation_op(&self.sort())]
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn NativeOp>(Box::new(Self {
                shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap(),
                strides: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                ..Default::default()
            })),
            input_enodes,
        )
    }
}
impl NativeOp for Recip {
    fn execute(&self, inputs: Vec<&NativeData>, dyn_map: &FxHashMap<char, usize>) -> NativeData {
        unary_impl(
            inputs[0],
            &self.shape,
            &self.strides,
            dyn_map,
            |f| f.recip(),
            |f| f.recip(),
            |f| f.recip(),
        )
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sqrt {
    pub shape: Vec<Expression>,
    pub strides: Vec<Expression>,
    pub input_shape: ShapeTracker,
}
impl Display for Sqrt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Sqrt")
    }
}
impl HLIROp for Sqrt {
    fn to_egglog(&self, inputs: &[(NodeIndex, String)]) -> String {
        format!(
            "(Op (Sqrt {} {} {}) {})",
            elist_to_egglog(&self.input_shape.dims),
            elist_to_egglog(&self.input_shape.strides),
            elist_to_egglog(&self.input_shape.contiguous().strides),
            ilist_egglog(&[&inputs[0].1]),
        )
    }
}

impl EgglogOp for Sqrt {
    fn sort(&self) -> SortDef {
        unary_sort("Sqrt")
    }
    fn cleanup(&self) -> bool {
        true
    }
    fn n_inputs(&self) -> usize {
        1
    }
    fn rewrites(&self) -> Vec<Rule> {
        vec![dtype_propagation_op(&self.sort())]
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn NativeOp>(Box::new(Self {
                shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap(),
                strides: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                ..Default::default()
            })),
            input_enodes,
        )
    }
}
impl NativeOp for Sqrt {
    fn execute(&self, inputs: Vec<&NativeData>, dyn_map: &FxHashMap<char, usize>) -> NativeData {
        unary_impl(
            inputs[0],
            &self.shape,
            &self.strides,
            dyn_map,
            |f| f.sqrt(),
            |f| f.sqrt(),
            |f| f.sqrt(),
        )
    }
}

// Binary Ops (A x A -> A)

fn bin_fn<A: Copy>(
    a_ind: StridedIterator,
    a: &[A],
    b_ind: StridedIterator,
    b: &NativeData,
    b_get: impl Fn(&NativeData, usize) -> A,
    op: impl Fn(A, A) -> A,
) -> Vec<A> {
    let a_shape = a_ind.shape.clone();
    let a_strides = a_ind.strides.clone();
    let b_shape = b_ind.shape.clone();
    let b_strides = b_ind.strides.clone();
    a_ind
        .zip(b_ind)
        .map(|(i, j)| {
            assert!(
                i < a.len(),
                "bin_fn: a index {i} out of bounds (a.len={}), shape={a_shape:?}, strides={a_strides:?}",
                a.len(),
            );
            assert!(
                j < b.len(),
                "bin_fn: b index {j} out of bounds (b.len={}), shape={b_shape:?}, strides={b_strides:?}",
                b.len(),
            );
            op(a[i], b_get(b, j))
        })
        .collect()
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Add {
    pub shape: Vec<Expression>,
    pub a_strides: Vec<Expression>,
    pub b_strides: Vec<Expression>,
    pub input_shapes: Vec<ShapeTracker>,
}
impl Display for Add {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Add")
    }
}
impl HLIROp for Add {
    fn to_egglog(&self, inputs: &[(NodeIndex, String)]) -> String {
        format!(
            "(Op (Add {} {} {} {}) {})",
            elist_to_egglog(&self.input_shapes[0].dims),
            elist_to_egglog(&self.input_shapes[0].strides),
            elist_to_egglog(&self.input_shapes[1].strides),
            elist_to_egglog(&self.input_shapes[0].contiguous().strides),
            ilist_egglog(&[&inputs[0].1, &inputs[1].1]),
        )
    }
}

impl EgglogOp for Add {
    fn sort(&self) -> SortDef {
        binary_sort("Add")
    }
    fn cleanup(&self) -> bool {
        true
    }
    fn n_inputs(&self) -> usize {
        2
    }
    fn rewrites(&self) -> Vec<Rule> {
        vec![dtype_propagation_op(&self.sort())]
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn NativeOp>(Box::new(Self {
                shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap(),
                a_strides: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                b_strides: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                ..Default::default()
            })),
            input_enodes,
        )
    }
}

impl NativeOp for Add {
    fn execute(&self, inputs: Vec<&NativeData>, dyn_map: &FxHashMap<char, usize>) -> NativeData {
        let (a, b) = (inputs[0], inputs[1]);
        let (a_ind, b_ind) = (
            StridedIterator::new(&self.shape, &self.a_strides, dyn_map),
            StridedIterator::new(&self.shape, &self.b_strides, dyn_map),
        );
        match a {
            NativeData::F32(a) => {
                NativeData::F32(bin_fn(a_ind, a, b_ind, b, NativeData::f32, |x, y| x + y))
            }
            NativeData::F16(a) => {
                NativeData::F16(bin_fn(a_ind, a, b_ind, b, NativeData::f16, |x, y| x + y))
            }
            NativeData::Bf16(a) => {
                NativeData::Bf16(bin_fn(a_ind, a, b_ind, b, NativeData::bf16, |x, y| x + y))
            }
            NativeData::Int(a) => {
                NativeData::Int(bin_fn(a_ind, a, b_ind, b, NativeData::i32, |x, y| x + y))
            }
            NativeData::Bool(_) => panic!("Cannot add Bool tensors, cast to F32 first"),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Mul {
    pub shape: Vec<Expression>,
    pub a_strides: Vec<Expression>,
    pub b_strides: Vec<Expression>,
    pub input_shapes: Vec<ShapeTracker>,
}
impl Display for Mul {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Mul")
    }
}
impl HLIROp for Mul {
    fn to_egglog(&self, inputs: &[(NodeIndex, String)]) -> String {
        format!(
            "(Op (Mul {} {} {} {}) {})",
            elist_to_egglog(&self.input_shapes[0].dims),
            elist_to_egglog(&self.input_shapes[0].strides),
            elist_to_egglog(&self.input_shapes[1].strides),
            elist_to_egglog(&self.input_shapes[0].contiguous().strides),
            ilist_egglog(&[&inputs[0].1, &inputs[1].1]),
        )
    }
}

impl EgglogOp for Mul {
    fn sort(&self) -> SortDef {
        binary_sort("Mul")
    }
    fn cleanup(&self) -> bool {
        true
    }
    fn n_inputs(&self) -> usize {
        2
    }
    fn rewrites(&self) -> Vec<Rule> {
        vec![dtype_propagation_op(&self.sort())]
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn NativeOp>(Box::new(Self {
                shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap(),
                a_strides: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                b_strides: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                ..Default::default()
            })),
            input_enodes,
        )
    }
}

impl NativeOp for Mul {
    fn execute(&self, inputs: Vec<&NativeData>, dyn_map: &FxHashMap<char, usize>) -> NativeData {
        let (a, b) = (inputs[0], inputs[1]);
        let (a_ind, b_ind) = (
            StridedIterator::new(&self.shape, &self.a_strides, dyn_map),
            StridedIterator::new(&self.shape, &self.b_strides, dyn_map),
        );
        match a {
            NativeData::F32(a) => {
                NativeData::F32(bin_fn(a_ind, a, b_ind, b, NativeData::f32, |x, y| x * y))
            }
            NativeData::F16(a) => {
                NativeData::F16(bin_fn(a_ind, a, b_ind, b, NativeData::f16, |x, y| x * y))
            }
            NativeData::Bf16(a) => {
                NativeData::Bf16(bin_fn(a_ind, a, b_ind, b, NativeData::bf16, |x, y| x * y))
            }
            NativeData::Int(a) => {
                NativeData::Int(bin_fn(a_ind, a, b_ind, b, NativeData::i32, |x, y| x * y))
            }
            NativeData::Bool(_) => panic!("Cannot multiply Bool tensors, cast to F32 first"),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Mod {
    pub shape: Vec<Expression>,
    pub a_strides: Vec<Expression>,
    pub b_strides: Vec<Expression>,
    pub input_shapes: Vec<ShapeTracker>,
}
impl Display for Mod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Mod")
    }
}
impl HLIROp for Mod {
    fn to_egglog(&self, inputs: &[(NodeIndex, String)]) -> String {
        format!(
            "(Op (Mod {} {} {} {}) {})",
            elist_to_egglog(&self.input_shapes[0].dims),
            elist_to_egglog(&self.input_shapes[0].strides),
            elist_to_egglog(&self.input_shapes[1].strides),
            elist_to_egglog(&self.input_shapes[0].contiguous().strides),
            ilist_egglog(&[&inputs[0].1, &inputs[1].1]),
        )
    }
}

impl EgglogOp for Mod {
    fn sort(&self) -> SortDef {
        binary_sort("Mod")
    }
    fn cleanup(&self) -> bool {
        true
    }
    fn n_inputs(&self) -> usize {
        2
    }
    fn rewrites(&self) -> Vec<Rule> {
        vec![dtype_propagation_op(&self.sort())]
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn NativeOp>(Box::new(Self {
                shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap(),
                a_strides: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                b_strides: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                ..Default::default()
            })),
            input_enodes,
        )
    }
}

impl NativeOp for Mod {
    fn execute(&self, inputs: Vec<&NativeData>, dyn_map: &FxHashMap<char, usize>) -> NativeData {
        let (a, b) = (inputs[0], inputs[1]);
        let (a_ind, b_ind) = (
            StridedIterator::new(&self.shape, &self.a_strides, dyn_map),
            StridedIterator::new(&self.shape, &self.b_strides, dyn_map),
        );
        match a {
            NativeData::F32(a) => {
                NativeData::F32(bin_fn(a_ind, a, b_ind, b, NativeData::f32, |x, y| x % y))
            }
            NativeData::F16(a) => {
                NativeData::F16(bin_fn(a_ind, a, b_ind, b, NativeData::f16, |x, y| x % y))
            }
            NativeData::Bf16(a) => {
                NativeData::Bf16(bin_fn(a_ind, a, b_ind, b, NativeData::bf16, |x, y| x % y))
            }
            NativeData::Int(a) => {
                NativeData::Int(bin_fn(a_ind, a, b_ind, b, NativeData::i32, |x, y| x % y))
            }
            NativeData::Bool(_) => panic!("Cannot mod Bool tensors"),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct LessThan {
    pub shape: Vec<Expression>,
    pub a_strides: Vec<Expression>,
    pub b_strides: Vec<Expression>,
    pub input_shapes: Vec<ShapeTracker>,
}
impl Display for LessThan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LessThan")
    }
}
impl HLIROp for LessThan {
    fn to_egglog(&self, inputs: &[(NodeIndex, String)]) -> String {
        format!(
            "(Op (LessThan {} {} {} {}) {})",
            elist_to_egglog(&self.input_shapes[0].dims),
            elist_to_egglog(&self.input_shapes[0].strides),
            elist_to_egglog(&self.input_shapes[1].strides),
            elist_to_egglog(&self.input_shapes[0].contiguous().strides),
            ilist_egglog(&[&inputs[0].1, &inputs[1].1]),
        )
    }
}

impl EgglogOp for LessThan {
    fn sort(&self) -> SortDef {
        binary_sort("LessThan")
    }
    fn cleanup(&self) -> bool {
        true
    }
    fn n_inputs(&self) -> usize {
        2
    }
    fn rewrites(&self) -> Vec<Rule> {
        // Comparison operations always output Bool
        vec![dtype_fixed_op(&self.sort(), &SORTS.bool_dt)]
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn NativeOp>(Box::new(Self {
                shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap(),
                a_strides: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                b_strides: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                ..Default::default()
            })),
            input_enodes,
        )
    }
}

impl NativeOp for LessThan {
    fn execute(&self, inputs: Vec<&NativeData>, dyn_map: &FxHashMap<char, usize>) -> NativeData {
        let (a, b) = (inputs[0], inputs[1]);
        let (a_ind, b_ind) = (
            StridedIterator::new(&self.shape, &self.a_strides, dyn_map),
            StridedIterator::new(&self.shape, &self.b_strides, dyn_map),
        );
        // Comparison always returns Bool
        NativeData::Bool(
            a_ind
                .zip(b_ind)
                .map(|(i, j)| NativeData::f32(a, i) < NativeData::f32(b, j))
                .collect(),
        )
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Gather {
    pub index_shape: Vec<Expression>,
    pub data_shape: Vec<Expression>,
    pub index_strides: Vec<Expression>,
    pub data_strides: Vec<Expression>,
    pub input_shapes: Vec<ShapeTracker>,
}
impl Display for Gather {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Gather")
    }
}
impl HLIROp for Gather {
    fn to_egglog(&self, inputs: &[(NodeIndex, String)]) -> String {
        format!(
            "(Op (Gather {} {} {} {}) {})",
            elist_to_egglog(&self.input_shapes[0].dims),
            elist_to_egglog(&self.input_shapes[0].strides),
            elist_to_egglog(&self.input_shapes[1].dims),
            elist_to_egglog(&self.input_shapes[1].strides),
            ilist_egglog(&[&inputs[0].1, &inputs[1].1]),
        )
    }
}

impl EgglogOp for Gather {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "Gather",
            &[
                ("index_shape", ELIST),
                ("index_strides", ELIST),
                ("data_shape", ELIST),
                ("data_strides", ELIST),
            ],
        )
    }
    fn cleanup(&self) -> bool {
        true
    }
    fn n_inputs(&self) -> usize {
        2
    }
    fn rewrites(&self) -> Vec<Rule> {
        // Gather inherits dtype from second input (data), not first (indexes).
        // Use a custom rule instead of the generic first-input propagation.
        let (_, kind_term) = self.sort().new_call();
        let e = v("__e");
        let indexes = v("__indexes");
        let data = v("__data");
        let tail = v("__tail");
        let dty = v("__dty");
        vec![
            Rule::new()
                .fact(eq(
                    e.clone(),
                    op_term(
                        kind_term,
                        Term::App {
                            variant: "ICons".to_string(),
                            args: vec![
                                indexes,
                                Term::App {
                                    variant: "ICons".to_string(),
                                    args: vec![data.clone(), tail],
                                },
                            ],
                        },
                    ),
                ))
                .fact(eq(dty.clone(), dtype(data)))
                .action(Action::Set(dtype(e), dty)),
        ]
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn NativeOp>(Box::new(Self {
                index_shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache)
                    .unwrap(),
                index_strides: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                data_shape: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                data_strides: extract_expr_list(egraph, kind_children[3], list_cache, expr_cache)
                    .unwrap(),
                ..Default::default()
            })),
            input_enodes,
        )
    }
}
impl NativeOp for Gather {
    fn execute(&self, inputs: Vec<&NativeData>, dyn_map: &FxHashMap<char, usize>) -> NativeData {
        let (indexes, data) = (inputs[0], inputs[1]);
        let indexes_ind = StridedIterator::new(&self.index_shape, &self.index_strides, dyn_map);
        let data_ind =
            StridedIterator::new(&self.data_shape, &self.data_strides, dyn_map).collect_vec();
        let NativeData::Int(indexes) = indexes else {
            panic!("indexes must be int!")
        };
        match data {
            NativeData::F32(a) => NativeData::F32(
                indexes_ind
                    .map(|i| a[data_ind[indexes[i] as usize]])
                    .collect(),
            ),
            NativeData::F16(a) => NativeData::F16(
                indexes_ind
                    .map(|i| a[data_ind[indexes[i] as usize]])
                    .collect(),
            ),
            NativeData::Bf16(a) => NativeData::Bf16(
                indexes_ind
                    .map(|i| a[data_ind[indexes[i] as usize]])
                    .collect(),
            ),
            NativeData::Int(a) => NativeData::Int(
                indexes_ind
                    .map(|i| a[data_ind[indexes[i] as usize]])
                    .collect(),
            ),
            NativeData::Bool(a) => NativeData::Bool(
                indexes_ind
                    .map(|i| a[data_ind[indexes[i] as usize]])
                    .collect(),
            ),
        }
    }
}

// Scatter Op (inverse of Gather)

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Scatter {
    pub dest_shape: Vec<Expression>,
    pub dest_strides: Vec<Expression>,
    pub index_shape: Vec<Expression>,
    pub index_strides: Vec<Expression>,
    pub src_strides: Vec<Expression>,
}
impl Display for Scatter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Scatter")
    }
}
impl HLIROp for Scatter {
    fn to_egglog(&self, inputs: &[(NodeIndex, String)]) -> String {
        format!(
            "(Op (Scatter {} {} {} {} {}) {})",
            elist_to_egglog(&self.dest_shape),
            elist_to_egglog(&self.dest_strides),
            elist_to_egglog(&self.index_shape),
            elist_to_egglog(&self.index_strides),
            elist_to_egglog(&self.src_strides),
            ilist_egglog(&[&inputs[0].1, &inputs[1].1, &inputs[2].1]),
        )
    }
}

impl EgglogOp for Scatter {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "Scatter",
            &[
                ("dest_shape", ELIST),
                ("dest_strides", ELIST),
                ("index_shape", ELIST),
                ("index_strides", ELIST),
                ("src_strides", ELIST),
            ],
        )
    }
    fn cleanup(&self) -> bool {
        true
    }
    fn n_inputs(&self) -> usize {
        3
    }
    fn rewrites(&self) -> Vec<Rule> {
        // Scatter inherits dtype from third input (src), not first (dest).
        let (_, kind_term) = self.sort().new_call();
        let e = v("__e");
        let dest = v("__dest");
        let indexes = v("__indexes");
        let src = v("__src");
        let tail = v("__tail");
        let dty = v("__dty");
        vec![
            Rule::new()
                .fact(eq(
                    e.clone(),
                    op_term(
                        kind_term,
                        Term::App {
                            variant: "ICons".to_string(),
                            args: vec![
                                dest,
                                Term::App {
                                    variant: "ICons".to_string(),
                                    args: vec![
                                        indexes,
                                        Term::App {
                                            variant: "ICons".to_string(),
                                            args: vec![src.clone(), tail],
                                        },
                                    ],
                                },
                            ],
                        },
                    ),
                ))
                .fact(eq(dty.clone(), dtype(src)))
                .action(Action::Set(dtype(e), dty)),
        ]
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn NativeOp>(Box::new(Self {
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
            })),
            input_enodes,
        )
    }
}
impl NativeOp for Scatter {
    fn execute(&self, inputs: Vec<&NativeData>, dyn_map: &FxHashMap<char, usize>) -> NativeData {
        let (dest, indexes, src) = (inputs[0], inputs[1], inputs[2]);
        let dest_ind =
            StridedIterator::new(&self.dest_shape, &self.dest_strides, dyn_map).collect_vec();
        let index_ind = StridedIterator::new(&self.index_shape, &self.index_strides, dyn_map);
        let src_ind =
            StridedIterator::new(&self.index_shape, &self.src_strides, dyn_map).collect_vec();
        let NativeData::Int(indexes) = indexes else {
            panic!("indexes must be int!")
        };
        macro_rules! scatter_impl {
            ($variant:ident, $dest_data:expr, $src_data:expr) => {{
                let mut output: Vec<_> = dest_ind.iter().map(|&i| $dest_data[i]).collect();
                for (src_idx, flat_i) in index_ind.enumerate() {
                    let idx = indexes[flat_i] as usize;
                    if idx < output.len() {
                        output[idx] = $src_data[src_ind[src_idx]];
                    }
                }
                NativeData::$variant(output)
            }};
        }
        match (dest, src) {
            (NativeData::F32(d), NativeData::F32(s)) => scatter_impl!(F32, d, s),
            (NativeData::F16(d), NativeData::F16(s)) => scatter_impl!(F16, d, s),
            (NativeData::Bf16(d), NativeData::Bf16(s)) => scatter_impl!(Bf16, d, s),
            (NativeData::Int(d), NativeData::Int(s)) => scatter_impl!(Int, d, s),
            (NativeData::Bool(d), NativeData::Bool(s)) => scatter_impl!(Bool, d, s),
            _ => panic!("dest and src must have the same dtype!"),
        }
    }
}

// Reduce Ops (A -> B (different shape))

#[derive(Debug, Clone, PartialEq, Default)]
pub struct SumReduce {
    pub dim: usize,
    pub shape: Vec<Expression>,
    pub strides: Vec<Expression>,
    pub iters: Expression,
    pub iter_stride: Expression,
    pub input_shape: ShapeTracker,
}
impl Display for SumReduce {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SumReduce(dim={})", self.dim)
    }
}
impl HLIROp for SumReduce {
    fn to_egglog(&self, inputs: &[(NodeIndex, String)]) -> String {
        let mut reduced_shape = self.input_shape;
        reduced_shape.remove_dim(self.dim);
        let reduced_dim = self.input_shape.dims[self.dim];
        let reduced_stride = self.input_shape.strides[self.dim];
        let mut reduced_strides = self.input_shape.strides;
        reduced_strides.remove(self.dim);

        format!(
            "(Op (Sum {} {} {} {} {}) {})",
            elist_to_egglog(&reduced_shape.dims),
            reduced_dim.to_egglog(),
            elist_to_egglog(&reduced_strides),
            reduced_stride.to_egglog(),
            elist_to_egglog(&reduced_shape.contiguous().strides),
            ilist_egglog(&[&inputs[0].1]),
        )
    }
}

impl EgglogOp for SumReduce {
    fn sort(&self) -> SortDef {
        reduce_sort("Sum")
    }
    fn cleanup(&self) -> bool {
        true
    }
    fn n_inputs(&self) -> usize {
        1
    }
    fn rewrites(&self) -> Vec<Rule> {
        vec![
            dtype_propagation_op(&self.sort()),
            // Batch-collapse rules: rewrite N-dim Mul+Sum → (N-1)-dim Mul+Sum
            // so that 2D cuBLAS rules can match. Fires recursively.
            Rule::raw(include_str!("egglog_utils/matmul_flattening/squeeze.egg")),
            Rule::raw(include_str!(
                "egglog_utils/matmul_flattening/batch_merge_a_contig.egg"
            )),
            Rule::raw(include_str!(
                "egglog_utils/matmul_flattening/batch_merge_b_contig.egg"
            )),
        ]
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn NativeOp>(Box::new(Self {
                dim: 0,
                shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap(),
                iters: extract_expr(egraph, kind_children[1], expr_cache).unwrap(),
                strides: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                iter_stride: extract_expr(egraph, kind_children[3], expr_cache).unwrap(),
                ..Default::default()
            })),
            input_enodes,
        )
    }
}

impl NativeOp for SumReduce {
    fn execute(&self, inputs: Vec<&NativeData>, dyn_map: &FxHashMap<char, usize>) -> NativeData {
        let ind = StridedIterator::new(&self.shape, &self.strides, dyn_map);
        // Resolve dyn vars in iter_stride, then evaluate z-stride at each iteration
        let mut resolved_stride = self.iter_stride;
        for (&var, &val) in dyn_map {
            resolved_stride = resolved_stride.substitute(var, Expression::from(val as i32));
        }
        let iters = self.iters.exec(dyn_map).unwrap();
        match inputs[0] {
            NativeData::F32(a) => NativeData::F32(
                ind.map(|start| {
                    (0..iters)
                        .map(|i| a[start + resolved_stride.exec_single_var(i)])
                        .sum()
                })
                .collect(),
            ),
            NativeData::F16(a) => NativeData::F16(
                ind.map(|start| {
                    (0..iters)
                        .map(|i| a[start + resolved_stride.exec_single_var(i)])
                        .sum()
                })
                .collect(),
            ),
            NativeData::Bf16(a) => NativeData::Bf16(
                ind.map(|start| {
                    (0..iters)
                        .map(|i| a[start + resolved_stride.exec_single_var(i)])
                        .sum()
                })
                .collect(),
            ),
            NativeData::Int(a) => NativeData::Int(
                ind.map(|start| {
                    (0..iters)
                        .map(|i| a[start + resolved_stride.exec_single_var(i)])
                        .sum()
                })
                .collect(),
            ),
            NativeData::Bool(_) => panic!("Cannot sum Bool tensors, cast to F32 first"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct MaxReduce {
    pub dim: usize,
    pub shape: Vec<Expression>,
    pub strides: Vec<Expression>,
    pub iters: Expression,
    pub iter_stride: Expression,
    pub input_shape: ShapeTracker,
}
impl Display for MaxReduce {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MaxReduce(dim={})", self.dim)
    }
}
impl HLIROp for MaxReduce {
    fn to_egglog(&self, inputs: &[(NodeIndex, String)]) -> String {
        let mut reduced_shape = self.input_shape;
        reduced_shape.remove_dim(self.dim);
        let reduced_dim = self.input_shape.dims[self.dim];
        let reduced_stride = self.input_shape.strides[self.dim];
        let mut reduced_strides = self.input_shape.strides;
        reduced_strides.remove(self.dim);
        format!(
            "(Op (Max {} {} {} {} {}) {})",
            elist_to_egglog(&reduced_shape.dims),
            reduced_dim.to_egglog(),
            elist_to_egglog(&reduced_strides),
            reduced_stride.to_egglog(),
            elist_to_egglog(&reduced_shape.contiguous().strides),
            ilist_egglog(&[&inputs[0].1]),
        )
    }
}

impl EgglogOp for MaxReduce {
    fn sort(&self) -> SortDef {
        reduce_sort("Max")
    }
    fn cleanup(&self) -> bool {
        true
    }
    fn n_inputs(&self) -> usize {
        1
    }
    fn rewrites(&self) -> Vec<Rule> {
        vec![dtype_propagation_op(&self.sort())]
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn NativeOp>(Box::new(Self {
                dim: 0,
                shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap(),
                iters: extract_expr(egraph, kind_children[1], expr_cache).unwrap(),
                strides: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                iter_stride: extract_expr(egraph, kind_children[3], expr_cache).unwrap(),
                ..Default::default()
            })),
            input_enodes,
        )
    }
}

impl NativeOp for MaxReduce {
    fn execute(&self, inputs: Vec<&NativeData>, dyn_map: &FxHashMap<char, usize>) -> NativeData {
        let ind = StridedIterator::new(&self.shape, &self.strides, dyn_map);
        // Resolve dyn vars in iter_stride, then evaluate z-stride at each iteration
        let mut resolved_stride = self.iter_stride;
        for (&var, &val) in dyn_map {
            resolved_stride = resolved_stride.substitute(var, Expression::from(val as i32));
        }
        let iters = self.iters.exec(dyn_map).unwrap();
        match inputs[0] {
            NativeData::F32(a) => NativeData::F32(
                ind.map(|start| {
                    (0..iters)
                        .map(|i| a[start + resolved_stride.exec_single_var(i)])
                        .max_by(|a, b| a.total_cmp(b))
                        .unwrap_or_default()
                })
                .collect(),
            ),
            NativeData::F16(a) => NativeData::F16(
                ind.map(|start| {
                    (0..iters)
                        .map(|i| a[start + resolved_stride.exec_single_var(i)])
                        .max_by(|a, b| a.total_cmp(b))
                        .unwrap_or_default()
                })
                .collect(),
            ),
            NativeData::Bf16(a) => NativeData::Bf16(
                ind.map(|start| {
                    (0..iters)
                        .map(|i| a[start + resolved_stride.exec_single_var(i)])
                        .max_by(|a, b| a.total_cmp(b))
                        .unwrap_or_default()
                })
                .collect(),
            ),
            NativeData::Int(a) => NativeData::Int(
                ind.map(|start| {
                    (0..iters)
                        .map(|i| a[start + resolved_stride.exec_single_var(i)])
                        .max()
                        .unwrap_or_default()
                })
                .collect(),
            ),
            NativeData::Bool(_) => panic!("Cannot max-reduce Bool tensors"),
        }
    }
}

// Fused Softmax: softmax(x, axis) = exp(x - max(x)) / sum(exp(x - max(x)))
// A single HLIR op that replaces the 6-op decomposed chain.
// On CUDA, KernelSoftmax provides a fused 3-pass kernel.
// On native, NativeOp implements softmax directly.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Softmax {
    pub axis: usize,
    pub input_shape: ShapeTracker,
    // Extracted fields (populated during egglog extraction, used by NativeOp)
    pub shape: Vec<Expression>,
    pub in_strides: Vec<Expression>,
    pub reduce_dim: Expression,
    pub reduce_stride: Expression,
}
impl Display for Softmax {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Softmax(axis={})", self.axis)
    }
}

/// Sort for Softmax: (shape, in_strides, out_strides, reduce_dim, reduce_stride)
pub fn softmax_sort(name: &str) -> SortDef {
    sort(
        OP_KIND,
        name,
        &[
            ("shape", ELIST),
            ("in_strides", ELIST),
            ("out_strides", ELIST),
            ("reduce_dim", EXPRESSION),
            ("reduce_stride", EXPRESSION),
        ],
    )
}

impl HLIROp for Softmax {
    fn to_egglog(&self, inputs: &[(NodeIndex, String)]) -> String {
        let reduce_dim = self.input_shape.dims[self.axis];
        let reduce_stride = self.input_shape.strides[self.axis];
        format!(
            "(Op (Softmax {} {} {} {} {}) {})",
            elist_to_egglog(&self.input_shape.dims),
            elist_to_egglog(&self.input_shape.strides),
            elist_to_egglog(&self.input_shape.contiguous().strides),
            reduce_dim.to_egglog(),
            reduce_stride.to_egglog(),
            ilist_egglog(&[&inputs[0].1]),
        )
    }
}

impl EgglogOp for Softmax {
    fn sort(&self) -> SortDef {
        softmax_sort("Softmax")
    }
    fn cleanup(&self) -> bool {
        true
    }
    fn n_inputs(&self) -> usize {
        1
    }
    fn rewrites(&self) -> Vec<Rule> {
        vec![dtype_propagation_op(&self.sort())]
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let shape = extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap();
        let in_strides =
            extract_expr_list(egraph, kind_children[1], list_cache, expr_cache).unwrap();
        let reduce_dim = extract_expr(egraph, kind_children[3], expr_cache).unwrap();
        let reduce_stride = extract_expr(egraph, kind_children[4], expr_cache).unwrap();
        (
            LLIROp::new::<dyn NativeOp>(Box::new(Self {
                axis: 0,
                input_shape: ShapeTracker::default(),
                shape,
                in_strides,
                reduce_dim,
                reduce_stride,
            })),
            input_enodes,
        )
    }
}

impl NativeOp for Softmax {
    fn execute(&self, inputs: Vec<&NativeData>, dyn_map: &FxHashMap<char, usize>) -> NativeData {
        match inputs[0] {
            NativeData::F32(a) => {
                // Use extracted fields (populated during egglog extraction)
                let dims: Vec<usize> = self
                    .shape
                    .iter()
                    .map(|d| d.exec(dyn_map).unwrap())
                    .collect();
                let n = self.reduce_dim.exec(dyn_map).unwrap();
                let mut reduce_stride_expr = self.reduce_stride;
                for (&var, &val) in dyn_map {
                    reduce_stride_expr =
                        reduce_stride_expr.substitute(var, Expression::from(val as i32));
                }

                // Compute row index strides (all dims except last, since softmax is always last-dim)
                let ndim = dims.len();
                let out_size: usize = dims.iter().product();
                let mut out = vec![0.0f32; out_size];

                // Use StridedIterator for the row dimensions
                let row_ind = StridedIterator::new(
                    &self.shape[..ndim - 1],
                    &self.in_strides[..ndim - 1],
                    dyn_map,
                );

                for (row_idx, in_base) in row_ind.enumerate() {
                    // Pass 1: find max
                    let mut max_val = f32::NEG_INFINITY;
                    for i in 0..n {
                        let val = a[in_base + reduce_stride_expr.exec_single_var(i)];
                        if val > max_val {
                            max_val = val;
                        }
                    }

                    // Pass 2: exp(x - max) and sum
                    let mut sum = 0.0f32;
                    let out_base = row_idx * n;
                    for i in 0..n {
                        let val =
                            (a[in_base + reduce_stride_expr.exec_single_var(i)] - max_val).exp();
                        out[out_base + i] = val;
                        sum += val;
                    }

                    // Pass 3: normalize
                    let inv_sum = 1.0 / sum;
                    for i in 0..n {
                        out[out_base + i] *= inv_sum;
                    }
                }

                NativeData::F32(out)
            }
            _ => panic!("Softmax only supports F32"),
        }
    }
}

pub trait NativeOp: Debug + AsAny + Send + Sync {
    fn execute(&self, inputs: Vec<&NativeData>, dyn_map: &FxHashMap<char, usize>) -> NativeData;
}

#[derive(Debug, Clone)]
pub enum NativeData {
    F32(Vec<f32>),
    F16(Vec<f16>),
    Bf16(Vec<bf16>),
    Int(Vec<i32>),
    Bool(Vec<bool>),
}

impl NativeData {
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn len(&self) -> usize {
        match self {
            NativeData::F32(v) => v.len(),
            NativeData::F16(v) => v.len(),
            NativeData::Bf16(v) => v.len(),
            NativeData::Int(v) => v.len(),
            NativeData::Bool(v) => v.len(),
        }
    }
    #[inline]
    pub fn f32(&self, i: usize) -> f32 {
        match self {
            NativeData::F32(v) => v[i],
            NativeData::F16(v) => v[i].to_f32(),
            NativeData::Bf16(v) => v[i].to_f32(),
            NativeData::Int(v) => v[i] as f32,
            NativeData::Bool(v) => {
                if v[i] {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    #[inline]
    pub fn f16(&self, i: usize) -> f16 {
        match self {
            NativeData::F16(v) => v[i],
            NativeData::F32(v) => f16::from_f32(v[i]),
            NativeData::Bf16(v) => f16::from_f32(v[i].to_f32()),
            NativeData::Int(v) => f16::from_f32(v[i] as f32),
            NativeData::Bool(v) => f16::from_f32(if v[i] { 1.0 } else { 0.0 }),
        }
    }

    #[inline]
    pub fn bf16(&self, i: usize) -> bf16 {
        match self {
            NativeData::Bf16(v) => v[i],
            NativeData::F32(v) => bf16::from_f32(v[i]),
            NativeData::F16(v) => bf16::from_f32(v[i].to_f32()),
            NativeData::Int(v) => bf16::from_f32(v[i] as f32),
            NativeData::Bool(v) => bf16::from_f32(if v[i] { 1.0 } else { 0.0 }),
        }
    }

    #[inline]
    pub fn i32(&self, i: usize) -> i32 {
        match self {
            NativeData::Int(v) => v[i],
            NativeData::F32(v) => v[i] as i32,
            NativeData::F16(v) => v[i].to_f32() as i32,
            NativeData::Bf16(v) => v[i].to_f32() as i32,
            NativeData::Bool(v) => {
                if v[i] {
                    1
                } else {
                    0
                }
            }
        }
    }

    #[inline]
    pub fn bool(&self, i: usize) -> bool {
        match self {
            NativeData::Bool(v) => v[i],
            NativeData::F32(v) => v[i] != 0.0,
            NativeData::F16(v) => v[i].to_f32() != 0.0,
            NativeData::Bf16(v) => v[i].to_f32() != 0.0,
            NativeData::Int(v) => v[i] != 0,
        }
    }
}

impl From<Vec<f32>> for NativeData {
    fn from(value: Vec<f32>) -> Self {
        NativeData::F32(value)
    }
}
impl From<Vec<f16>> for NativeData {
    fn from(value: Vec<f16>) -> Self {
        NativeData::F16(value)
    }
}
impl From<Vec<bf16>> for NativeData {
    fn from(value: Vec<bf16>) -> Self {
        NativeData::Bf16(value)
    }
}
impl From<Vec<i32>> for NativeData {
    fn from(value: Vec<i32>) -> Self {
        NativeData::Int(value)
    }
}
impl From<Vec<bool>> for NativeData {
    fn from(value: Vec<bool>) -> Self {
        NativeData::Bool(value)
    }
}

macro_rules! impl_native_data_from_ref {
    ($ty:ty, $variant:ident) => {
        impl From<&[$ty]> for NativeData {
            fn from(value: &[$ty]) -> Self {
                NativeData::$variant(value.to_vec())
            }
        }

        impl From<&Vec<$ty>> for NativeData {
            fn from(value: &Vec<$ty>) -> Self {
                NativeData::$variant(value.clone())
            }
        }
    };
}

macro_rules! impl_native_data_from_array_ref {
    ($ty:ty, $variant:ident) => {
        impl<const N: usize> From<&[$ty; N]> for NativeData {
            fn from(value: &[$ty; N]) -> Self {
                NativeData::$variant(value.to_vec())
            }
        }
    };
}

impl_native_data_from_ref!(f32, F32);
impl_native_data_from_ref!(f16, F16);
impl_native_data_from_ref!(bf16, Bf16);
impl_native_data_from_ref!(i32, Int);
impl_native_data_from_ref!(bool, Bool);

impl_native_data_from_array_ref!(f32, F32);
impl_native_data_from_array_ref!(f16, F16);
impl_native_data_from_array_ref!(bf16, Bf16);
impl_native_data_from_array_ref!(i32, Int);
impl_native_data_from_array_ref!(bool, Bool);

#[derive(Default)]
pub struct NativeRuntime {
    pub buffers: FxHashMap<NodeIndex, NativeData>,
    pub graph: StableGraph<Arc<Box<dyn NativeOp>>, ()>,
}

impl NativeRuntime {
    pub fn set_data(&mut self, id: impl ToId, data: impl Into<NativeData>) {
        let id = id.to_id();
        let local_id = self
            .graph
            .node_indices()
            .find(|n| {
                if let Some(Input { node, .. }) = (**self.graph[*n]).as_any().downcast_ref() {
                    *node == id.index()
                } else {
                    false
                }
            })
            .unwrap_or_else(|| panic!("{id:?} is not an Input node in the graph"));
        self.buffers.insert(local_id, data.into());
    }
}

impl Runtime for NativeRuntime {
    type Ops = ();
    type CompileArg = ();
    type ExecReturn = ();
    type ProfileMetric = usize;

    fn initialize(_: Self::CompileArg) -> Self {
        Self {
            buffers: Default::default(),
            graph: Default::default(),
        }
    }

    fn profile(
        &mut self,
        _: &LLIRGraph,
        _: &FxHashMap<char, usize>,
        _: usize,
    ) -> (Self::ProfileMetric, String) {
        (0, "0 ms".to_string())
    }

    fn load_llir(&mut self, llir_graph: &LLIRGraph) {
        // Extract nativeop graph
        let mut graph = StableGraph::new();
        for node in llir_graph.node_weights() {
            if let Some(op) = node.to_dialect::<dyn NativeOp>() {
                graph.add_node(op.clone());
            } else if let Some(input) = node.to_op::<Input>() {
                graph.add_node(Arc::new(Box::new(input.clone())));
            } else {
                let output = node.to_op::<Output>().unwrap();
                graph.add_node(Arc::new(Box::new(output.clone())));
            }
        }
        for edge in llir_graph.edge_indices() {
            let (start, end) = llir_graph.edge_endpoints(edge).unwrap();
            graph.add_edge(start, end, ());
        }

        self.graph = graph;
    }

    fn execute(&mut self, dyn_map: &FxHashMap<char, usize>) -> Self::ExecReturn {
        for node in toposort(&self.graph, None).unwrap() {
            if (**self.graph[node]).as_any().is::<Input>() {
                continue;
            }

            if (**self.graph[node]).as_any().is::<Output>() {
                // Copy source buffer into Output node's own slot
                let source = self
                    .graph
                    .edges_directed(node, Direction::Incoming)
                    .sorted_by_key(|e| e.id())
                    .next()
                    .unwrap()
                    .source();
                let data = self.buffers[&source].clone();
                self.buffers.insert(node, data);
                continue;
            }

            let span = info_span!("native_op", op = %format!("{:?}", self.graph[node]));
            let _entered = span.enter();
            let inputs = self
                .graph
                .edges_directed(node, Direction::Incoming)
                .sorted_by_key(|e| e.id())
                .map(|e| &self.buffers[&e.source()])
                .collect_vec();
            let output = self.graph[node].execute(inputs, dyn_map);
            self.buffers.insert(node, output);
        }

        // Consume all non-Output buffers (inputs + intermediates)
        let output_nodes: FxHashSet<NodeIndex> = self
            .graph
            .node_indices()
            .filter(|n| (**self.graph[*n]).as_any().is::<Output>())
            .collect();
        self.buffers.retain(|k, _| output_nodes.contains(k));
    }
}

impl NativeRuntime {
    pub fn get_f32(&self, id: impl ToId) -> &Vec<f32> {
        let id = id.to_id();
        let output_id = self
            .graph
            .node_indices()
            .find(|n| {
                if let Some(Output { node }) = (**self.graph[*n]).as_any().downcast_ref::<Output>()
                {
                    *node == id.index()
                } else {
                    false
                }
            })
            .unwrap();
        let NativeData::F32(f) = self.buffers.get(&output_id).unwrap() else {
            panic!()
        };
        f
    }
}

struct StridedIterator {
    shape: Vec<usize>,
    strides: Vec<Expression>,
    index: Vec<usize>,
    done: bool,
}

impl StridedIterator {
    fn new(shape: &[Expression], strides: &[Expression], dyn_map: &FxHashMap<char, usize>) -> Self {
        let shape: Vec<usize> = shape.iter().map(|e| e.exec(dyn_map).unwrap()).collect();
        // Resolve dynamic vars in strides but keep 'z' as a variable
        let strides: Vec<Expression> = strides
            .iter()
            .copied()
            .map(|e| e.resolve_vars(dyn_map))
            .collect();
        Self {
            index: vec![0; shape.len()],
            strides,
            done: shape.contains(&0),
            shape,
        }
    }
}

impl Iterator for StridedIterator {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let fin = self
            .strides
            .iter()
            .zip(self.index.iter())
            .map(|(s, &idx)| s.exec_single_var(idx))
            .sum();

        for i in (0..self.shape.len()).rev() {
            self.index[i] += 1;
            if self.index[i] < self.shape[i] {
                return Some(fin);
            }
            self.index[i] = 0;
        }

        self.done = true;
        Some(fin)
    }
}
