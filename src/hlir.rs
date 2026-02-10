use std::{fmt::Debug, sync::Arc};

use crate::egglog_utils::elist_to_egglog;
use crate::egglog_utils::extract_dtype;
use crate::egglog_utils::extract_expr;
use crate::egglog_utils::extract_expr_list;
use crate::egglog_utils::list_to_egglog;
use crate::op::OpParam::*;
use crate::op::*;
use crate::prelude::*;

use as_any::AsAny;
use itertools::Itertools;
use num_traits::Float;
use petgraph::{Direction, algo::toposort, prelude::StableGraph, visit::EdgeRef};
use rustc_hash::FxHashMap;
use tracing::info_span;

pub type HLIROps = (
    Input,
    Output,
    CustomOpHLIR,
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
    SumReduce,
    MaxReduce,
);

#[derive(Default, Debug, Clone)]
pub struct Input {
    pub node: usize,
    pub label: String,
    pub dtype: DType,
}

impl EgglogOp for Input {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("Input".to_string(), vec![Int, Str, Dty])
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
           ((= ?e (Input ?node ?label ?dty)))
           ((set (dtype ?e) ?dty))
        )"
            .to_string(),
        ]
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        _: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        _: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let node = egraph.enodes[children[0]]
            .0
            .replace("\"", "")
            .parse::<usize>()
            .unwrap();
        let label = egraph.enodes[children[1]].0.replace("\"", "");
        (
            LLIROp::new::<Input>(Box::new(Self {
                node,
                label,
                dtype: extract_dtype(egraph, children[2]),
            })),
            vec![],
        )
    }
}

impl HLIROp for Input {
    fn to_egglog(&self, _: &[(NodeIndex, String, ShapeTracker)]) -> String {
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

impl EgglogOp for Output {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("Output".to_string(), vec![Input, Int])
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
           ((= ?e (Output ?inp ?node)) (= ?dty (dtype ?inp)))
           ((set (dtype ?e) ?dty))
        )"
            .to_string(),
        ]
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        _: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        _: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<Output>(Box::new(Self {
                node: egraph.enodes[children[1]]
                    .0
                    .replace("\"", "")
                    .parse::<usize>()
                    .unwrap(),
            })),
            vec![children[0]],
        )
    }
}

impl HLIROp for Output {
    fn to_egglog(&self, inp: &[(NodeIndex, String, ShapeTracker)]) -> String {
        format!("(Output {} {})", inp[0].1, self.node)
    }
}

impl NativeOp for Output {
    fn execute(&self, _: Vec<&NativeData>, _: &FxHashMap<char, usize>) -> NativeData {
        unimplemented!()
    }
}

#[derive(Default, Debug, Clone)]
pub struct CustomOpHLIR {
    pub id: usize,
    pub dtype: DType,
}

impl EgglogOp for CustomOpHLIR {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("CustomOpHLIR".to_string(), vec![IList, Int, Dty])
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
           ((= ?e (CustomOpHLIR ?a ?b ?dty)))
           ((set (dtype ?e) ?dty))
        )"
            .to_string(),
        ]
    }

    fn cleanup(&self) -> bool {
        false
    }
}

impl HLIROp for CustomOpHLIR {
    fn to_egglog(&self, inp: &[(NodeIndex, String, ShapeTracker)]) -> String {
        format!(
            "(CustomOpHLIR {} {} ({:?}))",
            list_to_egglog(&inp.iter().map(|i| &i.1).collect_vec(), "ICons", "INil"),
            self.id,
            self.dtype
        )
    }
}

impl NativeOp for CustomOpHLIR {
    fn execute(&self, _: Vec<&NativeData>, _: &FxHashMap<char, usize>) -> NativeData {
        unimplemented!()
    }
}

/// Produces a single number constant from an expression or a float
#[derive(Clone, PartialEq, Default)]
pub struct Constant(pub f32);
impl Debug for Constant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Constant(",)?;
        self.0.fmt(f)?;
        write!(f, ")")
    }
}

impl HLIROp for Constant {
    fn to_egglog(&self, _: &[(NodeIndex, String, ShapeTracker)]) -> String {
        format!("(Constant {:.6})", self.0)
    }
}

impl EgglogOp for Constant {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("Constant".to_string(), vec![Float])
    }
    fn cleanup(&self) -> bool {
        true
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
           ((= ?e (Constant ?f)))
           ((set (dtype ?e) (F32)))
        )"
            .to_string(),
        ]
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        _: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        _: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn NativeOp>(Box::new(Self(
                egraph.enodes[children[0]]
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
impl HLIROp for Iota {
    fn to_egglog(&self, _: &[(NodeIndex, String, ShapeTracker)]) -> String {
        format!("(Iota {} {})", self.0.to_egglog(), self.1.to_egglog())
    }
}
impl EgglogOp for Iota {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("Iota".to_string(), vec![Expr, Expr])
    }

    fn cleanup(&self) -> bool {
        true
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
           ((= ?e (Iota ?expr ?range)))
           ((set (dtype ?e) (Int)))
        )"
            .to_string(),
        ]
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        _: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn NativeOp>(Box::new(Self(
                extract_expr(egraph, children[0], expr_cache).unwrap(),
                extract_expr(egraph, children[1], expr_cache).unwrap(),
            ))),
            vec![],
        )
    }
}
impl NativeOp for Iota {
    fn execute(&self, _: Vec<&NativeData>, dyn_map: &FxHashMap<char, usize>) -> NativeData {
        let length = self.1.exec(dyn_map).unwrap();
        NativeData::Int(
            (0..length)
                .map(|i| self.0.exec_single_var(i) as i32)
                .collect(),
        )
    }
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct Cast(pub Expression, pub DType);
impl HLIROp for Cast {
    fn to_egglog(&self, inp: &[(NodeIndex, String, ShapeTracker)]) -> String {
        format!("(Cast {} {} ({:?}))", inp[0].1, self.0.to_egglog(), self.1)
    }
}
impl EgglogOp for Cast {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("Cast".to_string(), vec![Input, Expr, Dty])
    }

    fn cleanup(&self) -> bool {
        true
    }

    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
           ((= ?e (Cast ?inp ?size ?dty)))
           ((set (dtype ?e) ?dty))
        )"
            .to_string(),
        ]
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        _: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        ec: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn NativeOp>(Box::new(Self(
                extract_expr(egraph, children[1], ec).unwrap(),
                extract_dtype(egraph, children[2]),
            ))),
            vec![children[0]],
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
            DType::NvFp4 => unimplemented!("Cast to NvFp4 is not supported in native interpreter"),
            DType::Mxfp4 => unimplemented!("Cast to Mxfp4 is not supported in native interpreter"),
        }
    }
}

/// Graph break for chunking search graphs
#[derive(Clone, PartialEq)]
pub struct GraphBreak;
impl Debug for GraphBreak {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GraphBreak")
    }
}

impl HLIROp for GraphBreak {
    fn to_egglog(&self, _: &[(NodeIndex, String, ShapeTracker)]) -> String {
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
}
impl HLIROp for Log2 {
    fn to_egglog(&self, inputs: &[(NodeIndex, String, ShapeTracker)]) -> String {
        format!(
            "(Log2 {} {} {} {})",
            elist_to_egglog(&inputs[0].2.dims),
            inputs[0].1,
            elist_to_egglog(&inputs[0].2.strides),
            elist_to_egglog(&inputs[0].2.contiguous().strides)
        )
    }
}
impl EgglogOp for Log2 {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("Log2".to_string(), vec![EList, Input, EList, EList])
    }
    fn cleanup(&self) -> bool {
        true
    }
    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
           ((= ?e (Log2 ?shape ?inp ?a ?b)) (= ?dty (dtype ?inp)))
           ((set (dtype ?e) ?dty))
        )"
            .to_string(),
        ]
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn NativeOp>(Box::new(Self {
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                strides: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
            })),
            vec![children[1]],
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
}
impl HLIROp for Exp2 {
    fn to_egglog(&self, inputs: &[(NodeIndex, String, ShapeTracker)]) -> String {
        format!(
            "(Exp2 {} {} {} {})",
            elist_to_egglog(&inputs[0].2.dims),
            inputs[0].1,
            elist_to_egglog(&inputs[0].2.strides),
            elist_to_egglog(&inputs[0].2.contiguous().strides)
        )
    }
}
impl EgglogOp for Exp2 {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("Exp2".to_string(), vec![EList, Input, EList, EList])
    }
    fn cleanup(&self) -> bool {
        true
    }
    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
           ((= ?e (Exp2 ?shape ?inp ?a ?b)) (= ?dty (dtype ?inp)))
           ((set (dtype ?e) ?dty))
        )"
            .to_string(),
        ]
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn NativeOp>(Box::new(Self {
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                strides: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
            })),
            vec![children[1]],
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
}
impl HLIROp for Sin {
    fn to_egglog(&self, inputs: &[(NodeIndex, String, ShapeTracker)]) -> String {
        format!(
            "(Sin {} {} {} {})",
            elist_to_egglog(&inputs[0].2.dims),
            inputs[0].1,
            elist_to_egglog(&inputs[0].2.strides),
            elist_to_egglog(&inputs[0].2.contiguous().strides)
        )
    }
}

impl EgglogOp for Sin {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("Sin".to_string(), vec![EList, Input, EList, EList])
    }
    fn cleanup(&self) -> bool {
        true
    }
    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
           ((= ?e (Sin ?shape ?inp ?a ?b)) (= ?dty (dtype ?inp)))
           ((set (dtype ?e) ?dty))
        )"
            .to_string(),
        ]
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn NativeOp>(Box::new(Self {
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                strides: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
            })),
            vec![children[1]],
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
}
impl HLIROp for Recip {
    fn to_egglog(&self, inputs: &[(NodeIndex, String, ShapeTracker)]) -> String {
        format!(
            "(Recip {} {} {} {})",
            elist_to_egglog(&inputs[0].2.dims),
            inputs[0].1,
            elist_to_egglog(&inputs[0].2.strides),
            elist_to_egglog(&inputs[0].2.contiguous().strides)
        )
    }
}

impl EgglogOp for Recip {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("Recip".to_string(), vec![EList, Input, EList, EList])
    }
    fn cleanup(&self) -> bool {
        true
    }
    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
           ((= ?e (Recip ?shape ?inp ?a ?b)) (= ?dty (dtype ?inp)))
           ((set (dtype ?e) ?dty))
        )"
            .to_string(),
        ]
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn NativeOp>(Box::new(Self {
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                strides: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
            })),
            vec![children[1]],
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
}
impl HLIROp for Sqrt {
    fn to_egglog(&self, inputs: &[(NodeIndex, String, ShapeTracker)]) -> String {
        format!(
            "(Sqrt {} {} {} {})",
            elist_to_egglog(&inputs[0].2.dims),
            inputs[0].1,
            elist_to_egglog(&inputs[0].2.strides),
            elist_to_egglog(&inputs[0].2.contiguous().strides)
        )
    }
}

impl EgglogOp for Sqrt {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("Sqrt".to_string(), vec![EList, Input, EList, EList])
    }
    fn cleanup(&self) -> bool {
        true
    }
    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
           ((= ?e (Sqrt ?shape ?inp ?a ?b)) (= ?dty (dtype ?inp)))
           ((set (dtype ?e) ?dty))
        )"
            .to_string(),
        ]
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn NativeOp>(Box::new(Self {
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                strides: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
            })),
            vec![children[1]],
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
    a_ind
        .zip(b_ind)
        .map(|(i, j)| op(a[i], b_get(b, j)))
        .collect()
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Add {
    shape: Vec<Expression>,
    a_strides: Vec<Expression>,
    b_strides: Vec<Expression>,
}
impl HLIROp for Add {
    fn to_egglog(&self, inputs: &[(NodeIndex, String, ShapeTracker)]) -> String {
        format!(
            "(Add {} {} {} {} {} {})",
            elist_to_egglog(&inputs[0].2.dims),
            inputs[0].1,
            elist_to_egglog(&inputs[0].2.strides),
            inputs[1].1,
            elist_to_egglog(&inputs[1].2.strides),
            elist_to_egglog(&inputs[0].2.contiguous().strides)
        )
    }
}

impl EgglogOp for Add {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "Add".to_string(),
            vec![EList, Input, EList, Input, EList, EList],
        )
    }
    fn cleanup(&self) -> bool {
        true
    }
    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
           ((= ?e (Add ?shape ?inp_a ?a ?inp_b ?b ?o)) (= ?dty (dtype ?inp_a)))
           ((set (dtype ?e) ?dty))
        )"
            .to_string(),
        ]
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn NativeOp>(Box::new(Self {
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                a_strides: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                b_strides: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
            })),
            vec![children[1], children[3]],
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
    shape: Vec<Expression>,
    a_strides: Vec<Expression>,
    b_strides: Vec<Expression>,
}
impl HLIROp for Mul {
    fn to_egglog(&self, inputs: &[(NodeIndex, String, ShapeTracker)]) -> String {
        format!(
            "(Mul {} {} {} {} {} {})",
            elist_to_egglog(&inputs[0].2.dims),
            inputs[0].1,
            elist_to_egglog(&inputs[0].2.strides),
            inputs[1].1,
            elist_to_egglog(&inputs[1].2.strides),
            elist_to_egglog(&inputs[0].2.contiguous().strides)
        )
    }
}

impl EgglogOp for Mul {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "Mul".to_string(),
            vec![EList, Input, EList, Input, EList, EList],
        )
    }
    fn cleanup(&self) -> bool {
        true
    }
    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
           ((= ?e (Mul ?shape ?inp_a ?a ?inp_b ?b ?o)) (= ?dty (dtype ?inp_a)) (= ?dty (dtype ?inp_b)))
           ((set (dtype ?e) ?dty))
        )"
            .to_string(),
        ]
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn NativeOp>(Box::new(Self {
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                a_strides: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                b_strides: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
            })),
            vec![children[1], children[3]],
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
    shape: Vec<Expression>,
    a_strides: Vec<Expression>,
    b_strides: Vec<Expression>,
}
impl HLIROp for Mod {
    fn to_egglog(&self, inputs: &[(NodeIndex, String, ShapeTracker)]) -> String {
        format!(
            "(Mod {} {} {} {} {} {})",
            elist_to_egglog(&inputs[0].2.dims),
            inputs[0].1,
            elist_to_egglog(&inputs[0].2.strides),
            inputs[1].1,
            elist_to_egglog(&inputs[1].2.strides),
            elist_to_egglog(&inputs[0].2.contiguous().strides)
        )
    }
}

impl EgglogOp for Mod {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "Mod".to_string(),
            vec![EList, Input, EList, Input, EList, EList],
        )
    }
    fn cleanup(&self) -> bool {
        true
    }
    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
           ((= ?e (Mod ?shape ?inp_a ?a ?inp_b ?b ?o)) (= ?dty (dtype ?inp_a)))
           ((set (dtype ?e) ?dty))
        )"
            .to_string(),
        ]
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn NativeOp>(Box::new(Self {
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                a_strides: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                b_strides: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
            })),
            vec![children[1], children[3]],
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
    shape: Vec<Expression>,
    a_strides: Vec<Expression>,
    b_strides: Vec<Expression>,
}
impl HLIROp for LessThan {
    fn to_egglog(&self, inputs: &[(NodeIndex, String, ShapeTracker)]) -> String {
        format!(
            "(LessThan {} {} {} {} {} {})",
            elist_to_egglog(&inputs[0].2.dims),
            inputs[0].1,
            elist_to_egglog(&inputs[0].2.strides),
            inputs[1].1,
            elist_to_egglog(&inputs[1].2.strides),
            elist_to_egglog(&inputs[0].2.contiguous().strides)
        )
    }
}

impl EgglogOp for LessThan {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "LessThan".to_string(),
            vec![EList, Input, EList, Input, EList, EList],
        )
    }
    fn cleanup(&self) -> bool {
        true
    }
    fn rewrites(&self) -> Vec<String> {
        vec![
            // Comparison operations always output Bool
            "(rule
           ((= ?e (LessThan ?shape ?inp_a ?a ?inp_b ?b ?o)))
           ((set (dtype ?e) (Bool)))
        )"
            .to_string(),
        ]
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn NativeOp>(Box::new(Self {
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                a_strides: extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap(),
                b_strides: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
            })),
            vec![children[1], children[3]],
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
    index_shape: Vec<Expression>,
    data_shape: Vec<Expression>,
    index_strides: Vec<Expression>,
    data_strides: Vec<Expression>,
}
impl HLIROp for Gather {
    fn to_egglog(&self, inputs: &[(NodeIndex, String, ShapeTracker)]) -> String {
        format!(
            "(Gather {} {} {} {} {} {})",
            inputs[0].1,
            elist_to_egglog(&inputs[0].2.dims),
            elist_to_egglog(&inputs[0].2.strides),
            inputs[1].1,
            elist_to_egglog(&inputs[1].2.dims),
            elist_to_egglog(&inputs[1].2.strides),
        )
    }
}

impl EgglogOp for Gather {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "Gather".to_string(),
            vec![Input, EList, EList, Input, EList, EList],
        )
    }
    fn cleanup(&self) -> bool {
        true
    }
    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
           ((= ?e (Gather ?indexes ?index_shape ?index_stride ?data ?data_shape ?data_stride)) (= ?dty (dtype ?data)))
           ((set (dtype ?e) ?dty))
        )"
            .to_string(),
        ]
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn NativeOp>(Box::new(Self {
                index_shape: extract_expr_list(egraph, children[1], list_cache, expr_cache)
                    .unwrap(),
                index_strides: extract_expr_list(egraph, children[2], list_cache, expr_cache)
                    .unwrap(),
                data_shape: extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap(),
                data_strides: extract_expr_list(egraph, children[5], list_cache, expr_cache)
                    .unwrap(),
            })),
            vec![children[0], children[3]],
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

// Reduce Ops (A -> B (different shape))

#[derive(Debug, Clone, PartialEq, Default)]
pub struct SumReduce {
    pub dim: usize,
    pub shape: Vec<Expression>,
    pub strides: Vec<Expression>,
    pub iters: Expression,
    pub iter_stride: Expression,
}
impl HLIROp for SumReduce {
    fn to_egglog(&self, inputs: &[(NodeIndex, String, ShapeTracker)]) -> String {
        let mut reduced_shape = inputs[0].2;
        reduced_shape.remove_dim(self.dim);
        let reduced_dim = inputs[0].2.dims[self.dim];
        let reduced_stride = inputs[0].2.strides[self.dim];
        let mut reduced_strides = inputs[0].2.strides;
        reduced_strides.remove(self.dim);
        format!(
            "(Sum {} {} {} {} {} {})",
            elist_to_egglog(&reduced_shape.dims),
            reduced_dim.to_egglog(),
            inputs[0].1,
            elist_to_egglog(&reduced_strides),
            reduced_stride.to_egglog(),
            elist_to_egglog(&reduced_shape.contiguous().strides)
        )
    }
}

impl EgglogOp for SumReduce {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "Sum".to_string(),
            vec![EList, Expr, Input, EList, Expr, EList],
        )
    }
    fn cleanup(&self) -> bool {
        true
    }
    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
           ((= ?e (Sum ?shape ?iters ?inp ?a ?stride ?o)) (= ?dty (dtype ?inp)))
           ((set (dtype ?e) ?dty))
        )"
            .to_string(),
        ]
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn NativeOp>(Box::new(Self {
                dim: 0,
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                strides: extract_expr_list(egraph, children[3], list_cache, expr_cache).unwrap(),
                iters: extract_expr(egraph, children[1], expr_cache).unwrap(),
                iter_stride: extract_expr(egraph, children[4], expr_cache).unwrap(),
            })),
            vec![children[2]],
        )
    }
}

impl NativeOp for SumReduce {
    fn execute(&self, inputs: Vec<&NativeData>, dyn_map: &FxHashMap<char, usize>) -> NativeData {
        let ind = StridedIterator::new(&self.shape, &self.strides, dyn_map);
        let iter_stride = self.iter_stride.exec(dyn_map).unwrap();
        let iters = self.iters.exec(dyn_map).unwrap();
        match inputs[0] {
            NativeData::F32(a) => NativeData::F32(
                ind.map(|start| (0..iters).map(|i| a[start + i * iter_stride]).sum())
                    .collect(),
            ),
            NativeData::F16(a) => NativeData::F16(
                ind.map(|start| (0..iters).map(|i| a[start + i * iter_stride]).sum())
                    .collect(),
            ),
            NativeData::Bf16(a) => NativeData::Bf16(
                ind.map(|start| (0..iters).map(|i| a[start + i * iter_stride]).sum())
                    .collect(),
            ),
            NativeData::Int(a) => NativeData::Int(
                ind.map(|start| (0..iters).map(|i| a[start + i * iter_stride]).sum())
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
}
impl HLIROp for MaxReduce {
    fn to_egglog(&self, inputs: &[(NodeIndex, String, ShapeTracker)]) -> String {
        let mut reduced_shape = inputs[0].2;
        reduced_shape.remove_dim(self.dim);
        let reduced_dim = inputs[0].2.dims[self.dim];
        let reduced_stride = inputs[0].2.strides[self.dim];
        let mut reduced_strides = inputs[0].2.strides;
        reduced_strides.remove(self.dim);
        format!(
            "(Max {} {} {} {} {} {})",
            elist_to_egglog(&reduced_shape.dims),
            reduced_dim.to_egglog(),
            inputs[0].1,
            elist_to_egglog(&reduced_strides),
            reduced_stride.to_egglog(),
            elist_to_egglog(&reduced_shape.contiguous().strides)
        )
    }
}

impl EgglogOp for MaxReduce {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "Max".to_string(),
            vec![EList, Expr, Input, EList, Expr, EList],
        )
    }
    fn cleanup(&self) -> bool {
        true
    }
    fn rewrites(&self) -> Vec<String> {
        vec![
            "(rule
           ((= ?e (Max ?shape ?iters ?inp ?a ?stride ?o)) (= ?dty (dtype ?inp)))
           ((set (dtype ?e) ?dty))
        )"
            .to_string(),
        ]
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn NativeOp>(Box::new(Self {
                dim: 0,
                shape: extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap(),
                strides: extract_expr_list(egraph, children[3], list_cache, expr_cache).unwrap(),
                iters: extract_expr(egraph, children[1], expr_cache).unwrap(),
                iter_stride: extract_expr(egraph, children[4], expr_cache).unwrap(),
            })),
            vec![children[2]],
        )
    }
}

impl NativeOp for MaxReduce {
    fn execute(&self, inputs: Vec<&NativeData>, dyn_map: &FxHashMap<char, usize>) -> NativeData {
        let ind = StridedIterator::new(&self.shape, &self.strides, dyn_map);
        let iter_stride = self.iter_stride.exec(dyn_map).unwrap();
        let iters = self.iters.exec(dyn_map).unwrap();
        match inputs[0] {
            NativeData::F32(a) => NativeData::F32(
                ind.map(|start| {
                    (0..iters)
                        .map(|i| a[start + i * iter_stride])
                        .max_by(|a, b| a.total_cmp(b))
                        .unwrap_or_default()
                })
                .collect(),
            ),
            NativeData::F16(a) => NativeData::F16(
                ind.map(|start| {
                    (0..iters)
                        .map(|i| a[start + i * iter_stride])
                        .max_by(|a, b| a.total_cmp(b))
                        .unwrap_or_default()
                })
                .collect(),
            ),
            NativeData::Bf16(a) => NativeData::Bf16(
                ind.map(|start| {
                    (0..iters)
                        .map(|i| a[start + i * iter_stride])
                        .max_by(|a, b| a.total_cmp(b))
                        .unwrap_or_default()
                })
                .collect(),
            ),
            NativeData::Int(a) => NativeData::Int(
                ind.map(|start| {
                    (0..iters)
                        .map(|i| a[start + i * iter_stride])
                        .max()
                        .unwrap_or_default()
                })
                .collect(),
            ),
            NativeData::Bool(_) => panic!("Cannot max-reduce Bool tensors"),
        }
    }
}

pub trait NativeOp: Debug + AsAny {
    fn execute(&self, inputs: Vec<&NativeData>, dyn_map: &FxHashMap<char, usize>) -> NativeData;
}

#[derive(Debug)]
pub enum NativeData {
    F32(Vec<f32>),
    F16(Vec<f16>),
    Bf16(Vec<bf16>),
    Int(Vec<i32>),
    Bool(Vec<bool>),
}

impl NativeData {
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
            if (**self.graph[node]).as_any().is::<Input>()
                || (**self.graph[node]).as_any().is::<Output>()
            {
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
        let data_id = self
            .graph
            .neighbors_directed(output_id, Direction::Incoming)
            .next()
            .unwrap();
        let NativeData::F32(f) = self.buffers.get(&data_id).unwrap() else {
            panic!()
        };
        f
    }
}

struct StridedIterator {
    shape: Vec<usize>,
    strides: Vec<usize>,
    index: Vec<usize>,
    done: bool,
}

impl StridedIterator {
    fn new(shape: &[Expression], strides: &[Expression], dyn_map: &FxHashMap<char, usize>) -> Self {
        let shape: Vec<usize> = shape.iter().map(|e| e.exec(dyn_map).unwrap()).collect();
        Self {
            index: vec![0; shape.len()],
            strides: strides
                .iter()
                .map(|e| e.exec(dyn_map).unwrap())
                .collect_vec(),
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
            .map(|(&s, &idx)| idx * s)
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
