//! Element coefficients, with freedom only on statically proved singleton axes.
//!
//! `element-stride-readings-of(Layout, axis, stride)` retains independently proved
//! exact readings as relation facts. A static singleton additionally proves Any;
//! this never unifies global coefficient expressions or discards older proofs.
//! Contextual list normalization keeps its per-witness merge cell, avoiding the
//! need to choose between unrelated exact readings for one layout/axis pair.
//!
//! Separately scheduled expression discovery uses whole linear certificates:
//! a shape-owned expression, a cons list of BigInt coefficients, and an origin.
//! Egglog performs list arithmetic, composes certificates, and publishes only
//! nonnegative zero-origin layouts. No container sorts or Rust list walkers are
//! needed for this analysis, and no runtime schedule runs discovery.
use crate::egglog_snippet::{EgglogSnippet, SpliceCategory};
use crate::egglog_utils::eclass::{EClass, ENode, EgglogConstructor};
use crate::layouts::{self, BitWidthTerm, IntExprTerm, Layout, LayoutFacts, ShapeTerm, SpanExpr};
use anyhow::{Result, anyhow, ensure};
use egglog::{
    ArcSort, AtomTerm, Core, EGraph, Primitive, Read, ReadPrim, ReadState, TypeInfo, Value, Write,
    WritePrim, WriteState,
    api::RawValues,
    ast::Span,
    constraint::{Constraint, SimpleTypeConstraint, TypeConstraint},
};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ElementStride {
    Any,
    Exact(IntExprTerm),
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ElementStridedLayout {
    pub shape: ShapeTerm,
    pub strides: Vec<ElementStride>,
    pub width: BitWidthTerm,
}

impl ElementStridedLayout {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.shape.0.len() == self.strides.len(),
            "stride rank differs from shape rank"
        );
        for (extent, stride) in self.shape.0.iter().zip(&self.strides) {
            match stride {
                ElementStride::Any => ensure!(
                    extent.eval_literal() == Some(1),
                    "Any requires a static singleton"
                ),
                ElementStride::Exact(e) => {
                    ensure!(
                        coordinate_free(e),
                        "an element coefficient contains coordinates"
                    );
                    if let Some(value) = e.eval_literal() {
                        ensure!(value >= 0, "negative strides require an explicit origin");
                    }
                }
            }
        }
        Ok(())
    }
}

fn coordinate_free(e: &IntExprTerm) -> bool {
    use IntExprTerm::*;
    match e {
        Coord { .. } => false,
        Lit(_) | Var(_) => true,
        Add(a, b)
        | Mul(a, b)
        | TruncDiv(a, b)
        | TruncRem(a, b)
        | CeilDiv(a, b)
        | Min(a, b)
        | Max(a, b)
        | LessThanCast(a, b) => coordinate_free(a) && coordinate_free(b),
    }
}

impl SpanExpr for ElementStridedLayout {
    fn span(&self) -> IntExprTerm {
        let mut span = IntExprTerm::Lit(1);
        for (extent, stride) in self.shape.0.iter().zip(&self.strides) {
            if let ElementStride::Exact(s) = stride {
                span = IntExprTerm::Add(
                    Box::new(span),
                    Box::new(IntExprTerm::Mul(
                        Box::new(IntExprTerm::Add(
                            Box::new(extent.clone()),
                            Box::new(IntExprTerm::Lit(-1)),
                        )),
                        Box::new(s.clone()),
                    )),
                );
            }
        }
        layouts::zero_for_empty_domain(&self.shape, span)
    }
}

impl LayoutFacts for ElementStridedLayout {
    fn shape(&self) -> &ShapeTerm {
        &self.shape
    }
    fn width(&self) -> BitWidthTerm {
        self.width
    }
    fn span_elements(&self) -> Option<IntExprTerm> {
        Some(self.span())
    }
    fn element_index(&self, coords: &[usize]) -> Result<usize> {
        self.validate()?;
        ensure!(
            coords.len() == self.strides.len(),
            "coordinate rank differs from layout rank"
        );
        let mut address = 0i64;
        for ((extent, stride), &coordinate) in self.shape.0.iter().zip(&self.strides).zip(coords) {
            let extent = extent
                .eval_literal()
                .ok_or_else(|| anyhow!("symbolic domain"))?;
            ensure!(
                coordinate < usize::try_from(extent)?,
                "coordinate out of domain"
            );
            if let ElementStride::Exact(s) = stride {
                let s = s
                    .eval_literal()
                    .ok_or_else(|| anyhow!("symbolic coefficient"))?;
                address = address
                    .checked_add(
                        i64::try_from(coordinate)?
                            .checked_mul(s)
                            .ok_or_else(|| anyhow!("address overflow"))?,
                    )
                    .ok_or_else(|| anyhow!("address overflow"))?;
            }
        }
        Ok(usize::try_from(address)?)
    }
}

fn decode_strides(class: &EClass<'_>, depth: usize) -> Option<Vec<ElementStride>> {
    if depth == 0 {
        return None;
    }
    if class.nodes_named("ElementStrideNil").next().is_some() {
        return Some(vec![]);
    }
    for node in class.nodes_named("ElementStrideCons") {
        let head = node.child(0)?;
        let stride = if head.nodes_named("ElementStrideAny").next().is_some() {
            Some(ElementStride::Any)
        } else {
            head.nodes_named("ElementStrideExact")
                .find_map(|n| layouts::int_expr(&n.child(0)?, None).map(ElementStride::Exact))
        };
        if let (Some(stride), Some(mut tail)) = (stride, decode_strides(&node.child(1)?, depth - 1))
        {
            tail.insert(0, stride);
            return Some(tail);
        }
    }
    None
}

impl EgglogConstructor for ElementStridedLayout {
    const NAME: &'static str = "ElementStridedLayoutLit";
    type Sort = Layout;
    fn decode(node: &ENode<'_>) -> Result<Self> {
        let layout = Self {
            shape: layouts::shape_term(&node.child_or_bail(0)?)
                .ok_or_else(|| anyhow!("invalid shape"))?,
            strides: decode_strides(&node.child_or_bail(1)?, 64)
                .ok_or_else(|| anyhow!("invalid stride list"))?,
            width: layouts::bit_width(&node.child_or_bail(2)?)
                .ok_or_else(|| anyhow!("invalid bit width"))?,
        };
        layout.validate()?;
        Ok(layout)
    }
    fn erase(self) -> Arc<dyn LayoutFacts> {
        Arc::new(self)
    }
}

struct Signature {
    name: &'static str,
    arity: usize,
    result: &'static str,
    span: Span,
}
impl TypeConstraint for Signature {
    fn get(
        &self,
        args: &[AtomTerm],
        info: &TypeInfo,
    ) -> Vec<Box<dyn Constraint<AtomTerm, ArcSort>>> {
        let mut sorts = vec![info.get_sort_by_name("ElementStride").unwrap().clone(); self.arity];
        sorts.push(info.get_sort_by_name(self.result).unwrap().clone());
        SimpleTypeConstraint::new(self.name, sorts, self.span.clone()).get(args, info)
    }
}

#[derive(Clone)]
struct Compatible;
impl Primitive for Compatible {
    fn name(&self) -> &str {
        "element-stride-compatible"
    }
    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        Box::new(Signature {
            name: "element-stride-compatible",
            arity: 2,
            result: "bool",
            span: span.clone(),
        })
    }
}
impl ReadPrim for Compatible {
    fn apply<'a, 'db>(&self, state: ReadState<'a, 'db>, args: &[Value]) -> Option<Value> {
        let [a, b] = args else {
            return None;
        };
        let any = state
            .eclass_of("ElementStrideAny", RawValues(vec![]))
            .expect("ElementStrideAny must be declared before comparison");
        Some(
            state
                .base_values()
                .get::<bool>(a == b || any == Some(*a) || any == Some(*b)),
        )
    }
}

#[derive(Clone)]
struct PreferAny;
impl Primitive for PreferAny {
    fn name(&self) -> &str {
        "element-stride-prefer-any"
    }
    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        Box::new(Signature {
            name: "element-stride-prefer-any",
            arity: 2,
            result: "ElementStride",
            span: span.clone(),
        })
    }
}
impl WritePrim for PreferAny {
    fn apply<'a, 'db>(&self, mut state: WriteState<'a, 'db>, args: &[Value]) -> Option<Value> {
        let [a, b] = args else {
            return None;
        };
        let any = state.add("ElementStrideAny", RawValues(vec![])).ok()?;
        // Only contextual normalization writes this cell: its exact coefficient
        // never changes. An arbitrary live-axis exact conflict is NOT freedom.
        if a == b {
            Some(*a)
        } else if *a == any || *b == any {
            Some(any)
        } else {
            None
        }
    }
}

pub(crate) fn register(egraph: &mut EGraph) {
    egraph.add_read_primitive(Compatible, None);
    egraph.add_write_primitive(PreferAny, None);
}

pub(crate) fn snippets() -> Vec<EgglogSnippet> {
    vec![
        EgglogSnippet {
            category: SpliceCategory::LogicalConstructors,
            text: include_str!("element_stride_constructors.egg"),
        },
        EgglogSnippet {
            category: SpliceCategory::Forward,
            text: include_str!("element_stride_rules.egg"),
        },
        EgglogSnippet {
            category: SpliceCategory::Forward,
            text: include_str!("element_stride_discovery.egg"),
        },
    ]
}

#[cfg(test)]
#[path = "element_stride_tests.rs"]
mod tests;
