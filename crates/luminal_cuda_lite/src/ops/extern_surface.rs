//! THE EGGLOG SURFACE OF A PURE-DATAFLOW EXTERN HOST OP, GENERATED.
//!
//! Every fused serving op follows one shape: `n` dense read operands, `p`
//! scalar metadata literals, one fresh dense output — a
//! [`luminal::graph::LogicalOp::Extern`] term the model spells and this
//! runtime implements. The six egglog snippets such an op contributes
//! (logical constructor, dtype rule, shape rule, forward layout mint,
//! implementation constructor, implementation match) differ only in
//! arity, the dtype rule and the shape rule, so this module writes them
//! from an [`ExternSurface`] description instead of a hand-copied `.egg`
//! per op. The hand-written surfaces of [`crate::ops::paged_attention`]
//! and [`crate::ops::moe_mxfp4`] are the templates the generator follows.
//!
//! [`extern_host_op!`] then supplies the struct boilerplate every such op
//! shares (the pure form, the DPS form, the matcher, the prototype); the
//! op module keeps only its spec, its kernel and its `execute`.

use std::sync::OnceLock;

use luminal::egglog_snippet::{EgglogSnippet, SpliceCategory};

/// A metadata literal's egglog sort.
#[derive(Debug, Clone, Copy)]
pub enum ParamKind {
    I64,
    F64,
}

impl ParamKind {
    fn sort(self) -> &'static str {
        match self {
            ParamKind::I64 => "i64",
            ParamKind::F64 => "f64",
        }
    }
}

/// How the output's dtype follows from the term.
#[derive(Debug, Clone, Copy)]
pub enum DtypeRule {
    /// The dtype of operand `i`.
    OfOperand(usize),
    /// A fixed dtype constructor, e.g. `"(F32)"` or `"(Int)"`.
    Fixed(&'static str),
    /// Chosen by the value of integer param `param`: one rule per case,
    /// `(value, dtype constructor)`.
    ByParam {
        param: usize,
        cases: &'static [(i64, &'static str)],
    },
}

/// One trailing extent of a [`ShapeRule::LeadingThen`] shape.
#[derive(Debug, Clone, Copy)]
pub enum TailDim {
    /// The value of integer param `i`.
    Param(usize),
    /// A literal.
    Lit(usize),
}

/// How the output's shape follows from the term.
#[derive(Debug, Clone, Copy)]
pub enum ShapeRule {
    /// The shape of operand `i`.
    OfOperand(usize),
    /// The leading extent of operand `operand`, then `tail`.
    LeadingThen {
        operand: usize,
        tail: &'static [TailDim],
    },
}

/// The description one op module writes.
pub struct ExternSurface {
    /// The logical (extern) constructor, e.g. `LogicalRmsNorm`.
    pub logical: &'static str,
    /// The implementation constructor, e.g. `LayoutTensorOpRmsNorm`.
    pub implementation: &'static str,
    /// Operand names, in order (documentation and slot names).
    pub operands: &'static [&'static str],
    /// Metadata literal names and sorts, in order.
    pub params: &'static [(&'static str, ParamKind)],
    pub dtype: DtypeRule,
    pub shape: ShapeRule,
    /// A leading comment for the logical constructor (what the op computes).
    pub doc: &'static str,
    /// Generated text, leaked once so the snippet type's `&'static str`
    /// contract holds. Initialize with `OnceLock::new()`.
    pub texts: OnceLock<Vec<&'static str>>,
    /// Likewise for the metadata slot table.
    pub slots: OnceLock<Vec<(&'static str, usize)>>,
}

impl ExternSurface {
    /// The term pattern `(Logical… ?a0 … ?p0 …)` with `?p{param}`
    /// replaced by `literal` when given.
    fn term(&self, literal: Option<(usize, i64)>) -> String {
        let mut parts = vec![self.logical.to_string()];
        for i in 0..self.operands.len() {
            parts.push(format!("?a{i}"));
        }
        for i in 0..self.params.len() {
            match literal {
                Some((p, value)) if p == i => parts.push(value.to_string()),
                _ => parts.push(format!("?p{i}")),
            }
        }
        format!("({})", parts.join(" "))
    }

    fn logical_constructor(&self) -> String {
        let mut sorts: Vec<&str> = vec!["LogicalTensor"; self.operands.len()];
        sorts.extend(self.params.iter().map(|(_, kind)| kind.sort()));
        let mut text = String::new();
        for line in self.doc.lines() {
            text.push_str("; ");
            text.push_str(line);
            text.push('\n');
        }
        text.push_str("; Operands: ");
        text.push_str(&self.operands.join(", "));
        text.push_str(". Metadata: ");
        text.push_str(
            &self
                .params
                .iter()
                .map(|(name, _)| *name)
                .collect::<Vec<_>>()
                .join(", "),
        );
        text.push_str(".\n");
        text.push_str(&format!(
            "(constructor {} ({}) LogicalTensor)\n",
            self.logical,
            sorts.join(" ")
        ));
        text
    }

    fn dtype_rule(&self) -> String {
        let one = |literal: Option<(usize, i64)>, action: &str| {
            format!(
                "; Dtype Propagation : {name}\n(rule\n  (\n    (= ?out {term})\n{extra}  )\n  (\n    (set (dtype-of ?out) {action})\n  )\n  :ruleset prop\n)\n",
                name = self.logical,
                term = self.term(literal),
                extra = match self.dtype {
                    DtypeRule::OfOperand(i) => format!("    (= ?dt (dtype-of ?a{i}))\n"),
                    _ => String::new(),
                },
            )
        };
        match self.dtype {
            DtypeRule::OfOperand(_) => one(None, "?dt"),
            DtypeRule::Fixed(ctor) => one(None, ctor),
            DtypeRule::ByParam { param, cases } => cases
                .iter()
                .map(|(value, ctor)| one(Some((param, *value)), ctor))
                .collect(),
        }
    }

    fn shape_rule(&self) -> String {
        let (premise, action) = match self.shape {
            ShapeRule::OfOperand(i) => (
                format!("    (= ?shape (shape-of ?a{i}))\n"),
                "?shape".to_string(),
            ),
            ShapeRule::LeadingThen { operand, tail } => {
                let mut list = "(IntExprNil)".to_string();
                for dim in tail.iter().rev() {
                    let extent = match dim {
                        TailDim::Param(i) => format!("(IntLit ?p{i})"),
                        TailDim::Lit(n) => format!("(IntLit {n})"),
                    };
                    list = format!("(IntExprCons {extent} {list})");
                }
                (
                    format!(
                        "    (= (shape-of ?a{operand}) (ShapeLit (IntExprCons ?lead ?rest)))\n"
                    ),
                    format!("(ShapeLit (IntExprCons ?lead {list}))"),
                )
            }
        };
        format!(
            "; Shape Propagation : {name}\n(rule\n  (\n    (= ?out {term})\n{premise}  )\n  (\n    (set (shape-of ?out) {action})\n  )\n  :ruleset prop\n)\n",
            name = self.logical,
            term = self.term(None),
        )
    }

    fn forward_rule(&self) -> String {
        format!(
            "; Forward LayoutTensor Propagation : {name} — a fresh materialized\n; result in the canonical right-major layout.\n(rule\n  (\n    (= ?out {term})\n    (= ?a0_layout_tensor (LayoutTensorLit ?a0 ?a0_layout))\n    (= ?bits (bits-of (dtype-of ?out)))\n    (= ?shape (shape-of ?out))\n  )\n  (\n    (let out_layout_tensor (LayoutTensorLit ?out (RightMajorContiguousElementLayoutLit ?shape ?bits)))\n  )\n)\n",
            name = self.logical,
            term = self.term(None),
        )
    }

    fn implementation_constructor(&self) -> String {
        let mut sorts: Vec<&str> = vec!["LayoutTensor"; self.operands.len()];
        sorts.extend(self.params.iter().map(|(_, kind)| kind.sort()));
        sorts.push("Layout");
        format!(
            "; {name} host op: {n} dense read operands, {p} metadata literals, one\n; fresh written output.\n(constructor {name} ({sorts}) LayoutTensorOp)\n",
            name = self.implementation,
            n = self.operands.len(),
            p = self.params.len(),
            sorts = sorts.join(" "),
        )
    }

    fn match_rule(&self) -> String {
        let mut premises = format!("    (= ?out_logical {})\n", self.term(None));
        for i in 0..self.operands.len() {
            premises.push_str(&format!(
                "    (= ?a{i}_lt (LayoutTensorLit ?a{i} ?a{i}_layout))\n    (= ?a{i}_layout (RightMajorContiguousElementLayoutLit ?a{i}_shape ?a{i}_bits))\n"
            ));
        }
        premises.push_str(
            "    (= ?out_lt (LayoutTensorLit ?out_logical ?out_layout))\n    (= ?out_layout (RightMajorContiguousElementLayoutLit ?out_shape ?out_bits))\n    (= (injectivity-of ?out_lt) (Injective))\n",
        );
        let mut reads = "(LayoutTensorNil)".to_string();
        for i in (0..self.operands.len()).rev() {
            reads = format!("(LayoutTensorCons ?a{i}_lt {reads})");
        }
        let mut args: Vec<String> = (0..self.operands.len())
            .map(|i| format!("?a{i}_lt"))
            .collect();
        args.extend((0..self.params.len()).map(|i| format!("?p{i}")));
        args.push("?out_layout".to_string());
        format!(
            "; Layout Tensor Op {name} — every operand DENSE (right-major\n; contiguous) and a dense output; a view-shaped operand reaches the op\n; through the materializing copy the e-graph mints for it.\n(rule\n  (\n{premises}  )\n  (\n    (let ?layout_tensor_op\n      (LayoutTensorOpLit\n        {reads}\n        (LayoutTensorCons ?out_lt (LayoutTensorNil))))\n    (union ?layout_tensor_op\n      ({name} {args}))\n  )\n)\n",
            name = self.implementation,
            args = args.join(" "),
        )
    }

    /// The six snippets, generated once.
    pub fn snippets(&'static self) -> Vec<EgglogSnippet> {
        let texts = self.texts.get_or_init(|| {
            [
                self.logical_constructor(),
                self.dtype_rule(),
                self.shape_rule(),
                self.forward_rule(),
                self.implementation_constructor(),
                self.match_rule(),
            ]
            .into_iter()
            .map(|text| &*Box::leak(text.into_boxed_str()))
            .collect()
        });
        let categories = [
            SpliceCategory::LogicalConstructors,
            SpliceCategory::Dtype,
            SpliceCategory::Shape,
            SpliceCategory::Forward,
            SpliceCategory::LayoutOpConstructors,
            SpliceCategory::Match,
        ];
        categories
            .into_iter()
            .zip(texts.iter())
            .map(|(category, text)| EgglogSnippet { category, text })
            .collect()
    }

    /// The matcher's metadata slot table: params at children
    /// `operands.len()..`, then `out_layout`.
    pub fn metadata_slots(&'static self) -> &'static [(&'static str, usize)] {
        self.slots.get_or_init(|| {
            let mut slots: Vec<(&'static str, usize)> = self
                .params
                .iter()
                .enumerate()
                .map(|(i, (name, _))| (*name, self.operands.len() + i))
                .collect();
            slots.push(("out_layout", self.operands.len() + self.params.len()));
            slots
        })
    }

    /// The child index of param `i` at an extraction site.
    pub fn param_child(&self, i: usize) -> usize {
        self.operands.len() + i
    }

    /// The operand's slot name (`dest0` for the DPS destination).
    pub fn operand_name(&self, dest: usize, operand: usize) -> String {
        if operand == dest {
            "dest0".to_string()
        } else {
            self.operands
                .get(operand)
                .map(|name| name.to_string())
                .unwrap_or_else(|| format!("in{operand}"))
        }
    }
}

/// The struct boilerplate of a pure-dataflow extern host op: the pure
/// form `$op`, the destination-passing form `$dps` (which the module
/// implements [`crate::host::HostOp`] for), the matcher `$matcher`, and
/// `prototype()`. `$extract` maps an extraction site to the spec.
#[macro_export]
macro_rules! extern_host_op {
    (
        surface: $surface:expr,
        label: $label:literal,
        spec: $spec:ty,
        op: $op:ident,
        dps: $dps:ident,
        matcher: $matcher:ident,
        prototype: $prototype:expr,
        extract: $extract:expr $(,)?
    ) => {
        #[doc = concat!("`", $label, "` — pure dataflow form.")]
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub struct $op {
            pub spec: $spec,
        }

        impl luminal::buffer_tensor_ir::OpSlotNames for $op {
            fn operand_name(&self, operand: usize) -> String {
                $surface.operand_name(usize::MAX, operand)
            }
        }

        impl luminal::buffer_tensor_ir::BufferTensorIrOp for $op {
            fn label(&self) -> &str {
                $label
            }
        }

        impl luminal::layout_ir::Bufferizable for $op {}

        impl luminal::layout_ir::ToDps for $op {
            fn to_dps(&self) -> Option<Box<dyn luminal::layout_ir::LayoutIrOp>> {
                Some(Box::new($dps { spec: self.spec }))
            }
        }

        impl luminal::layout_ir::LayoutIrOp for $op {}

        #[doc = concat!("`", $label, "` — destination-passing form: the reads, then `dest0`.")]
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub struct $dps {
            pub spec: $spec,
        }

        impl $dps {
            /// The destination operand's index (after every read operand).
            pub fn dest() -> usize {
                $surface.operands.len()
            }
        }

        impl luminal::buffer_tensor_ir::OpSlotNames for $dps {
            fn operand_name(&self, operand: usize) -> String {
                $surface.operand_name(Self::dest(), operand)
            }
        }

        impl luminal::buffer_tensor_ir::BufferTensorIrOp for $dps {
            fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
                Some($crate::CudaOpInterface::host::<Self>())
            }

            fn label(&self) -> &str {
                $label
            }

            fn operand_reads_memory(&self, operand: usize) -> bool {
                operand != Self::dest()
            }
        }

        impl luminal::layout_ir::Bufferizable for $dps {
            fn alias_info(&self) -> Vec<luminal::layout_ir::AliasInfo> {
                vec![luminal::layout_ir::AliasInfo {
                    operand: Self::dest(),
                    result: 0,
                    sharing: luminal::layout_ir::Sharing::Must,
                }]
            }
        }

        impl luminal::layout_ir::ToDps for $dps {
            fn to_dps(&self) -> Option<Box<dyn luminal::layout_ir::LayoutIrOp>> {
                None
            }
        }

        impl luminal::layout_ir::LayoutIrOp for $dps {}

        #[doc = concat!("Matches the `", $label, "` implementation constructor.")]
        #[derive(Debug, Clone, Copy, Default)]
        pub struct $matcher;

        impl luminal::layout_ir::OpMatcher for $matcher {
            fn egglog_constructor(&self) -> &'static str {
                $surface.implementation
            }

            fn snippets(&self) -> Vec<luminal::egglog_snippet::EgglogSnippet> {
                $surface.snippets()
            }

            fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
                $surface.metadata_slots()
            }

            fn extract(
                &self,
                site: &luminal::layout_ir::ExtractionSite<'_>,
            ) -> Box<dyn luminal::layout_ir::LayoutIrOp> {
                let extract: fn(&luminal::layout_ir::ExtractionSite<'_>) -> $spec = $extract;
                Box::new($op {
                    spec: extract(site),
                })
            }
        }

        /// The registry prototype.
        pub fn prototype() -> $op {
            $op { spec: $prototype }
        }
    };
}
