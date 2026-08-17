// =========================================================================
// Generic CUDA elementwise ops used inside FusionStart/FusionEnd regions.
//
// CUDA elementwise execution is represented as a FusionEnd-rooted region even
// for a single op. These ops are therefore region-internal only; standalone
// compilation is intentionally unsupported.
// =========================================================================

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{DTYPE, ELIST, OP_KIND, STRING},
        extract_dtype, extract_expr_list,
    },
    op::*,
    prelude::*,
};

use crate::kernel::KernelOp;

pub type Ops = (CudaUnaryElementwise, CudaBinaryElementwise);

type CompileOut = (
    CudaFunction,
    Arc<CudaModule>,
    String,
    (Expression, Expression, Expression),
    (Expression, Expression, Expression),
    Expression,
    FxHashMap<Symbol, CudaSlice<u8>>,
);

fn extract_string_label(egraph: &SerializedEGraph, node: &ENodeId) -> String {
    egraph.enodes[node].0.trim_matches('"').to_string()
}

#[derive(Default, Debug, Clone)]
pub struct CudaUnaryElementwise {
    pub(crate) op: String,
    pub(crate) shape: Vec<Expression>,
    pub(crate) in_strides: Vec<Expression>,
    pub(crate) out_strides: Vec<Expression>,
    pub(crate) dtype: DType,
}

impl EgglogOp for CudaUnaryElementwise {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "CudaUnaryElementwise",
            &[
                ("op", STRING),
                ("shape", ELIST),
                ("strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        1
    }

    fn rewrites(&self) -> Vec<Rule> {
        let mut rules = Vec::new();
        for (hlir, opcode) in [
            ("Sin", "Sin"),
            ("Sqrt", "Sqrt"),
            ("Exp2", "Exp2"),
            ("Log2", "Log2"),
            ("Recip", "Recip"),
        ] {
            // Each FusionStart is stamped with ITS input's dtype — it
            // describes how the external producer's buffer is loaded, and the
            // codegen reads every local at its producer's dtype. Stamping the
            // consumer's dtype instead would violate the construction
            // invariant whenever input and output dtypes differ.
            rules.push(Rule::raw(format!(
                "(rule (
                    (= ?u (Op ({hlir} ?shape ?s ?out_s) (ICons ?x (INil))))
                    (= ?dt (dtype ?u))
                    (= ?dt_x (dtype ?x))
                 ) (
                    (let ?fs (Op (FusionStart ?shape ?s ?dt_x) (ICons ?x (INil))))
                    (let ?elem (Op (CudaUnaryElementwise \"{opcode}\" ?shape ?s ?out_s ?dt)
                                   (ICons ?fs (INil))))
                    (let ?fe (Op (FusionEnd ?shape ?out_s ?dt) (ICons ?elem (INil))))
                    (union ?u ?fe)
                    (set (dtype ?fe) ?dt)
                 ) :ruleset kernel_lower :name \"cuda-elem-singleton-{hlir}\")"
            )));
        }

        // The rsqrt / direct-exp / direct-sigmoid substitutions are removed:
        // they replaced exact HLIR chains with approximate single
        // instructions (`rsqrtf`, `expf`, fused sigmoid) matched through
        // constant tolerance windows — V2 violations under the exact-HLIR
        // reproduction ruling. See the exact-HLIR reproduction ruling.

        rules
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
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                op: extract_string_label(egraph, kind_children[0]),
                shape: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache).unwrap(),
                in_strides: extract_expr_list(egraph, kind_children[2], list_cache, expr_cache)
                    .unwrap(),
                out_strides: extract_expr_list(egraph, kind_children[3], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, kind_children[4]),
            })),
            input_enodes,
        )
    }
}

impl KernelOp for CudaUnaryElementwise {
    fn compile(
        &self,
        _stream: &Arc<CudaStream>,
        _compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> CompileOut {
        unreachable!("CudaUnaryElementwise must be compiled through fusion region codegen")
    }

    fn output_size(&self) -> Expression {
        self.shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        self.output_bytes()
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.output_size()
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "CudaUnaryElementwise"
    }
}

#[derive(Default, Debug, Clone)]
pub struct CudaBinaryElementwise {
    pub(crate) op: String,
    pub(crate) out_shape: Vec<Expression>,
    pub(crate) a_stride: Vec<Expression>,
    pub(crate) b_stride: Vec<Expression>,
    pub(crate) out_stride: Vec<Expression>,
    pub(crate) dtype: DType,
}

impl EgglogOp for CudaBinaryElementwise {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "CudaBinaryElementwise",
            &[
                ("op", STRING),
                ("shape", ELIST),
                ("a_strides", ELIST),
                ("b_strides", ELIST),
                ("out_strides", ELIST),
                ("dtype", DTYPE),
            ],
        )
    }

    fn n_inputs(&self) -> usize {
        2
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![
            Rule::raw(
                // FusionStart dtypes follow each input's producer; see the
                // unary rules for why anything else is construction-illegal.
                "(rule (
                    (= ?bin (Op (Add ?shape ?a_s ?b_s ?out_s) (ICons ?a (ICons ?b (INil)))))
                    (= ?dt (dtype ?bin))
                    (= ?dt_a (dtype ?a))
                    (= ?dt_b (dtype ?b))
                 ) (
                    (let ?fs_a (Op (FusionStart ?shape ?a_s ?dt_a) (ICons ?a (INil))))
                    (let ?fs_b (Op (FusionStart ?shape ?b_s ?dt_b) (ICons ?b (INil))))
                    (let ?elem (Op (CudaBinaryElementwise \"Add\" ?shape ?a_s ?b_s ?out_s ?dt)
                                   (ICons ?fs_a (ICons ?fs_b (INil)))))
                    (let ?fe (Op (FusionEnd ?shape ?out_s ?dt) (ICons ?elem (INil))))
                    (union ?bin ?fe)
                    (set (dtype ?fe) ?dt)
                 ) :ruleset kernel_lower :name \"cuda-elem-singleton-Add\")",
            ),
            Rule::raw(
                // The op and FusionEnd carry the RESULT dtype (?bin, not ?a —
                // Mul(bf16, f32) stamped bf16 would round the product); the
                // FusionStarts carry their producers' dtypes as above.
                "(rule (
                    (= ?bin (Op (Mul ?shape ?a_s ?b_s ?out_s) (ICons ?a (ICons ?b (INil)))))
                    (= ?dt (dtype ?bin))
                    (= ?dt_a (dtype ?a))
                    (= ?dt_b (dtype ?b))
                 ) (
                    (let ?fs_a (Op (FusionStart ?shape ?a_s ?dt_a) (ICons ?a (INil))))
                    (let ?fs_b (Op (FusionStart ?shape ?b_s ?dt_b) (ICons ?b (INil))))
                    (let ?elem (Op (CudaBinaryElementwise \"Mul\" ?shape ?a_s ?b_s ?out_s ?dt)
                                   (ICons ?fs_a (ICons ?fs_b (INil)))))
                    (let ?fe (Op (FusionEnd ?shape ?out_s ?dt) (ICons ?elem (INil))))
                    (union ?bin ?fe)
                    (set (dtype ?fe) ?dt)
                 ) :ruleset kernel_lower :name \"cuda-elem-singleton-Mul\")",
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
        // Preserve every extracted metadata list verbatim. Singleton fusion
        // rules derive these lists from one matched HLIR operation, so their
        // rank/layout contract is established by construction; extraction
        // must not mutate that metadata.
        let out_shape =
            extract_expr_list(egraph, kind_children[1], list_cache, expr_cache).unwrap();
        let a_stride = extract_expr_list(egraph, kind_children[2], list_cache, expr_cache).unwrap();
        let b_stride = extract_expr_list(egraph, kind_children[3], list_cache, expr_cache).unwrap();
        let out_stride =
            extract_expr_list(egraph, kind_children[4], list_cache, expr_cache).unwrap();
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                op: extract_string_label(egraph, kind_children[0]),
                out_shape,
                a_stride,
                b_stride,
                out_stride,
                dtype: extract_dtype(egraph, kind_children[5]),
            })),
            input_enodes,
        )
    }
}

impl KernelOp for CudaBinaryElementwise {
    fn compile(
        &self,
        _stream: &Arc<CudaStream>,
        _compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> CompileOut {
        unreachable!("CudaBinaryElementwise must be compiled through fusion region codegen")
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        self.output_bytes() * 2
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.output_size()
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "CudaBinaryElementwise"
    }
}
