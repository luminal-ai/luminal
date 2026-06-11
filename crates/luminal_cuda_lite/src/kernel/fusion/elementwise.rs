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
    FxHashMap<char, CudaSlice<u8>>,
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
            rules.push(Rule::raw(format!(
                "(rule (
                    (= ?u (Op ({hlir} ?shape ?s ?out_s) (ICons ?x (INil))))
                    (= ?dt (dtype ?u))
                 ) (
                    (let ?fs (Op (FusionStart ?shape ?s ?dt) (ICons ?x (INil))))
                    (let ?elem (Op (CudaUnaryElementwise \"{opcode}\" ?shape ?s ?out_s ?dt)
                                   (ICons ?fs (INil))))
                    (let ?fe (Op (FusionEnd ?shape ?out_s ?dt) (ICons ?elem (INil))))
                    (union ?u ?fe)
                    (set (dtype ?fe) ?dt)
                 ) :ruleset kernel_lower :name \"cuda-elem-singleton-{hlir}\")"
            )));
        }

        rules.push(Rule::raw(
            "(rule (
                    (= ?sqrt (Op (Sqrt ?shape ?x_stride ?sqrt_stride) (ICons ?x (INil))))
                    (= ?recip (Op (Recip ?shape ?sqrt_stride ?out_stride) (ICons ?sqrt (INil))))
                    (= ?dt (dtype ?recip))
                 ) (
                    (let ?fs (Op (FusionStart ?shape ?x_stride ?dt) (ICons ?x (INil))))
                    (let ?elem (Op (CudaUnaryElementwise \"Rsqrt\" ?shape ?x_stride ?out_stride ?dt)
                                   (ICons ?fs (INil))))
                    (let ?fe (Op (FusionEnd ?shape ?out_stride ?dt) (ICons ?elem (INil))))
                    (union ?recip ?fe)
                    (set (dtype ?fe) ?dt)
                 ) :ruleset kernel_lower :name \"cuda-elem-rsqrt-from-sqrt-recip\")",
        ));

        rules.push(Rule::raw(
            "(rule
                (
                    (= ?mul (Op (Mul ?shape ?x_stride ?const_stride ?inter_stride) (ICons ?x (ICons ?exp_const (INil)))))
                    (= ?exp2 (Op (Exp2 ?shape ?inter_stride ?out_stride) (ICons ?mul (INil))))
                    (= ?dt (dtype ?x))
                    (= ?cv (Op (Constant ?val) (INil)))
                    (= ?exp_const ?cv)
                    (> ?val 1.44)
                    (< ?val 1.45)
                )
                (
                    (let ?fs (Op (FusionStart ?shape ?x_stride ?dt) (ICons ?x (INil))))
                    (let ?elem (Op (CudaUnaryElementwise \"Exp\" ?shape ?x_stride ?out_stride ?dt)
                                   (ICons ?fs (INil))))
                    (let ?fe (Op (FusionEnd ?shape ?out_stride ?dt) (ICons ?elem (INil))))
                    (union ?exp2 ?fe)
                    (set (dtype ?fe) ?dt)
                )
                :ruleset direct_kernel
                :name \"direct-exp-region\"
            )",
        ));

        rules.push(Rule::raw(
            "(datatype*
                (CudaSigmoidScaledState
                    (MkCudaSigmoidScaledState IR EList EList DType)
                )
            )
            (function cuda_sigmoid_scaled (IR) CudaSigmoidScaledState :merge new)

            (rule
            (
                (= ?neg1 (Op (Constant ?nv) (INil)))
                (< ?nv -0.99)
                (> ?nv -1.01)
                (= ?neg_x (Op (Mul ?shape ?x_stride ?neg_stride ?neg_out_stride) (ICons ?x (ICons ?neg1 (INil)))))
                (= ?log2e (Op (Constant ?lv) (INil)))
                (> ?lv 1.44)
                (< ?lv 1.45)
                (= ?scaled (Op (Mul ?shape ?neg_out_stride ?log2e_stride ?scaled_stride) (ICons ?neg_x (ICons ?log2e (INil)))))
                (= ?dt (dtype ?x))
            )
            (
                (set (cuda_sigmoid_scaled ?scaled)
                    (MkCudaSigmoidScaledState ?x ?shape ?x_stride ?dt))
            )
            :ruleset direct_kernel
            :name \"direct-sigmoid-scaled-region-marker\"
            )

            (rule
            (
                (= ?scaled_state (cuda_sigmoid_scaled ?scaled))
                (= ?scaled_state (MkCudaSigmoidScaledState ?x ?shape ?x_stride ?dt))
                (= ?exp2 (Op (Exp2 ?shape ?scaled_stride ?exp_stride) (ICons ?scaled (INil))))
                (= ?one (Op (Constant ?ov) (INil)))
                (> ?ov 0.99)
                (< ?ov 1.01)
                (= ?plus_one (Op (Add ?shape ?exp_stride ?one_stride ?add_stride) (ICons ?exp2 (ICons ?one (INil)))))
                (= ?sig_out (Op (Recip ?shape ?add_stride ?out_stride) (ICons ?plus_one (INil))))
            )
            (
                (let ?fs (Op (FusionStart ?shape ?x_stride ?dt) (ICons ?x (INil))))
                (let ?elem (Op (CudaUnaryElementwise \"Sigmoid\" ?shape ?x_stride ?out_stride ?dt)
                               (ICons ?fs (INil))))
                (let ?fe (Op (FusionEnd ?shape ?out_stride ?dt) (ICons ?elem (INil))))
                (union ?sig_out ?fe)
                (set (dtype ?fe) ?dt)
            )
            :ruleset direct_kernel
            :name \"direct-sigmoid-region\"
            )",
        ));

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
                "(rule (
                    (= ?bin (Op (Add ?shape ?a_s ?b_s ?out_s) (ICons ?a (ICons ?b (INil)))))
                    (= ?dt (dtype ?bin))
                 ) (
                    (let ?fs_a (Op (FusionStart ?shape ?a_s ?dt) (ICons ?a (INil))))
                    (let ?fs_b (Op (FusionStart ?shape ?b_s ?dt) (ICons ?b (INil))))
                    (let ?elem (Op (CudaBinaryElementwise \"Add\" ?shape ?a_s ?b_s ?out_s ?dt)
                                   (ICons ?fs_a (ICons ?fs_b (INil)))))
                    (let ?fe (Op (FusionEnd ?shape ?out_s ?dt) (ICons ?elem (INil))))
                    (union ?bin ?fe)
                    (set (dtype ?fe) ?dt)
                 ) :ruleset kernel_lower :name \"cuda-elem-singleton-Add\")",
            ),
            Rule::raw(
                "(rule (
                    (= ?bin (Op (Mul ?shape ?a_s ?b_s ?out_s) (ICons ?a (ICons ?b (INil)))))
                    (= ?dt (dtype ?a))
                 ) (
                    (let ?fs_a (Op (FusionStart ?shape ?a_s ?dt) (ICons ?a (INil))))
                    (let ?fs_b (Op (FusionStart ?shape ?b_s ?dt) (ICons ?b (INil))))
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
        let mut out_shape =
            extract_expr_list(egraph, kind_children[1], list_cache, expr_cache).unwrap();
        let mut a_stride =
            extract_expr_list(egraph, kind_children[2], list_cache, expr_cache).unwrap();
        let mut b_stride =
            extract_expr_list(egraph, kind_children[3], list_cache, expr_cache).unwrap();
        let mut out_stride =
            extract_expr_list(egraph, kind_children[4], list_cache, expr_cache).unwrap();
        let n = out_shape
            .len()
            .min(a_stride.len())
            .min(b_stride.len())
            .min(out_stride.len());
        out_shape.truncate(n);
        a_stride.truncate(n);
        b_stride.truncate(n);
        out_stride.truncate(n);
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
