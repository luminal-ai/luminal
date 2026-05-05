// =========================================================================
// Fusion boundary markers — FusionStart and FusionEnd.
//
// Tag-like LLIR ops that bracket a region of elementwise ops destined to
// be emitted as a single CUDA kernel:
//   - N FusionStart nodes per region (one per FS leaf — distinct external
//     reads),
//   - exactly 1 FusionEnd per region.
//
// `FusionEnd::rewrites()` carries the seven rule families that build and
// extend regions (pair-fuse / grow / merge); the actual single-kernel
// codegen lives in `region_codegen`. Like FusedX, both markers'
// `compile()` is `unreachable!()` — region codegen folds them away
// before kernel_to_host's compile loop reaches an interior node.
// =========================================================================

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{DTYPE, ELIST, OP_KIND},
        extract_dtype, extract_expr_list,
    },
    op::*,
    prelude::*,
};

use crate::kernel::KernelOp;

pub type Ops = (FusionStart, FusionEnd);

type CompileOut = (
    CudaFunction,
    Arc<CudaModule>,
    String,
    (Expression, Expression, Expression),
    (Expression, Expression, Expression),
    Expression,
    FxHashMap<char, CudaSlice<u8>>,
);

// =========================================================================
// FusionStart
// =========================================================================

#[derive(Default, Debug, Clone)]
pub struct FusionStart {
    pub(crate) shape: Vec<Expression>,
    pub(crate) strides: Vec<Expression>,
    pub(crate) dtype: DType,
}

impl EgglogOp for FusionStart {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "FusionStart",
            &[("shape", ELIST), ("strides", ELIST), ("dtype", DTYPE)],
        )
    }
    fn n_inputs(&self) -> usize {
        1
    }
    fn rewrites(&self) -> Vec<Rule> {
        // No idempotence rule. `FusionStart(FusionStart(x)) ≡ FusionStart(x)`
        // would unify nested markers and create eclass cycles via the
        // pair-fuse rules; without it, occasional re-firings produce extra
        // semantically-correct identity layers, bounded by the run schedule.
        Vec::new()
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
                shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap(),
                strides: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, kind_children[2]),
            })),
            input_enodes,
        )
    }
}

impl KernelOp for FusionStart {
    fn compile(
        &self,
        _stream: &Arc<CudaStream>,
        _compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> CompileOut {
        unreachable!("FusionStart must be compiled through fusion region codegen")
    }
    fn output_size(&self) -> Expression {
        self.shape.iter().copied().product()
    }
    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }
    fn output_dtype(&self) -> DType {
        self.dtype
    }
    fn kernel_name(&self) -> &'static str {
        "FusionStart"
    }
}

// =========================================================================
// FusionEnd
// =========================================================================

#[derive(Default, Debug, Clone)]
pub struct FusionEnd {
    pub(crate) shape: Vec<Expression>,
    pub(crate) strides: Vec<Expression>,
    pub(crate) dtype: DType,
}

impl EgglogOp for FusionEnd {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "FusionEnd",
            &[("shape", ELIST), ("strides", ELIST), ("dtype", DTYPE)],
        )
    }
    fn n_inputs(&self) -> usize {
        1
    }

    fn rewrites(&self) -> Vec<Rule> {
        // Seven rule families build and extend FE-bracketed regions. Each
        // pair-fuse rule's LHS pattern matches *un-fused* `KernelX` ops; the
        // RHS produces `FusedX` variants in a different egglog sort, so the
        // rule's own output cannot re-match its LHS — cascade is prevented
        // by typing rather than by a discriminator field.
        //
        // Stride compatibility is expressed by reusing variable names: a
        // unary inside a region matches `(KernelU ?shape ?s ?s ?dt)` (in =
        // out, no transpose); a binary feeding a downstream op binds the
        // binary's out-stride to the downstream op's in-stride along the
        // connecting side.
        let mut rules = Vec::new();

        // (KernelX kind, FusedX kind)
        let unaries: &[(&str, &str)] = &[
            ("KernelSin", "FusedSin"),
            ("KernelSqrt", "FusedSqrt"),
            ("KernelExp", "FusedExp"),
            ("KernelExp2", "FusedExp2"),
            ("KernelLog2", "FusedLog2"),
            ("KernelRecip", "FusedRecip"),
        ];
        // (KernelX kind, FusedX kind, rule-name label)
        let binaries: &[(&str, &str, &str)] = &[
            ("KernelAdd", "FusedAdd", "Add"),
            ("KernelMul", "FusedMul", "Mul"),
        ];

        // 1. Pair-fuse U → U: U2(U1(x)) → FE(FU2(FU1(FS(x)))).
        for (ki1, fi1) in unaries {
            for (ko2, fo2) in unaries {
                rules.push(Rule::raw(format!(
                    "(rule (
                        (= ?u1 (Op ({ki1} ?shape ?s ?s ?dt) (ICons ?x (INil))))
                        (= ?u2 (Op ({ko2} ?shape ?s ?s ?dt) (ICons ?u1 (INil))))
                     ) (
                        (let ?fs (Op (FusionStart ?shape ?s ?dt) (ICons ?x (INil))))
                        (let ?fu1 (Op ({fi1} ?shape ?s ?s ?dt) (ICons ?fs (INil))))
                        (let ?fu2 (Op ({fo2} ?shape ?s ?s ?dt) (ICons ?fu1 (INil))))
                        (let ?fe (Op (FusionEnd ?shape ?s ?dt) (ICons ?fu2 (INil))))
                        (union ?u2 ?fe)
                     ) :ruleset fusion_pair :name \"pair-fuse-U-U-{ki1}-{ko2}\")"
                )));
            }
        }

        // 2. Pair-fuse B → U: U(B(a, b)) → FE(FU(FB(FS(a), FS(b)))).
        for (kb, fb, lb) in binaries {
            for (ku, fu) in unaries {
                rules.push(Rule::raw(format!(
                    "(rule (
                        (= ?bin (Op ({kb} ?shape ?a_s ?b_s ?o_s ?dt)
                                     (ICons ?a (ICons ?b (INil)))))
                        (= ?u (Op ({ku} ?shape ?o_s ?o_s ?dt) (ICons ?bin (INil))))
                     ) (
                        (let ?fs_a (Op (FusionStart ?shape ?a_s ?dt) (ICons ?a (INil))))
                        (let ?fs_b (Op (FusionStart ?shape ?b_s ?dt) (ICons ?b (INil))))
                        (let ?fbin (Op ({fb} ?shape ?a_s ?b_s ?o_s ?dt)
                                       (ICons ?fs_a (ICons ?fs_b (INil)))))
                        (let ?fu (Op ({fu} ?shape ?o_s ?o_s ?dt) (ICons ?fbin (INil))))
                        (let ?fe (Op (FusionEnd ?shape ?o_s ?dt) (ICons ?fu (INil))))
                        (union ?u ?fe)
                     ) :ruleset fusion_pair :name \"pair-fuse-B-U-{lb}-{ku}\")"
                )));
            }
        }

        // 3. Pair-fuse U → B (lhs / rhs): unary feeds binary's A or B input.
        //    LHS:  B(U(a), b) → FE(FB(FU(FS(a)), FS(b))).
        //    RHS:  B(a, U(b)) → FE(FB(FS(a), FU(FS(b)))).
        for (ku, fu) in unaries {
            for (kb, fb, lb) in binaries {
                rules.push(Rule::raw(format!(
                    "(rule (
                        (= ?u (Op ({ku} ?shape ?u_s ?u_s ?dt) (ICons ?a (INil))))
                        (= ?bin (Op ({kb} ?shape ?u_s ?b_s ?o_s ?dt)
                                     (ICons ?u (ICons ?b (INil)))))
                     ) (
                        (let ?fs_a (Op (FusionStart ?shape ?u_s ?dt) (ICons ?a (INil))))
                        (let ?fs_b (Op (FusionStart ?shape ?b_s ?dt) (ICons ?b (INil))))
                        (let ?fu (Op ({fu} ?shape ?u_s ?u_s ?dt) (ICons ?fs_a (INil))))
                        (let ?fbin (Op ({fb} ?shape ?u_s ?b_s ?o_s ?dt)
                                       (ICons ?fu (ICons ?fs_b (INil)))))
                        (let ?fe (Op (FusionEnd ?shape ?o_s ?dt) (ICons ?fbin (INil))))
                        (union ?bin ?fe)
                     ) :ruleset fusion_pair :name \"pair-fuse-U-B-lhs-{ku}-{lb}\")"
                )));
                rules.push(Rule::raw(format!(
                    "(rule (
                        (= ?u (Op ({ku} ?shape ?u_s ?u_s ?dt) (ICons ?b (INil))))
                        (= ?bin (Op ({kb} ?shape ?a_s ?u_s ?o_s ?dt)
                                     (ICons ?a (ICons ?u (INil)))))
                     ) (
                        (let ?fs_a (Op (FusionStart ?shape ?a_s ?dt) (ICons ?a (INil))))
                        (let ?fs_b (Op (FusionStart ?shape ?u_s ?dt) (ICons ?b (INil))))
                        (let ?fu (Op ({fu} ?shape ?u_s ?u_s ?dt) (ICons ?fs_b (INil))))
                        (let ?fbin (Op ({fb} ?shape ?a_s ?u_s ?o_s ?dt)
                                       (ICons ?fs_a (ICons ?fu (INil)))))
                        (let ?fe (Op (FusionEnd ?shape ?o_s ?dt) (ICons ?fbin (INil))))
                        (union ?bin ?fe)
                     ) :ruleset fusion_pair :name \"pair-fuse-U-B-rhs-{ku}-{lb}\")"
                )));
            }
        }

        // 4. Pair-fuse B → B (lhs / rhs): inner binary feeds outer's A or B.
        for (kbi, fbi, lbi) in binaries {
            for (kbo, fbo, lbo) in binaries {
                rules.push(Rule::raw(format!(
                    "(rule (
                        (= ?bi (Op ({kbi} ?shape ?ai_s ?bi_s ?oi_s ?dt)
                                    (ICons ?a (ICons ?b (INil)))))
                        (= ?bo (Op ({kbo} ?shape ?oi_s ?co_s ?oo_s ?dt)
                                    (ICons ?bi (ICons ?c (INil)))))
                     ) (
                        (let ?fs_a (Op (FusionStart ?shape ?ai_s ?dt) (ICons ?a (INil))))
                        (let ?fs_b (Op (FusionStart ?shape ?bi_s ?dt) (ICons ?b (INil))))
                        (let ?fs_c (Op (FusionStart ?shape ?co_s ?dt) (ICons ?c (INil))))
                        (let ?fbi (Op ({fbi} ?shape ?ai_s ?bi_s ?oi_s ?dt)
                                       (ICons ?fs_a (ICons ?fs_b (INil)))))
                        (let ?fbo (Op ({fbo} ?shape ?oi_s ?co_s ?oo_s ?dt)
                                       (ICons ?fbi (ICons ?fs_c (INil)))))
                        (let ?fe (Op (FusionEnd ?shape ?oo_s ?dt) (ICons ?fbo (INil))))
                        (union ?bo ?fe)
                     ) :ruleset fusion_pair :name \"pair-fuse-B-B-lhs-{lbi}-{lbo}\")"
                )));
                rules.push(Rule::raw(format!(
                    "(rule (
                        (= ?bi (Op ({kbi} ?shape ?ai_s ?bi_s ?oi_s ?dt)
                                    (ICons ?a (ICons ?b (INil)))))
                        (= ?bo (Op ({kbo} ?shape ?co_s ?oi_s ?oo_s ?dt)
                                    (ICons ?c (ICons ?bi (INil)))))
                     ) (
                        (let ?fs_a (Op (FusionStart ?shape ?ai_s ?dt) (ICons ?a (INil))))
                        (let ?fs_b (Op (FusionStart ?shape ?bi_s ?dt) (ICons ?b (INil))))
                        (let ?fs_c (Op (FusionStart ?shape ?co_s ?dt) (ICons ?c (INil))))
                        (let ?fbi (Op ({fbi} ?shape ?ai_s ?bi_s ?oi_s ?dt)
                                       (ICons ?fs_a (ICons ?fs_b (INil)))))
                        (let ?fbo (Op ({fbo} ?shape ?co_s ?oi_s ?oo_s ?dt)
                                       (ICons ?fs_c (ICons ?fbi (INil)))))
                        (let ?fe (Op (FusionEnd ?shape ?oo_s ?dt) (ICons ?fbo (INil))))
                        (union ?bo ?fe)
                     ) :ruleset fusion_pair :name \"pair-fuse-B-B-rhs-{lbi}-{lbo}\")"
                )));
            }
        }

        // 5. Grow FE → U: U(FE(inner)) → FE(FU(inner)). No new FS.
        for (ku, fu) in unaries {
            rules.push(Rule::raw(format!(
                "(rule (
                    (= ?fe (Op (FusionEnd ?shape ?s ?dt) (ICons ?inner (INil))))
                    (= ?u (Op ({ku} ?shape ?s ?s ?dt) (ICons ?fe (INil))))
                 ) (
                    (let ?fu (Op ({fu} ?shape ?s ?s ?dt) (ICons ?inner (INil))))
                    (let ?new_fe (Op (FusionEnd ?shape ?s ?dt) (ICons ?fu (INil))))
                    (union ?u ?new_fe)
                 ) :ruleset fusion_grow :name \"grow-FE-U-{ku}\")"
            )));
        }

        // 6. Grow FE → B (lhs / rhs): one input is the FE, the other external.
        for (kb, fb, lb) in binaries {
            rules.push(Rule::raw(format!(
                "(rule (
                    (= ?fe (Op (FusionEnd ?shape ?a_s ?dt) (ICons ?inner_a (INil))))
                    (= ?bin (Op ({kb} ?shape ?a_s ?b_s ?o_s ?dt)
                                 (ICons ?fe (ICons ?b (INil)))))
                 ) (
                    (let ?fs_b (Op (FusionStart ?shape ?b_s ?dt) (ICons ?b (INil))))
                    (let ?fbin (Op ({fb} ?shape ?a_s ?b_s ?o_s ?dt)
                                   (ICons ?inner_a (ICons ?fs_b (INil)))))
                    (let ?new_fe (Op (FusionEnd ?shape ?o_s ?dt) (ICons ?fbin (INil))))
                    (union ?bin ?new_fe)
                 ) :ruleset fusion_grow :name \"grow-FE-B-lhs-{lb}\")"
            )));
            rules.push(Rule::raw(format!(
                "(rule (
                    (= ?fe (Op (FusionEnd ?shape ?b_s ?dt) (ICons ?inner_b (INil))))
                    (= ?bin (Op ({kb} ?shape ?a_s ?b_s ?o_s ?dt)
                                 (ICons ?a (ICons ?fe (INil)))))
                 ) (
                    (let ?fs_a (Op (FusionStart ?shape ?a_s ?dt) (ICons ?a (INil))))
                    (let ?fbin (Op ({fb} ?shape ?a_s ?b_s ?o_s ?dt)
                                   (ICons ?fs_a (ICons ?inner_b (INil)))))
                    (let ?new_fe (Op (FusionEnd ?shape ?o_s ?dt) (ICons ?fbin (INil))))
                    (union ?bin ?new_fe)
                 ) :ruleset fusion_grow :name \"grow-FE-B-rhs-{lb}\")"
            )));
        }

        // 7. Merge two FEs at a binary: B(FE(ia), FE(ib)) → FE(FB(ia, ib)).
        //
        // This is destructive: after creating the larger region, subsume the
        // two smaller FusionEnd rows. Without that, independently-grown left
        // and right regions form a Cartesian product, then those alternatives
        // can merge again higher in the graph.
        for (kb, fb, lb) in binaries {
            rules.push(Rule::raw(format!(
                "(rule (
                    (= ?fe_a (Op (FusionEnd ?shape ?a_s ?dt) (ICons ?inner_a (INil))))
                    (= ?fe_b (Op (FusionEnd ?shape ?b_s ?dt) (ICons ?inner_b (INil))))
                    (= ?bin (Op ({kb} ?shape ?a_s ?b_s ?o_s ?dt)
                                 (ICons ?fe_a (ICons ?fe_b (INil)))))
                 ) (
                    (let ?fbin (Op ({fb} ?shape ?a_s ?b_s ?o_s ?dt)
                                   (ICons ?inner_a (ICons ?inner_b (INil)))))
                    (let ?new_fe (Op (FusionEnd ?shape ?o_s ?dt) (ICons ?fbin (INil))))
                    (union ?bin ?new_fe)
                    (subsume (Op (FusionEnd ?shape ?a_s ?dt) (ICons ?inner_a (INil))))
                    (subsume (Op (FusionEnd ?shape ?b_s ?dt) (ICons ?inner_b (INil))))
                 ) :ruleset fusion_merge :name \"merge-FE-FE-{lb}\")"
            )));
        }

        // No dissolve rule (`FS(FE(x)) → x`): unioning FS's eclass with FE's
        // inner eclass creates self-referential eclasses after grow rules
        // extend the downstream region, and extraction then panics with
        // `Cycle(NodeIndex(_))`. Grow rules already compose adjacent regions
        // correctly without dissolve.

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
                shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache).unwrap(),
                strides: extract_expr_list(egraph, kind_children[1], list_cache, expr_cache)
                    .unwrap(),
                dtype: extract_dtype(egraph, kind_children[2]),
            })),
            input_enodes,
        )
    }
}

impl KernelOp for FusionEnd {
    fn compile(
        &self,
        _stream: &Arc<CudaStream>,
        _compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> CompileOut {
        unreachable!("FusionEnd must be compiled through fusion region codegen")
    }
    fn output_size(&self) -> Expression {
        self.shape.iter().copied().product()
    }
    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }
    fn output_dtype(&self) -> DType {
        self.dtype
    }
    fn kernel_name(&self) -> &'static str {
        "FusionEnd"
    }
}
