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
// codegen lives in `region_codegen`. Both markers' `compile()` is
// `unreachable!()` — region codegen folds them away
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
    FxHashMap<Symbol, CudaSlice<u8>>,
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
    fn output_aliases_input(&self) -> Option<usize> {
        Some(0)
    }
    fn mutates_aliased_input(&self) -> bool {
        false
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

    fn egglog_declarations(&self) -> Vec<String> {
        vec![
            "(ruleset fusion_grow_safe_late)".to_string(),
            "(ruleset fusion_merge_safe_late)".to_string(),
        ]
    }

    fn rewrites(&self) -> Vec<Rule> {
        // Grow already well-formed regions through adjacent elementwise work.
        // These rules are legality-by-construction:
        // - one shared `?shape` fixes the iteration domain;
        // - the producer FusionEnd stride is reused as the exact consumer
        //   input stride, so no transpose/view is crossed;
        // - every external FusionStart is stamped with its producer dtype;
        // - every interior/result node is stamped with its own result dtype.
        //
        // Deliberately do not union a FusionStart with an absorbed producer.
        // That older rule family could make cyclic eclasses. Forward growth
        // and binary merge cover chains and DAG joins without changing an
        // existing region boundary's eclass.
        let mut rules = Vec::new();
        let unaries: &[(&str, &str)] = &[
            ("Sin", "Sin"),
            ("Sqrt", "Sqrt"),
            ("Exp2", "Exp2"),
            ("Log2", "Log2"),
            ("Recip", "Recip"),
        ];
        let binaries: &[(&str, &str)] = &[("Add", "Add"), ("Mul", "Mul")];

        // U(FE(inner)): match the FE's materialized output layout exactly at
        // U's input, then carry U's possibly different output layout forward.
        for (hlir, opcode) in unaries {
            rules.push(Rule::raw(format!(
                "(rule (
                    (= ?fe (Op (FusionEnd ?shape ?in_s ?dt_in) (ICons ?inner (INil))))
                    (= ?u (Op ({hlir} ?shape ?in_s ?out_s) (ICons ?fe (INil))))
                    (= ?dt_out (dtype ?u))
                 ) (
                    (let ?elem (Op (CudaUnaryElementwise \"{opcode}\" ?shape ?in_s ?out_s ?dt_out)
                                   (ICons ?inner (INil))))
                    (let ?new_fe (Op (FusionEnd ?shape ?out_s ?dt_out) (ICons ?elem (INil))))
                    (union ?u ?new_fe)
                    (set (dtype ?new_fe) ?dt_out)
                 ) :ruleset fusion_grow_safe_late :name \"grow-safe-FE-U-{hlir}\")"
            )));
        }

        // Cast is flat elementwise and its HLIR contract requires `size` to
        // equal the input buffer's element count. Since the matched FE is an
        // alternative in that exact input eclass, retaining its iteration
        // domain and layout is exact even when egglog has not normalized the
        // corresponding symbolic product to `size`.
        rules.push(Rule::raw(
            "(rule (
                (= ?fe (Op (FusionEnd ?shape ?s ?dt_in) (ICons ?inner (INil))))
                (= ?cast (Op (Cast ?size ?dt_out) (ICons ?fe (INil))))
             ) (
                (let ?elem (Op (CudaUnaryElementwise \"Cast\" ?shape ?s ?s ?dt_out)
                               (ICons ?inner (INil))))
                (let ?new_fe (Op (FusionEnd ?shape ?s ?dt_out) (ICons ?elem (INil))))
                (union ?cast ?new_fe)
                (set (dtype ?new_fe) ?dt_out)
             ) :ruleset fusion_grow_safe_late :name \"grow-safe-FE-Cast\")",
        ));

        // B(FE(inner), external) and its mirror. Dtypes are per-value rather
        // than assumed uniform, which preserves mixed bf16/f32 arithmetic.
        for (hlir, opcode) in binaries {
            rules.push(Rule::raw(format!(
                "(rule (
                    (= ?fe (Op (FusionEnd ?shape ?a_s ?dt_a) (ICons ?inner_a (INil))))
                    (= ?bin (Op ({hlir} ?shape ?a_s ?b_s ?out_s)
                                 (ICons ?fe (ICons ?b (INil)))))
                    (= ?dt_b (dtype ?b))
                    (= ?dt_out (dtype ?bin))
                 ) (
                    (let ?fs_b (Op (FusionStart ?shape ?b_s ?dt_b) (ICons ?b (INil))))
                    (let ?elem (Op (CudaBinaryElementwise \"{opcode}\" ?shape ?a_s ?b_s ?out_s ?dt_out)
                                   (ICons ?inner_a (ICons ?fs_b (INil)))))
                    (let ?new_fe (Op (FusionEnd ?shape ?out_s ?dt_out) (ICons ?elem (INil))))
                    (union ?bin ?new_fe)
                    (set (dtype ?new_fe) ?dt_out)
                 ) :ruleset fusion_grow_safe_late :name \"grow-safe-FE-B-lhs-{hlir}\")"
            )));
            rules.push(Rule::raw(format!(
                "(rule (
                    (= ?fe (Op (FusionEnd ?shape ?b_s ?dt_b) (ICons ?inner_b (INil))))
                    (= ?bin (Op ({hlir} ?shape ?a_s ?b_s ?out_s)
                                 (ICons ?a (ICons ?fe (INil)))))
                    (= ?dt_a (dtype ?a))
                    (= ?dt_out (dtype ?bin))
                 ) (
                    (let ?fs_a (Op (FusionStart ?shape ?a_s ?dt_a) (ICons ?a (INil))))
                    (let ?elem (Op (CudaBinaryElementwise \"{opcode}\" ?shape ?a_s ?b_s ?out_s ?dt_out)
                                   (ICons ?fs_a (ICons ?inner_b (INil)))))
                    (let ?new_fe (Op (FusionEnd ?shape ?out_s ?dt_out) (ICons ?elem (INil))))
                    (union ?bin ?new_fe)
                    (set (dtype ?new_fe) ?dt_out)
                 ) :ruleset fusion_grow_safe_late :name \"grow-safe-FE-B-rhs-{hlir}\")"
            )));

            // Join two independently well-formed regions at one binary op.
            rules.push(Rule::raw(format!(
                "(rule (
                    (= ?fe_a (Op (FusionEnd ?shape ?a_s ?dt_a) (ICons ?inner_a (INil))))
                    (= ?fe_b (Op (FusionEnd ?shape ?b_s ?dt_b) (ICons ?inner_b (INil))))
                    (= ?bin (Op ({hlir} ?shape ?a_s ?b_s ?out_s)
                                 (ICons ?fe_a (ICons ?fe_b (INil)))))
                    (= ?dt_out (dtype ?bin))
                 ) (
                    (let ?elem (Op (CudaBinaryElementwise \"{opcode}\" ?shape ?a_s ?b_s ?out_s ?dt_out)
                                   (ICons ?inner_a (ICons ?inner_b (INil)))))
                    (let ?new_fe (Op (FusionEnd ?shape ?out_s ?dt_out) (ICons ?elem (INil))))
                    (union ?bin ?new_fe)
                    (set (dtype ?new_fe) ?dt_out)
                 ) :ruleset fusion_merge_safe_late :name \"merge-safe-FE-FE-{hlir}\")"
            )));
        }

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
