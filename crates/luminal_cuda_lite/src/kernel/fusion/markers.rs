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
    fn output_aliases_input(&self) -> Option<usize> {
        Some(0)
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
        // Generic region growth works directly from HLIR elementwise ops into
        // `Cuda*Elementwise` region nodes. The concrete HLIR op still appears in
        // the egraph, so fusion remains a normal nondestructive alternative, but
        // the region-internal representation is arity based instead of one
        // dedicated fused sort per operation.
        let mut rules = Vec::new();

        let unaries: &[(&str, &str)] = &[
            ("Sin", "Sin"),
            ("Sqrt", "Sqrt"),
            ("Exp2", "Exp2"),
            ("Log2", "Log2"),
            ("Recip", "Recip"),
        ];
        let binaries: &[(&str, &str)] = &[("Add", "Add"), ("Mul", "Mul")];

        // Grow FE → unary consumer: U(FE(inner)) → FE(CudaUnary(inner)).
        for (hlir, opcode) in unaries {
            rules.push(Rule::raw(format!(
                "(rule (
                    (= ?fe (Op (FusionEnd ?shape ?s ?dt) (ICons ?inner (INil))))
                    (= ?u (Op ({hlir} ?shape ?s ?s) (ICons ?fe (INil))))
                 ) (
                    (let ?elem (Op (CudaUnaryElementwise \"{opcode}\" ?shape ?s ?s ?dt)
                                   (ICons ?inner (INil))))
                    (let ?new_fe (Op (FusionEnd ?shape ?s ?dt) (ICons ?elem (INil))))
                    (union ?u ?new_fe)
                    (set (dtype ?new_fe) ?dt)
                 ) :ruleset fusion_grow :name \"grow-FE-U-{hlir}\")"
            )));
        }

        // Grow FE → Cast consumer: Cast(FE(inner)) → FE(CudaCast(inner)).
        // Cast is the one region op whose output dtype differs from its
        // input's; the new FE takes the cast's target dtype. HLIR Cast is
        // positionwise (out[z] = (T)in[z]), so it preserves the producer's
        // layout and the FE's shape/strides carry over unchanged.
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
             ) :ruleset fusion_grow :name \"grow-FE-Cast\")",
        ));

        // Absorb a Cast producer through a FusionStart boundary:
        // FS(Cast(x)) → CudaCast(FS(x)). The region reads x with the same
        // strides it would have used on the cast's output (cast preserves
        // layout) and converts in-register.
        rules.push(Rule::raw(
            "(rule (
                (= ?cast (Op (Cast ?size ?dt_out) (ICons ?x (INil))))
                (= ?fs_c (Op (FusionStart ?shape ?s ?dt_out) (ICons ?cast (INil))))
                (= ?dt_in (dtype ?x))
             ) (
                (let ?fs_x (Op (FusionStart ?shape ?s ?dt_in) (ICons ?x (INil))))
                (let ?elem (Op (CudaUnaryElementwise \"Cast\" ?shape ?s ?s ?dt_out)
                               (ICons ?fs_x (INil))))
                (union ?fs_c ?elem)
             ) :ruleset fusion_grow :name \"grow-Cast-FS\")",
        ));

        // Cast variant of the nested-FS-FE cleanup. Without it, the
        // grow-FE-Cast + grow-Cast-FS pair congruence-merges an FS eclass
        // with the FE eclass it wraps (FS(dt_in)(FE-result) vs the fused
        // cast elem reading the FE's interior), leaving a 2-node FS/FE
        // cycle that extraction can select. Deleting the redundant
        // FS-wrapping-FE enode breaks the cycle; the fused form remains.
        rules.push(Rule::raw(
            "(rule (
                (= ?inner_fe (Op (FusionEnd ?shape ?s ?dt_in) (ICons ?inner (INil))))
                (= ?bad_fs (Op (FusionStart ?shape ?s ?dt_in) (ICons ?inner_fe (INil))))
                (= ?bad_elem (Op (CudaUnaryElementwise \"Cast\" ?shape ?s ?s ?dt_out)
                                 (ICons ?bad_fs (INil))))
                (= ?bad_fe (Op (FusionEnd ?shape ?s ?dt_out) (ICons ?bad_elem (INil))))
                (= ?good_elem (Op (CudaUnaryElementwise \"Cast\" ?shape ?s ?s ?dt_out)
                                  (ICons ?inner (INil))))
                (= ?good_fe (Op (FusionEnd ?shape ?s ?dt_out) (ICons ?good_elem (INil))))
                (= ?bad_fe ?good_fe)
             ) (
                (delete (Op (FusionStart ?shape ?s ?dt_in) (ICons ?inner_fe (INil))))
             ) :ruleset cleanup :name \"cleanup-nested-FS-FE-cast\")",
        ));

        // Genome freeze: grow-Cast-FS leaves each cast-bearing FS eclass
        // with two extraction-equivalent variants — the bare FS reading the
        // materialized cast output, and the absorbed CudaCast reading the
        // original buffer and converting in-register. Both wirings move
        // the same bytes for the bandwidth-bound regions they sit in, but
        // each pair doubles the search genome and makes candidates emit
        // textually fresh region kernels. Delete the bare-FS variant once
        // growth is done (cleanup runs after the fusion_grow cycles, so
        // the row is not re-derived); the premise proves the absorbed
        // survivor exists in the same eclass, so the eclass never empties.
        rules.push(Rule::raw(
            "(rule (
                (= ?cast (Op (Cast ?size ?dt_out) (ICons ?x (INil))))
                (= ?fs_c (Op (FusionStart ?shape ?s ?dt_out) (ICons ?cast (INil))))
                (= ?fs_c (Op (CudaUnaryElementwise \"Cast\" ?shape ?is ?os ?dt_out)
                             (ICons ?fs_x (INil))))
             ) (
                (delete (Op (FusionStart ?shape ?s ?dt_out) (ICons ?cast (INil))))
             ) :ruleset cleanup :name \"cleanup-FS-with-absorbed-cast\")",
        ));

        // Grow FE → binary consumer, left and right orientations.
        for (hlir, opcode) in binaries {
            rules.push(Rule::raw(format!(
                "(rule (
                    (= ?fe (Op (FusionEnd ?shape ?a_s ?dt) (ICons ?inner_a (INil))))
                    (= ?bin (Op ({hlir} ?shape ?a_s ?b_s ?out_s)
                                 (ICons ?fe (ICons ?b (INil)))))
                 ) (
                    (let ?fs_b (Op (FusionStart ?shape ?b_s ?dt) (ICons ?b (INil))))
                    (let ?elem (Op (CudaBinaryElementwise \"{opcode}\" ?shape ?a_s ?b_s ?out_s ?dt)
                                   (ICons ?inner_a (ICons ?fs_b (INil)))))
                    (let ?new_fe (Op (FusionEnd ?shape ?out_s ?dt) (ICons ?elem (INil))))
                    (union ?bin ?new_fe)
                    (set (dtype ?new_fe) ?dt)
                 ) :ruleset fusion_grow :name \"grow-FE-B-lhs-{hlir}\")"
            )));
            rules.push(Rule::raw(format!(
                "(rule (
                    (= ?fe (Op (FusionEnd ?shape ?b_s ?dt) (ICons ?inner_b (INil))))
                    (= ?bin (Op ({hlir} ?shape ?a_s ?b_s ?out_s)
                                 (ICons ?a (ICons ?fe (INil)))))
                 ) (
                    (let ?fs_a (Op (FusionStart ?shape ?a_s ?dt) (ICons ?a (INil))))
                    (let ?elem (Op (CudaBinaryElementwise \"{opcode}\" ?shape ?a_s ?b_s ?out_s ?dt)
                                   (ICons ?fs_a (ICons ?inner_b (INil)))))
                    (let ?new_fe (Op (FusionEnd ?shape ?out_s ?dt) (ICons ?elem (INil))))
                    (union ?bin ?new_fe)
                    (set (dtype ?new_fe) ?dt)
                 ) :ruleset fusion_grow :name \"grow-FE-B-rhs-{hlir}\")"
            )));
        }

        // Absorb an elementwise producer through a FusionStart boundary. This
        // makes a region that initially treats `producer(...)` as an external
        // input able to pull that producer inside later.
        for (hlir, opcode) in unaries {
            rules.push(Rule::raw(format!(
                "(rule (
                    (= ?u (Op ({hlir} ?shape ?s ?s) (ICons ?x (INil))))
                    (= ?fs_u (Op (FusionStart ?shape ?s ?dt) (ICons ?u (INil))))
                 ) (
                    (let ?fs_x (Op (FusionStart ?shape ?s ?dt) (ICons ?x (INil))))
                    (let ?elem (Op (CudaUnaryElementwise \"{opcode}\" ?shape ?s ?s ?dt)
                                   (ICons ?fs_x (INil))))
                    (union ?fs_u ?elem)
                 ) :ruleset fusion_grow :name \"grow-U-FS-{hlir}\")"
            )));
            rules.push(Rule::raw(format!(
                "(rule (
                    (= ?inner_fe (Op (FusionEnd ?shape ?s ?dt) (ICons ?inner (INil))))
                    (= ?bad_fs (Op (FusionStart ?shape ?s ?dt) (ICons ?inner_fe (INil))))
                    (= ?bad_elem (Op (CudaUnaryElementwise \"{opcode}\" ?shape ?s ?s ?dt)
                                     (ICons ?bad_fs (INil))))
                    (= ?bad_fe (Op (FusionEnd ?shape ?s ?dt) (ICons ?bad_elem (INil))))
                    (= ?good_elem (Op (CudaUnaryElementwise \"{opcode}\" ?shape ?s ?s ?dt)
                                      (ICons ?inner (INil))))
                    (= ?good_fe (Op (FusionEnd ?shape ?s ?dt) (ICons ?good_elem (INil))))
                    (= ?bad_fe ?good_fe)
                 ) (
                    (delete (Op (FusionStart ?shape ?s ?dt) (ICons ?inner_fe (INil))))
                 ) :ruleset cleanup :name \"cleanup-nested-FS-FE-unary-{hlir}\")"
            )));
        }
        for (hlir, opcode) in binaries {
            rules.push(Rule::raw(format!(
                "(rule (
                    (= ?bin (Op ({hlir} ?shape ?a_s ?b_s ?out_s)
                                 (ICons ?a (ICons ?b (INil)))))
                    (= ?fs_bin (Op (FusionStart ?shape ?out_s ?dt) (ICons ?bin (INil))))
                 ) (
                    (let ?fs_a (Op (FusionStart ?shape ?a_s ?dt) (ICons ?a (INil))))
                    (let ?fs_b (Op (FusionStart ?shape ?b_s ?dt) (ICons ?b (INil))))
                    (let ?elem (Op (CudaBinaryElementwise \"{opcode}\" ?shape ?a_s ?b_s ?out_s ?dt)
                                   (ICons ?fs_a (ICons ?fs_b (INil)))))
                    (union ?fs_bin ?elem)
                 ) :ruleset fusion_grow :name \"grow-B-FS-{hlir}\")"
            )));
            rules.push(Rule::raw(format!(
                "(rule (
                    (= ?inner_fe (Op (FusionEnd ?shape ?a_s ?dt) (ICons ?inner_a (INil))))
                    (= ?bad_fs (Op (FusionStart ?shape ?a_s ?dt) (ICons ?inner_fe (INil))))
                    (= ?fs_b (Op (FusionStart ?shape ?b_s ?dt) (ICons ?b (INil))))
                    (= ?bad_elem (Op (CudaBinaryElementwise \"{opcode}\" ?shape ?a_s ?b_s ?out_s ?dt)
                                     (ICons ?bad_fs (ICons ?fs_b (INil)))))
                    (= ?bad_fe (Op (FusionEnd ?shape ?out_s ?dt) (ICons ?bad_elem (INil))))
                    (= ?good_elem (Op (CudaBinaryElementwise \"{opcode}\" ?shape ?a_s ?b_s ?out_s ?dt)
                                      (ICons ?inner_a (ICons ?fs_b (INil)))))
                    (= ?good_fe (Op (FusionEnd ?shape ?out_s ?dt) (ICons ?good_elem (INil))))
                    (= ?bad_fe ?good_fe)
                 ) (
                    (delete (Op (FusionStart ?shape ?a_s ?dt) (ICons ?inner_fe (INil))))
                 ) :ruleset cleanup :name \"cleanup-nested-FS-FE-binary-lhs-{hlir}\")"
            )));
            rules.push(Rule::raw(format!(
                "(rule (
                    (= ?inner_fe (Op (FusionEnd ?shape ?b_s ?dt) (ICons ?inner_b (INil))))
                    (= ?bad_fs (Op (FusionStart ?shape ?b_s ?dt) (ICons ?inner_fe (INil))))
                    (= ?fs_a (Op (FusionStart ?shape ?a_s ?dt) (ICons ?a (INil))))
                    (= ?bad_elem (Op (CudaBinaryElementwise \"{opcode}\" ?shape ?a_s ?b_s ?out_s ?dt)
                                     (ICons ?fs_a (ICons ?bad_fs (INil)))))
                    (= ?bad_fe (Op (FusionEnd ?shape ?out_s ?dt) (ICons ?bad_elem (INil))))
                    (= ?good_elem (Op (CudaBinaryElementwise \"{opcode}\" ?shape ?a_s ?b_s ?out_s ?dt)
                                      (ICons ?fs_a (ICons ?inner_b (INil)))))
                    (= ?good_fe (Op (FusionEnd ?shape ?out_s ?dt) (ICons ?good_elem (INil))))
                    (= ?bad_fe ?good_fe)
                 ) (
                    (delete (Op (FusionStart ?shape ?b_s ?dt) (ICons ?inner_fe (INil))))
                 ) :ruleset cleanup :name \"cleanup-nested-FS-FE-binary-rhs-{hlir}\")"
            )));
        }

        // Merge two FEs at a binary: B(FE(ia), FE(ib)) → FE(CudaBinary(ia, ib)).
        for (hlir, opcode) in binaries {
            rules.push(Rule::raw(format!(
                "(rule (
                    (= ?fe_a (Op (FusionEnd ?shape ?a_s ?dt) (ICons ?inner_a (INil))))
                    (= ?fe_b (Op (FusionEnd ?shape ?b_s ?dt) (ICons ?inner_b (INil))))
                    (= ?bin (Op ({hlir} ?shape ?a_s ?b_s ?out_s)
                                 (ICons ?fe_a (ICons ?fe_b (INil)))))
                 ) (
                    (let ?elem (Op (CudaBinaryElementwise \"{opcode}\" ?shape ?a_s ?b_s ?out_s ?dt)
                                   (ICons ?inner_a (ICons ?inner_b (INil)))))
                    (let ?new_fe (Op (FusionEnd ?shape ?out_s ?dt) (ICons ?elem (INil))))
                    (union ?bin ?new_fe)
                    (set (dtype ?new_fe) ?dt)
                 ) :ruleset fusion_merge :name \"merge-FE-FE-{hlir}\")"
            )));
        }

        // No dissolve rule (`FS(FE(x)) → x`): unioning FS's eclass with FE's
        // inner eclass creates self-referential eclasses after grow rules
        // extend the downstream region, and extraction then panics with
        // `Cycle(NodeIndex(_))`. Grow rules already compose adjacent regions
        // correctly without dissolve.

        rules.push(Rule::raw(
            "(rule (
                (= ?fe (Op (FusionEnd ?fe_shape ?fe_s ?dt) (ICons ?inner (INil))))
                (= ?inner (Op (CudaUnaryElementwise ?op ?inner_shape ?inner_in_s ?inner_s ?dt) ?inner_inputs))
                (!= ?fe_shape ?inner_shape)
             ) (
                (delete (Op (FusionEnd ?fe_shape ?fe_s ?dt) (ICons ?inner (INil))))
             ) :ruleset cleanup :name \"delete-malformed-FE-unary-shape\")",
        ));
        rules.push(Rule::raw(
            "(rule (
                (= ?fe (Op (FusionEnd ?fe_shape ?fe_s ?dt) (ICons ?inner (INil))))
                (= ?inner (Op (CudaUnaryElementwise ?op ?inner_shape ?inner_in_s ?inner_s ?dt) ?inner_inputs))
                (!= ?fe_s ?inner_s)
             ) (
                (delete (Op (FusionEnd ?fe_shape ?fe_s ?dt) (ICons ?inner (INil))))
             ) :ruleset cleanup :name \"delete-malformed-FE-unary-strides\")",
        ));
        rules.push(Rule::raw(
            "(rule (
                (= ?fe (Op (FusionEnd ?fe_shape ?fe_s ?dt) (ICons ?inner (INil))))
                (= ?inner (Op (CudaBinaryElementwise ?op ?inner_shape ?a_s ?b_s ?inner_s ?dt) ?inner_inputs))
                (!= ?fe_shape ?inner_shape)
             ) (
                (delete (Op (FusionEnd ?fe_shape ?fe_s ?dt) (ICons ?inner (INil))))
             ) :ruleset cleanup :name \"delete-malformed-FE-binary-shape\")",
        ));
        rules.push(Rule::raw(
            "(rule (
                (= ?fe (Op (FusionEnd ?fe_shape ?fe_s ?dt) (ICons ?inner (INil))))
                (= ?inner (Op (CudaBinaryElementwise ?op ?inner_shape ?a_s ?b_s ?inner_s ?dt) ?inner_inputs))
                (!= ?fe_s ?inner_s)
             ) (
                (delete (Op (FusionEnd ?fe_shape ?fe_s ?dt) (ICons ?inner (INil))))
             ) :ruleset cleanup :name \"delete-malformed-FE-binary-strides\")",
        ));
        rules.push(Rule::raw(
            "(rule (
                (= ?fe (Op (FusionEnd ?fe_shape ?fe_s ?dt) (ICons ?inner (INil))))
                (= ?inner (Op (FusionEnd ?inner_shape ?inner_s ?dt) ?inner_inputs))
                (!= ?fe_shape ?inner_shape)
             ) (
                (delete (Op (FusionEnd ?fe_shape ?fe_s ?dt) (ICons ?inner (INil))))
             ) :ruleset cleanup :name \"delete-malformed-FE-nested-shape\")",
        ));
        rules.push(Rule::raw(
            "(rule (
                (= ?fe (Op (FusionEnd ?fe_shape ?fe_s ?dt) (ICons ?inner (INil))))
                (= ?inner (Op (FusionEnd ?inner_shape ?inner_s ?dt) ?inner_inputs))
                (!= ?fe_s ?inner_s)
             ) (
                (delete (Op (FusionEnd ?fe_shape ?fe_s ?dt) (ICons ?inner (INil))))
             ) :ruleset cleanup :name \"delete-malformed-FE-nested-strides\")",
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
