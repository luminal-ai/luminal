//! Fused stable-argsort ranks: one kernel for the O(E²) pairwise rank
//! computation that `stable_argsort(descending)` spells as ~8 HLIR kernels
//! (two broadcast value views, three LessThans, the index tie-break chain,
//! and the comparison sum).
//!
//! `rank[s, j] = Σ_i (x[s,i] > x[s,j]) || (x[s,i] == x[s,j] && i < j)`
//!
//! The kernel reproduces the chain's exact count-based semantics (including
//! the index tie-break) and the rank→index scatter in one launch: element j
//! lands at sorted position rank(j) — bit-identical to the decomposed
//! spelling. The rewrite matches all the way through the scatter tail and
//! unions into the sorted-indices eclass emitted by
//! `stable_argsort(descending)` / `topk_indexes`.

use std::sync::Arc;

use crate::{
    compile_module_image_for_current_device,
    kernel::{KernelOp, hlir::generate_dyn_dims_defines},
};
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{ELIST, OP_KIND},
        extract_expr_list,
    },
    op::*,
    prelude::*,
};

#[derive(Default, Debug, Clone)]
pub struct KernelStableSortIdx {
    /// Output shape `(rows, E)`; rows may be dynamic.
    out_shape: Vec<Expression>,
}

impl EgglogOp for KernelStableSortIdx {
    fn sort(&self) -> SortDef {
        sort(OP_KIND, "KernelStableSortIdx", &[("out_shape", ELIST)])
    }

    fn n_inputs(&self) -> usize {
        1
    }

    fn rewrites(&self) -> Vec<Rule> {
        // Mirrors the exact emission of stable_argsort(axis=1, descending):
        //   a_val = x·1.0 viewed (rows, E_j, E_i→0)   strides (z·E, z, 0)
        //   b_val = x·1.0 viewed (rows, 0, E_i)       strides (z·E, 0, z)
        //   primary = b_val < a_val                   (descending: a > b)
        //   not_lt  = (a_val < b_val)·(−1) + 1
        //   not_gt  = (b_val < a_val)·(−1) + 1
        //   val_eq  = not_lt · not_gt
        //   idx_cmp = iota_i < iota_j (broadcast casts)
        //   cmp     = cast(primary) + val_eq·cast(idx_cmp)
        //   ranks   = Cast(Int)(Sum_axis1(cmp))
        //
        // Three relation-staged rules (pre → late → late): one monolithic
        // ~25-atom join explodes combinatorially on rolled bodies with
        // several distinct layer instances (every loosely-paired binary atom
        // multiplies the join by the instance count). Each stage walks a
        // chain from a strongly-pinned anchor; later stages key every atom
        // off relation-bound variables.
        vec![Rule::raw(
            "(relation stable_ranks_part (IR IR IR IR IR EList))
            (relation stable_ranks (IR IR EList))
            (rule
                (
                    ; ranks = Cast(Int)(Sum(cast(primary) + tiebreak))
                    (= ?a_val (Op (Mul ?av_shape
                        (ECons ?av_row (ECons (MIter) (ECons (MNum 0) (ENil))))
                        ?av_b ?av_o)
                        (ICons ?x (ICons ?one_a (INil)))))
                    (= ?one_a (Op (Constant 1.000000) (INil)))
                    (= ?b_val (Op (Mul ?bv_shape
                        (ECons ?bv_row (ECons (MNum 0) (ECons (MIter) (ENil))))
                        ?bv_b ?bv_o)
                        (ICons ?x2 (ICons ?one_b (INil)))))
                    (= ?x ?x2)
                    (= ?one_b (Op (Constant 1.000000) (INil)))
                    (= ?prim (Op (LessThan ?p_shape ?p_a ?p_b ?p_o)
                        (ICons ?b_val (ICons ?a_val (INil)))))
                    (= ?prim_f (Op (Cast ?pf_size (F32)) (ICons ?prim (INil))))
                    (= ?cmp (Op (Add ?c_shape ?c_a ?c_b ?c_o)
                        (ICons ?prim_f (ICons ?eqidx (INil)))))
                    (= ?sum (Op (Sum ?out_shape ?e_dim ?sum_in ?sum_k ?sum_out)
                        (ICons ?cmp (INil))))
                    (= ?ranks (Op (Cast ?r_size (Int)) (ICons ?sum (INil))))
                    (= (F32) (dtype ?x))
                )
                (
                    (stable_ranks_part ?ranks ?eqidx ?a_val ?b_val ?x ?out_shape)
                )
                :ruleset kernel_fuse_late_pre
                :name \"stable ranks part\"
            )
            (rule
                (
                    (stable_ranks_part ?ranks ?eqidx ?a_val ?b_val ?x ?out_shape)

                    ; tie-break: val_eq · (iota_i < iota_j)
                    (= ?eqidx (Op (Mul ?ei_shape ?ei_a ?ei_b ?ei_o)
                        (ICons ?val_eq (ICons ?idx_f (INil)))))
                    (= ?idx_f (Op (Cast ?if_size (F32)) (ICons ?idx_cmp (INil))))
                    (= ?idx_cmp (Op (LessThan ?ic_shape
                        (ECons (MNum 0) (ECons (MIter) (ECons (MNum 0) (ENil))))
                        (ECons (MNum 0) (ECons (MNum 0) (ECons (MIter) (ENil))))
                        ?ic_o)
                        (ICons ?iota_a (ICons ?iota_b (INil)))))

                    (= ?val_eq (Op (Mul ?ve_shape ?ve_a ?ve_b ?ve_o)
                        (ICons ?not_lt (ICons ?not_gt (INil)))))
                    (= ?not_lt (Op (Add ?nl_shape ?nl_a ?nl_b ?nl_o)
                        (ICons ?nl_neg (ICons ?nl_one (INil)))))
                    (= ?nl_neg (Op (Mul ?nln_shape ?nln_a ?nln_b ?nln_o)
                        (ICons ?lt_f (ICons ?nl_negone (INil)))))
                    (= ?lt_f (Op (Cast ?ltf_size (F32)) (ICons ?lt (INil))))
                    (= ?lt (Op (LessThan ?lt_shape ?lt_a ?lt_b ?lt_o)
                        (ICons ?a_val2 (ICons ?b_val2 (INil)))))
                    (= ?a_val ?a_val2)
                    (= ?b_val ?b_val2)
                    (= ?not_gt (Op (Add ?ng_shape ?ng_a ?ng_b ?ng_o)
                        (ICons ?ng_neg (ICons ?ng_one (INil)))))
                    (= ?ng_neg (Op (Mul ?ngn_shape ?ngn_a ?ngn_b ?ngn_o)
                        (ICons ?gt_f (ICons ?ng_negone (INil)))))
                    (= ?gt_f (Op (Cast ?gtf_size (F32)) (ICons ?gt (INil))))
                    (= ?gt (Op (LessThan ?gt_shape ?gt_a ?gt_b ?gt_o)
                        (ICons ?b_val3 (ICons ?a_val3 (INil)))))
                    (= ?a_val ?a_val3)
                    (= ?b_val ?b_val3)
                )
                (
                    (stable_ranks ?ranks ?x ?out_shape)
                )
                :ruleset kernel_fuse_late
                :name \"stable ranks tiebreak\"
            )
            (rule
                (
                    (stable_ranks ?ranks ?x ?out_shape)

                    ; rank → sorted-index scatter tail
                    (= ?rsc (Op (Mul ?rs_shape ?rs_a ?rs_b ?rs_o)
                        (ICons ?ranks (ICons ?one_s (INil)))))
                    (= ?one_s (Op (Iota (MNum 1) ?os_range) (INil)))
                    (= ?adj (Op (Add ?aj_shape ?aj_a ?aj_b ?aj_o)
                        (ICons ?base (ICons ?rsc (INil)))))
                    (= ?base (Op (Iota (MMul (MIter) ?e2) ?b_range) (INil)))
                    (= ?sorted (Op (Scatter ?sc_ds ?sc_dst ?sc_is ?sc_istr ?sc_ss)
                        (ICons ?zeros (ICons ?adj (ICons ?vals (INil))))))
                    (= ?zeros (Op (Iota (MNum 0) ?z_range) (INil)))
                    (= ?vals (Op (Iota (MIter) ?v_range) (INil)))

                    (= ?out_shape (ECons ?rows (ECons ?e (ENil))))
                    (= ?e ?e2)
                )
                (
                    (let ?kr (Op (KernelStableSortIdx ?out_shape) (ICons ?x (INil))))
                    (union ?sorted ?kr)
                    (set (dtype ?kr) (Int))
                )
                :ruleset kernel_fuse_late
                :name \"kernel stable ranks descending\"
            )",
        )]
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
                out_shape: extract_expr_list(egraph, kind_children[0], list_cache, expr_cache)
                    .unwrap(),
            }) as Box<dyn KernelOp>),
            input_enodes,
        )
    }
}

impl KernelOp for KernelStableSortIdx {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        let rows = self.out_shape[0];
        let e = self.out_shape[1].to_usize().expect("ranks E is static");
        assert!(e <= 1024, "stable ranks kernel supports E <= 1024");

        let vars: FxHashSet<char> = rows.dyn_vars().into_iter().collect();
        let (dyn_defines, _sorted) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };

        let kernel = format!(
            "{dyn_defines}
extern \"C\" {{
    __global__ void stable_sort_idx_k(int *out, const float *x{dyn_dims_param}) {{
        __shared__ float row[{e}];
        long long r = blockIdx.x;
        int j = threadIdx.x;
        const float* xr = x + r * {e};
        if (j < {e}) {{
            row[j] = xr[j];
            // Prefill with 0 (the decomposed chain's scatter dest is a
            // materialized zeros iota). For real inputs the ranks are a
            // permutation and every slot is overwritten; with NaN/garbage
            // inputs (search-time dummy buffers) ranks collide and unwritten
            // slots would otherwise hold stale ints that downstream
            // consumers use as gather/expert indices — an OOB crash.
            out[r * {e} + j] = 0;
        }}
        __syncthreads();
        if (j >= {e}) return;
        float vj = row[j];
        int count = 0;
        for (int i = 0; i < {e}; i++) {{
            float vi = row[i];
            // descending rank with index tie-break (lower index first)
            count += (vi > vj) || (vi == vj && i < j);
        }}
        // ranks are a permutation: element j lands at sorted position count
        out[r * {e} + count] = j;
    }}
}}"
        );

        let (module, func) = if let Some((m, f)) = compile_cache.get(&kernel) {
            (m.clone(), f.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("stable_sort_idx_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        let tpb = e.next_multiple_of(32).min(1024);
        (
            func,
            module,
            kernel,
            (rows, Expression::from(1usize), Expression::from(1usize)),
            (
                Expression::from(tpb),
                Expression::from(1usize),
                Expression::from(1usize),
            ),
            Expression::from(0usize),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * 4
    }

    fn output_dtype(&self) -> DType {
        DType::Int
    }

    fn all_dyn_vars(&self) -> FxHashSet<char> {
        self.out_shape[0].dyn_vars().into_iter().collect()
    }

    fn bytes_loaded(&self) -> Expression {
        self.output_size() * 4
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        let e = self.out_shape[1];
        self.out_shape[0] * e * e * 2
    }

    fn kernel_name(&self) -> &'static str {
        "StableSortIdx"
    }
}
