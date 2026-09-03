//! M4 PHASE 5 ACCEPTANCE (CPU side): the view op is ELECTABLE on
//! CUDA-lite — real searched plans fold movement to producer redirects
//! and consumers read through the composed access.
//!
//! Mirrors `plan_smoke` on view-heavy fixtures, through the REAL
//! CudaRuntime ladder (load → search under the CUDA allow list → plan
//! inspection). Everything here is device-free: electing and folding a
//! view is planner work; only the read-through happens on the device
//! (the device differentials pin that half).
//!
//! Assertion discipline:
//!  * movement that folds is PINNED as zero materialize nodes — the op
//!    whose whole purpose is materializing an index map
//!    (`IndexMapApplyMaterialize`, label = IR identity) must not appear;
//!  * NO unfolded-view compute nodes — re-checked here by the same
//!    effect-predicate shape the plan validator uses (the validator in
//!    `luminal::bufferize` stays the fence; this keeps the acceptance
//!    test honest if the fence ever moves);
//!  * buffer/copy counts are PINNED per fixture (regression tripwires
//!    for the folded shape);
//!  * consumers. operand descriptors must CARRY A LAYOUT WHOSE READ
//!    DOES NOT SIMPLIFY TO THE IDENTITY —
//!    the view's own composed layout as the e-graph minted it — checked
//!    by EVALUATING that layout to a flat parent element index and
//!    comparing against the hand-computed map. (The hop chain is retired:
//!    corrected contract, 2026-08-31. The e-graph composes views at view
//!    creation; the decoded `L` IS the read path, and how it is spelled
//!    is the e-graph's business.)

use luminal::buffer_tensor_ir::TypedBuffer;
use luminal::bufferize::{BufferIrGraph, BufferNode};
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::implementation_search::ImplementationSearchOptions;
use luminal::prelude::{FxHashMap, NodeIndex};
use luminal_cuda_lite::CudaRuntime;

/// Search budget for the view fixtures: profiling is static (bytes
/// moved), so generations are cheap — enough sampling that the
/// all-views plan is reliably in the profiled set, seeded for
/// deterministic pins.
fn view_search_options() -> ImplementationSearchOptions {
    ImplementationSearchOptions {
        generations: 4,
        generation_size: 8,
        mutations: 4,
        trials: 1,
        seed: 0,
        search_log: false,
    }
}

/// Load → search on the CUDA runtime; return the best plan.
fn plan_for(
    cx: &Graph,
    inputs: &[(NodeIndex, TypedBuffer)],
) -> BufferIrGraph<luminal_cuda_lite::CudaLayout> {
    let mut rt = CudaRuntime::load(cx).expect("cuda load");
    let data: FxHashMap<NodeIndex, TypedBuffer> = inputs.iter().cloned().collect();
    let outcome = rt
        .search(&data, &view_search_options())
        .expect("cuda search");
    assert!(outcome.plans_profiled > 0, "no plans profiled");
    rt.plan().expect("plan loaded").clone()
}

/// The plan-shape audit shared by every fixture. Returns
/// (compute_count, copy_count, buffer_count, folded slots) — a "folded
/// slot" being an operand whose carried layout does NOT reduce to the
/// identity read over its own domain.
type FoldedSlot = (String, usize, luminal_cuda_lite::CudaLayout);
fn audit(
    plan: &BufferIrGraph<luminal_cuda_lite::CudaLayout>,
) -> (usize, usize, usize, Vec<FoldedSlot>) {
    let mut computes = 0usize;
    let mut copies = 0usize;
    let mut composed = Vec::new();
    for node in plan.dag.node_weights() {
        match node {
            BufferNode::BufferCopy { .. } => copies += 1,
            BufferNode::Compute {
                op,
                reads,
                writes,
                ties,
                operand_info,
                result_info,
            } => {
                let label = op.label();
                if label == "BufferAlloc" || label == "BufferFree" {
                    continue;
                }
                computes += 1;

                // ZERO materialize nodes for foldable movement: every
                // fixture's movement is within the parsed expression
                // subset, so the materializing spelling must lose to
                // the fold. (Label = IR identity, house policy.)
                assert_ne!(
                    label,
                    "IndexMapApplyMaterialize",
                    "foldable movement was materialized:\n{}",
                    plan.summary()
                );

                // NO unfolded-view compute nodes — the same
                // effect-predicate shape `validate_plan` fences on.
                let derives = |result: usize| ties.iter().any(|(_, r)| *r == result);
                let view_shaped = !reads.is_empty()
                    && !writes.is_empty()
                    && (0..reads.len()).all(|o| !op.operand_reads_memory(o))
                    && (0..writes.len()).all(|r| !op.result_writes_memory(r) && derives(r));
                assert!(!view_shaped, "unfolded view ({label}) reached the plan");

                // Every kernel-bearing elected op has a codegen row.
                assert!(
                    luminal_cuda_lite::kernels::codegen_for(op.as_ref()).is_some(),
                    "elected op {label} has no codegen row"
                );

                for (slot, info) in operand_info.iter().enumerate() {
                    let dims = info
                        .layout
                        .mirror
                        .literal_extents()
                        .expect("elected slot layouts are literal in these fixtures");
                    // Ask the PRODUCTION read path what it emits: a read
                    // whose expression simplifies to the bare `i` needs
                    // no chain and is the flat read. An unlowerable
                    // layout is certainly not one.
                    let flat = luminal_cuda_lite::kernels::layout_read_index(
                        "probe",
                        &info.layout,
                        &dims,
                        luminal_cuda_lite::kernels::Coords::FlatIndex { prefix: "c" },
                    )
                    .is_ok_and(|(chain, idx)| chain.is_empty() && idx == "i");
                    if !flat {
                        composed.push((label.to_string(), slot, info.layout.clone()));
                    }
                }
                for info in result_info {
                    let dims = info
                        .layout
                        .mirror
                        .literal_extents()
                        .expect("elected slot layouts are literal in these fixtures");
                    // THE WRITE-CAPABILITY CONSTRAINT'S REGRESSION
                    // TEST. The backend does not check this (ruling
                    // 2026-09-01); the constraint lives in egglog —
                    // every codegen'd kernel's match rule fires only on
                    // a right-major-contiguous out class
                    // (ops/*/match_functional.egg). If this assertion
                    // ever fires, that guard has a hole, and the
                    // consequence is silent corruption (kernels write
                    // out[i] unconditionally), so treat a failure here
                    // as a wrong-bytes bug, not a test nit.
                    let flat = luminal_cuda_lite::kernels::layout_read_index(
                        "probe",
                        &info.layout,
                        &dims,
                        luminal_cuda_lite::kernels::Coords::FlatIndex { prefix: "c" },
                    )
                    .is_ok_and(|(chain, idx)| chain.is_empty() && idx == "i");
                    assert!(
                        flat,
                        "{label}: a compute RESULT is produced by the node, never read \
                         through a fold — every kernel writes out[i], so its elected \
                         layout must BE the flat index over its dims"
                    );
                }
            }
            _ => {}
        }
    }
    (computes, copies, plan.buffers.len(), composed)
}

/// Evaluate a mirror term at concrete coordinates (front-indexed;
/// `Coord{axis_from_end}` reads `coords[rank-1-axis_from_end]`). The
/// fixtures' layouts use only the affine subset.
fn eval_term(expr: &luminal::layouts::IntExprTerm, coords: &[usize]) -> i64 {
    use luminal::layouts::IntExprTerm as T;
    let rank = coords.len();
    match expr {
        T::Lit(v) => *v,
        T::Coord { axis_from_end } => {
            let axis = usize::try_from(*axis_from_end).expect("non-negative axis");
            assert!(axis < rank, "coordinate axis {axis} out of rank {rank}");
            coords[rank - 1 - axis] as i64
        }
        T::Add(a, b) => eval_term(a, coords) + eval_term(b, coords),
        T::Mul(a, b) => eval_term(a, coords) * eval_term(b, coords),
        T::TruncDiv(a, b) => eval_term(a, coords) / eval_term(b, coords),
        T::TruncRem(a, b) => eval_term(a, coords) % eval_term(b, coords),
        other => panic!("fixture layouts use no {other:?}"),
    }
}

/// THE READ PATH, evaluated: the slot's own carried layout at one
/// out-coordinate, down to the FLAT ELEMENT INDEX into the residence's
/// bytes. This replaces the hop-chain walk — there are no intermediate
/// parents any more, so there is one answer, not a chain of coordinate
/// frames. (Honesty note carried from the kernels module: with the chain
/// gone, the only bounds fence is the final index against the layout's
/// span where the constructor discloses one.)
fn flat_index(layout: &luminal_cuda_lite::CudaLayout, out_coord: &[usize]) -> i64 {
    use luminal::layouts::MirrorLayout as M;
    let flat = match &layout.mirror {
        M::RightMajor(rm) => {
            let extents = rm.shape.0.iter().map(|e| eval_term(e, &[]) as usize);
            out_coord
                .iter()
                .zip(extents)
                .fold(0usize, |acc, (&c, d)| acc * d + c) as i64
        }
        M::LeftMajor(lm) => {
            let extents: Vec<usize> = lm
                .shape
                .0
                .iter()
                .map(|e| eval_term(e, &[]) as usize)
                .collect();
            let mut stride = 1usize;
            let mut acc = 0usize;
            for (&c, &d) in out_coord.iter().zip(&extents) {
                acc += c * stride;
                stride *= d;
            }
            acc as i64
        }
        M::Strided(st) => st.chain.iter().map(|s| eval_term(s, out_coord)).sum(),
        M::ElementOffset(eo) => eval_term(&eo.offset, out_coord),
        M::BitOffset(bo) => {
            let bits = eval_term(&bo.offset, out_coord);
            assert_eq!(bits % bo.width.0, 0, "bit offset is element-aligned");
            bits / bo.width.0
        }
    };
    assert!(flat >= 0, "negative element index {flat}");
    flat
}

/// TRANSPOSE CONSUMER: x(2,3) permuted then multiplied. The searched
/// plan must fold the permute and hand the mul a swap map.
#[test]
fn transpose_consumer_folds_and_carries_the_swap_map() {
    let mut cx = Graph::new();
    let x = cx.tensor((2usize, 3usize), DType::F32);
    let c = cx.tensor((3usize, 2usize), DType::F32);
    let _out = (x.permute((1, 0)) * c).output();

    let plan = plan_for(
        &cx,
        &[
            (x.id, vec![1.0f32, 2., 3., 4., 5., 6.].into()),
            (c.id, vec![1.0f32; 6].into()),
        ],
    );
    let (computes, copies, buffers, composed) = audit(&plan);
    // One real kernel (the mul), no copies, three buffers (x, c, out).
    assert_eq!(computes, 1, "plan shape drifted:\n{}", plan.summary());
    assert_eq!(copies, 0, "plan shape drifted:\n{}", plan.summary());
    assert_eq!(buffers, 3, "plan shape drifted:\n{}", plan.summary());
    assert!(
        !composed.is_empty(),
        "no operand carries a folded layout:\n{}",
        plan.summary()
    );
    let (label, slot, layout) = &composed[0];
    assert_eq!(label, "MulFunctionalGeneric");
    // The layout's DOMAIN is the view's shape (3,2) — the value's own
    // extents, which is exactly why no `dims` field is needed.
    assert_eq!(layout.mirror.literal_extents(), Some(vec![3, 2]));
    for i in 0..3usize {
        for j in 0..2usize {
            // Parent x is (2,3) row-major; the transpose's (i,j) is
            // parent (j,i), flat j*3 + i.
            assert_eq!(
                flat_index(layout, &[i, j]),
                (j * 3 + i) as i64,
                "transpose: mul operand {slot} out ({i},{j}) must read parent flat {}",
                j * 3 + i
            );
        }
    }
}

/// SLICE CONSUMER: rows 1..3 of a (4,6), multiplied. Fold + offset map.
#[test]
fn slice_consumer_folds_and_carries_the_offset_map() {
    let mut cx = Graph::new();
    let x = cx.tensor((4usize, 6usize), DType::F32);
    let c = cx.tensor((2usize, 6usize), DType::F32);
    let _out = (x.slice((1..3, ..)) * c).output();

    let plan = plan_for(
        &cx,
        &[
            (x.id, (0..24).map(|v| v as f32).collect::<Vec<f32>>().into()),
            (c.id, vec![1.0f32; 12].into()),
        ],
    );
    let (computes, copies, buffers, composed) = audit(&plan);
    assert_eq!(computes, 1, "plan shape drifted:\n{}", plan.summary());
    assert_eq!(copies, 0, "plan shape drifted:\n{}", plan.summary());
    assert_eq!(buffers, 3, "plan shape drifted:\n{}", plan.summary());
    assert!(
        !composed.is_empty(),
        "no operand carries a folded layout:\n{}",
        plan.summary()
    );
    let (_, _, layout) = &composed[0];
    assert_eq!(layout.mirror.literal_extents(), Some(vec![2, 6]));
    for i in 0..2usize {
        for j in 0..6usize {
            // Parent x is (4,6) row-major; rows 1..3, so out (i,j) is
            // parent (i+1, j), flat (i+1)*6 + j.
            assert_eq!(
                flat_index(layout, &[i, j]),
                ((i + 1) * 6 + j) as i64,
                "slice: out ({i},{j}) must read parent flat {}",
                (i + 1) * 6 + j
            );
        }
    }
}

/// BROADCAST CONSUMER: a (3,) row broadcast over (2,3), multiplied.
/// Views read through non-injective maps legally (stride-0 axis).
#[test]
fn broadcast_consumer_folds_and_carries_the_stride0_map() {
    let mut cx = Graph::new();
    let x = cx.tensor(3usize, DType::F32);
    let c = cx.tensor((2usize, 3usize), DType::F32);
    let _out = (x.expand_dim(0, 2) * c).output();

    let plan = plan_for(
        &cx,
        &[
            (x.id, vec![1.0f32, 2., 3.].into()),
            (c.id, vec![1.0f32; 6].into()),
        ],
    );
    let (computes, copies, buffers, composed) = audit(&plan);
    assert_eq!(computes, 1, "plan shape drifted:\n{}", plan.summary());
    assert_eq!(copies, 0, "plan shape drifted:\n{}", plan.summary());
    assert_eq!(buffers, 3, "plan shape drifted:\n{}", plan.summary());
    assert!(
        !composed.is_empty(),
        "no operand carries a folded layout:\n{}",
        plan.summary()
    );
    let (_, _, layout) = &composed[0];
    assert_eq!(layout.mirror.literal_extents(), Some(vec![2, 3]));
    for i in 0..2usize {
        for j in 0..3usize {
            // Parent x is (3,) — the broadcast axis is stride-0, so every
            // i reads the same parent element j.
            assert_eq!(
                flat_index(layout, &[i, j]),
                j as i64,
                "broadcast: out ({i},{j}) must read parent flat {j} for every i"
            );
        }
    }
}

/// CHAINED-MATMUL-SHAPED: (a·b)·c through the decomposed frontend
/// spelling (expand/permute movement + mul + sum at both stages). All
/// movement is foldable, so the plan is exactly the four kernels —
/// two muls, two reduces — with zero materializes and zero copies.
#[test]
fn chained_matmul_folds_all_movement() {
    let mut cx = Graph::new();
    let a = cx.tensor((2usize, 3usize), DType::F32);
    let b = cx.tensor((3usize, 4usize), DType::F32);
    let c = cx.tensor((4usize, 2usize), DType::F32);
    let _out = a.matmul(b).matmul(c).output();

    let plan = plan_for(
        &cx,
        &[
            (a.id, vec![1.0f32; 6].into()),
            (b.id, vec![1.0f32; 12].into()),
            (c.id, vec![1.0f32; 8].into()),
        ],
    );
    let (computes, copies, buffers, composed) = audit(&plan);
    // 2 broadcast-muls + 2 reduces; inputs a,b,c + the four kernel
    // results (out is the last reduce's destination) = 7 buffers.
    assert_eq!(computes, 4, "plan shape drifted:\n{}", plan.summary());
    assert_eq!(copies, 0, "plan shape drifted:\n{}", plan.summary());
    assert_eq!(buffers, 7, "plan shape drifted:\n{}", plan.summary());
    // Both muls read at least one operand through a composed access
    // (the expand_dim broadcasts and the rhs permute+expand).
    let mul_slots = composed
        .iter()
        .filter(|(label, _, _)| label == "MulFunctionalGeneric")
        .count();
    assert!(
        mul_slots >= 2,
        "expected both broadcast-muls to read through folds, got {mul_slots}:\n{}",
        plan.summary()
    );
}
