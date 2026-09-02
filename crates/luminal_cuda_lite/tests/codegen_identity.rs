//! M4 Phase 3 pin, RESTATED at Phase 5: CUDA codegen strings are
//! IDENTICAL whether `CodegenCtx` geometry comes from the shared buffer
//! table (the pre-Phase-3 device path) or from the plan node's own
//! `SlotDescriptor`s (the Phase-3 device path) — FOR NODES WITHOUT
//! COMPOSED ACCESS. The Phase-3 wording ("no views are electable on
//! real backends") was the zero-behavior premise; Phase 5 flips it
//! deliberately: the view op is now claimed through the plan-transparent
//! class, so view-consumer nodes carry composed access their descriptors
//! know and the buffer table never can. On those nodes the descriptor
//! route MUST diverge (it emits the Phase-4 strided read-through); the
//! divergence itself is pinned below, and the strided string gates +
//! `view_admission` own the read-through's content.
//!
//! Set `CODEGEN_DUMP_DIR` to also write every generated source to disk
//! (used to diff before/after captures across the Phase-3 landing).

use luminal::bufferize::{BufferId, BufferIrGraph, BufferNode};
use luminal::dtype::{DType, PlanDtype};
use luminal::prelude::FxHashMap;
use luminal_cuda_lite::kernels::Coords;
use luminal_cuda_lite::{kernels, CudaRuntime};
use std::collections::HashMap;

/// Does this layout's read, taken at coordinates that ARE `i`
/// decomposed over `dims`, simplify all the way back to the bare `i`?
///
/// There is no such predicate in the runtime any more (ruling
/// 2026-09-01: "delete the whole reads_identity function") — production
/// codegen never asks, it just lowers and emits whatever the expression
/// became. This is a TEST-side observation of that expression, which is
/// exactly what these pins are for: it asks the production lowering
/// [`kernels::layout_read_index`] and reports whether it needed a chain.
/// An unlowerable layout answers `false` — it is certainly not a flat
/// read.
fn reads_flat(layout: &luminal_cuda_lite::layouts::CudaLayout, dims: &[usize]) -> bool {
    kernels::layout_read_index("probe", layout, dims, Coords::FlatIndex { prefix: "c" })
        .is_ok_and(|(chain, idx)| chain.is_empty() && idx == "i")
}

/// The PRE-Phase-3 construction, restated for the corrected contract:
/// per-node geometry looked up in the shared BUFFER table by BufferId —
/// which now means the buffer's own carried layout (the RESIDENT's
/// layout), never a plan `dims`/`dtype` field. That is precisely why the
/// route still diverges on folded operands: the residence's layout is
/// the parent's, and the operand wanted the view's.
///
/// It may now REFUSE, and that is the point. Reading a folded operand's
/// geometry out of the buffer table hands the elementwise template an
/// operand whose extents are not the dest's (the matmul fixture: a
/// `[8,3]` residence under a `[4,3,8]` broadcast). The template's
/// coherence check used to be asked only of operands taking the
/// expression read, so a right-major residence of the wrong shape
/// sailed past it and this route emitted a silently-reinterpreting
/// kernel — a wrong kernel nobody ran, but a wrong kernel. Since the
/// one-read-path ruling (2026-08-31) the check is asked of every
/// operand, so the replication route refuses instead. `Err` here IS
/// divergence, of the loudest kind.
fn sources_via_buffer_table(
    plan: &BufferIrGraph<luminal_cuda_lite::CudaLayout>,
) -> Vec<(String, Result<Vec<String>, String>)> {
    let geometry: HashMap<BufferId, (Vec<usize>, PlanDtype)> = plan
        .buffers
        .iter()
        .map(|(id, buffer)| {
            let dims = buffer
                .layout
                .mirror
                .literal_extents()
                .expect("plan buffer's layout has literal extents");
            (
                id.clone(),
                (
                    dims,
                    buffer.layout.dtype.expect("plan buffer's layout is typed"),
                ),
            )
        })
        .collect();
    let mut out = Vec::new();
    for node in plan.dag.node_weights() {
        let BufferNode::Compute {
            op, reads, writes, ..
        } = node
        else {
            continue;
        };
        let label = op.label().to_string();
        if label == "BufferAlloc" || label == "BufferFree" {
            continue;
        }
        let kernel = kernels::codegen_for(op.as_ref())
            .unwrap_or_else(|| panic!("elected op {label} has no codegen row"));
        let ctx = kernels::CodegenCtx {
            operand_dims: reads.iter().map(|id| geometry[id].0.clone()).collect(),
            operand_dtypes: reads.iter().map(|id| geometry[id].1).collect(),
            dest_dims: writes.iter().map(|id| geometry[id].0.clone()).collect(),
            dest_dtypes: writes.iter().map(|id| geometry[id].1).collect(),
            // PROTOTYPE (Option B): the buffer table's layout is the
            // WRITER's (resident) layout — for a folded operand that is
            // the parent's dense layout, so this route stays flat and
            // the folded divergence below is required exactly as before.
            operand_layouts: reads
                .iter()
                .map(|id| plan.buffers[id].layout.clone())
                .collect(),
        };
        out.push((
            label,
            (kernel.codegen)(op.as_ref(), &ctx)
                .map(|ls| ls.into_iter().map(|l| l.source).collect())
                .map_err(|e| e.to_string()),
        ));
    }
    out
}

fn searched_plan(
    build: impl FnOnce(
        &mut luminal::graph::Graph,
    ) -> FxHashMap<
        luminal::prelude::petgraph::graph::NodeIndex,
        luminal::buffer_tensor_ir::TypedBuffer,
    >,
) -> BufferIrGraph<luminal_cuda_lite::CudaLayout> {
    let mut cx = luminal::graph::Graph::new();
    let data = build(&mut cx);
    let mut rt = CudaRuntime::load(&cx).expect("load");
    let outcome = rt
        .search(&data, &luminal::test_support::harness_search_options())
        .expect("search under the CUDA allow list");
    assert!(outcome.plans_profiled > 0, "no plans profiled");
    rt.plan().expect("plan loaded").clone()
}

fn representative_plans() -> Vec<(&'static str, BufferIrGraph<luminal_cuda_lite::CudaLayout>)> {
    vec![
        (
            "elementwise",
            searched_plan(|cx| {
                let a = cx.tensor((2usize, 3usize), DType::F32);
                let b = cx.tensor((2usize, 3usize), DType::F32);
                let _ = ((a + b) * a).output();
                [
                    (a.id, vec![1.0f32, 2., 3., 4., 5., 6.].into()),
                    (b.id, vec![10.0f32, 20., 30., 40., 50., 60.].into()),
                ]
                .into_iter()
                .collect()
            }),
        ),
        (
            "matmul",
            searched_plan(|cx| {
                let x = cx.tensor((4usize, 8usize), DType::F32);
                let w = cx.tensor((8usize, 3usize), DType::F32);
                let _ = x.matmul(w).output();
                [
                    (x.id, vec![0.5f32; 32].into()),
                    (w.id, vec![0.25f32; 24].into()),
                ]
                .into_iter()
                .collect()
            }),
        ),
        (
            "mul_sum",
            searched_plan(|cx| {
                let a = cx.tensor((3usize, 4usize), DType::F32);
                let b = cx.tensor((3usize, 4usize), DType::F32);
                let _ = (a * b).sum(1).output();
                [
                    (a.id, vec![1.0f32; 12].into()),
                    (b.id, vec![2.0f32; 12].into()),
                ]
                .into_iter()
                .collect()
            }),
        ),
    ]
}

/// The Phase-3 device path: geometry from the node's own descriptors.
/// The third tuple slot records whether the node read through a fold
/// (any operand carrying composed access) — the Phase-5 restatement
/// keys on it.
fn sources_via_descriptors(
    plan: &BufferIrGraph<luminal_cuda_lite::CudaLayout>,
) -> Vec<(String, Vec<String>, bool)> {
    let mut out = Vec::new();
    for node in plan.dag.node_weights() {
        let BufferNode::Compute {
            op,
            reads,
            writes,
            operand_info,
            result_info,
            ..
        } = node
        else {
            continue;
        };
        let label = op.label().to_string();
        if label == "BufferAlloc" || label == "BufferFree" {
            continue;
        }
        assert_eq!(
            operand_info.len(),
            reads.len(),
            "{label}: operand descriptors parallel reads"
        );
        assert_eq!(
            result_info.len(),
            writes.len(),
            "{label}: result descriptors parallel writes"
        );
        // Phase 5: composed access is now LEGAL here — the view op is
        // electable, folded views hand their consumers the access. The
        // old zero-behavior assert (composed_access always None) died
        // with the premise; the caller now pins where divergence from
        // the buffer-table route is required vs forbidden.
        // Option B: the divergence discriminator is the slot LAYOUT —
        // an operand whose read does not simplify to the identity.
        // (A view whose composed layout IS direct would be a flat read
        // on both routes, correctly.)
        let folded = operand_info.iter().any(|slot| {
            let dims = slot
                .layout
                .mirror
                .literal_extents()
                .expect("elected slot layouts are literal in these fixtures");
            // Ask the PRODUCTION read path: an operand whose expression
            // simplifies to the bare `i` emits no chain and is the flat
            // read; anything else (including an unlowerable layout) is a
            // fold.
            !reads_flat(&slot.layout, &dims)
        });
        let kernel = kernels::codegen_for(op.as_ref())
            .unwrap_or_else(|| panic!("elected op {label} has no codegen row"));
        let ctx = kernels::CodegenCtx::from_descriptors(&label, operand_info, result_info)
            .unwrap_or_else(|e| panic!("descriptor ctx for {label}: {e}"));
        let sources: Vec<String> = (kernel.codegen)(op.as_ref(), &ctx)
            .unwrap_or_else(|e| panic!("codegen for {label}: {e}"))
            .into_iter()
            .map(|l| l.source)
            .collect();
        out.push((label, sources, folded));
    }
    out
}

// ---------------------------------------------------------------------------
// M4 Phase 4: strided READS through synthetic ComposedAccess descriptors —
// string-level gates, host-side (no device). Each test builds a CodegenCtx
// through `from_descriptors` (the only codegen path) and asserts the
// generated source contains the exact index expressions — and, since
// the 2026-08-31 ruling, NO bounds checks at all; the flat `a[i]` fast
// path must stay byte-identical.
// ---------------------------------------------------------------------------

mod strided {
    use luminal::buffer_tensor_ir::BufferTensorIrOp;
    use luminal::bufferize::{BufferId, SlotDescriptor};
    use luminal::dtype::PlanDtype;
    use luminal_cuda_lite::{kernels, ops};

    use luminal::layouts::{
        BitWidthTerm, ElementOffsetExpressionLayout, IntExprTerm, MirrorLayout,
        RightMajorContiguousElementLayout, ShapeTerm, StridedElementLayout,
    };
    use luminal_cuda_lite::CudaLayout;

    fn lit(v: i64) -> IntExprTerm {
        IntExprTerm::Lit(v)
    }
    fn coord(axis_from_end: i64) -> IntExprTerm {
        IntExprTerm::Coord { axis_from_end }
    }
    fn mul(a: IntExprTerm, b: IntExprTerm) -> IntExprTerm {
        IntExprTerm::Mul(Box::new(a), Box::new(b))
    }
    fn add(a: IntExprTerm, b: IntExprTerm) -> IntExprTerm {
        IntExprTerm::Add(Box::new(a), Box::new(b))
    }
    fn shape(dims: &[i64]) -> ShapeTerm {
        ShapeTerm(dims.iter().map(|&d| lit(d)).collect())
    }
    fn typed(mirror: MirrorLayout) -> CudaLayout {
        CudaLayout {
            mirror,
            dtype: Some(PlanDtype::F32),
        }
    }
    fn rm_layout(dims: &[i64]) -> CudaLayout {
        typed(MirrorLayout::RightMajor(
            RightMajorContiguousElementLayout {
                shape: shape(dims),
                width: BitWidthTerm(32),
            },
        ))
    }
    fn strided_layout(dims: &[i64], chain: Vec<IntExprTerm>) -> CudaLayout {
        typed(MirrorLayout::Strided(StridedElementLayout {
            shape: shape(dims),
            chain,
            width: BitWidthTerm(32),
        }))
    }
    fn offset_layout(dims: &[i64], offset: IntExprTerm) -> CudaLayout {
        typed(MirrorLayout::ElementOffset(ElementOffsetExpressionLayout {
            offset,
            shape: shape(dims),
            width: BitWidthTerm(32),
        }))
    }

    /// A slot whose layout is the dense row-major read over its dims.
    /// EVERYTHING the codegen needs — extents, dtype, read path — comes
    /// from that one layout: the descriptor has no dims/dtype/hop fields
    /// left to fill (corrected contract, 2026-08-31).
    fn slot(dims: Vec<i64>) -> SlotDescriptor<CudaLayout> {
        slot_l(rm_layout(&dims))
    }

    /// PROTOTYPE (Option B): a slot carrying its OWN elected layout —
    /// the one vocabulary every family reads through.
    fn slot_l(layout: CudaLayout) -> SlotDescriptor<CudaLayout> {
        SlotDescriptor {
            value: luminal::prelude::egraph_serialize::ClassId::from("val$synthetic"),
            buffer: BufferId::Allocated(0),
            layout,
        }
    }

    /// The same slot with a different dtype fact on its carried layout
    /// (a RUNTIME-side field: dtype rides `CudaLayout`, never the plan).
    fn slot_dt(dims: Vec<i64>, dtype: PlanDtype) -> SlotDescriptor<CudaLayout> {
        let mut s = slot(dims);
        s.layout.dtype = Some(dtype);
        s
    }

    /// Generate the single kernel source for `op` with the given
    /// descriptors, through the table row (the real dispatch path).
    fn generate(
        op: &dyn BufferTensorIrOp,
        operand_info: &[SlotDescriptor<CudaLayout>],
        result_info: &[SlotDescriptor<CudaLayout>],
    ) -> String {
        let ctx = kernels::CodegenCtx::from_descriptors(op.label(), operand_info, result_info)
            .expect("descriptor ctx builds");
        let row = kernels::codegen_for(op).expect("codegen row");
        let launches = (row.codegen)(op, &ctx).expect("codegen succeeds");
        assert_eq!(launches.len(), 1, "single-launch op");
        launches.into_iter().next().unwrap().source
    }

    fn assert_contains(source: &str, needles: &[&str]) {
        for needle in needles {
            assert!(
                source.contains(needle),
                "generated source missing `{needle}`:\n{source}"
            );
        }
    }

    /// This runtime emits no `__trap()` (ruling 2026-08-31; the record
    /// is the NO RUNTIME BOUNDS TRAPS note in `luminal_cuda_lite::
    /// kernels`). Traps returning belongs behind a feature flag, and
    /// this assertion is what will notice if one comes back by accident.
    fn assert_no_traps(source: &str) {
        assert!(
            !source.contains("__trap"),
            "the CUDA runtime emits no traps:\n{source}"
        );
    }

    /// Transpose, OPTION B: out [3,2] reading its parent's bytes through
    /// the SLOT'S OWN composed strided layout (shape [3,2], chain
    /// from-end [coord0*3, coord1] — element (i,j) at parent flat
    /// j*3+i), through the Copy row's unary template. NO hop chain is
    /// supplied at all — the layout alone drives the read (the
    /// hop-machinery death demonstration).
    #[test]
    fn transpose_read_indexes_the_parent_at_swapped_coords() {
        let layout = strided_layout(&[3, 2], vec![mul(coord(0), lit(3)), coord(1)]);
        let op = ops::materialize_layout_copy::MaterializeLayoutCopyDps;
        let source = generate(
            &op,
            &[slot_l(layout), slot(vec![3, 2])],
            &[slot(vec![3, 2])],
        );
        assert_contains(
            &source,
            &[
                // out-coordinate prelude over [3,2]
                "long long c1 = (long long)(rem % 2ULL); rem /= 2ULL;",
                "long long c0 = (long long)(rem % 3ULL); rem /= 3ULL;",
                // the layout's offset expression, lowered directly
                "long long a_idx = (c1 * 3LL) + c0;",
                "out[i] = a[a_idx];",
            ],
        );
        assert_no_traps(&source);
        assert!(
            !source.contains("a[i]"),
            "flat read must be rewritten:\n{source}"
        );
        assert!(
            !source.contains("a_h0_0"),
            "Option B: no hop variables — the layout drives the read:\n{source}"
        );
    }

    /// Zero-base slice with pitch > cols: out [4,5] over parent [4,8]
    /// (identity coords, larger row pitch), on one operand of a binary
    /// add — the other operand stays flat `b[i]`.
    #[test]
    fn pitched_slice_read_on_one_binary_operand_keeps_the_other_flat() {
        // OPTION B: the slice value's composed layout — shape [4,5],
        // chain from-end [coord0 (stride 1), coord1*8 (the parent's
        // pitch)]. The other operand keeps its direct layout and stays
        // flat `b[i]`.
        let layout = strided_layout(&[4, 5], vec![coord(0), mul(coord(1), lit(8))]);
        let op = ops::add::AddFunctionalDps;
        let source = generate(
            &op,
            &[slot_l(layout), slot(vec![4, 5]), slot(vec![4, 5])],
            &[slot(vec![4, 5])],
        );
        assert_contains(
            &source,
            &[
                // pitch 8, not the value's 5
                "long long a_idx = c1 + (c0 * 8LL);",
                "out[i] = a[a_idx] + b[i];",
            ],
        );
        assert_no_traps(&source);
    }

    /// Broadcast-shaped map: out [2,3] reading parent [1,3] with a Lit 0
    /// entry on the broadcast axis.
    #[test]
    fn broadcast_read_pins_the_broadcast_axis_to_zero() {
        // OPTION B: the broadcast value's composed layout — shape [2,3],
        // chain from-end [coord0 (stride 1), 0 (the dead axis residue)].
        let layout = strided_layout(&[2, 3], vec![coord(0), lit(0)]);
        let op = ops::materialize_layout_copy::MaterializeLayoutCopyDps;
        let source = generate(
            &op,
            &[slot_l(layout), slot(vec![2, 3])],
            &[slot(vec![2, 3])],
        );
        assert_contains(
            &source,
            &[
                // the dead axis contributes the literal zero residue
                "long long a_idx = c1 + 0LL;",
                "out[i] = a[a_idx];",
            ],
        );
        assert_no_traps(&source);
    }

    /// Two folds, OPTION B: the e-graph composes — the slot carries ONE
    /// layout whose offset expression is the whole composition (here the
    /// synthetic composition of a transpose then a +1-row offset into a
    /// [4,3] parent: (c1+1)*3 + c0), spelled as an offset-EXPRESSION
    /// form. The whole composition is ONE expression: no hop variables,
    /// no intermediate reads, and (ruling 2026-08-31) no bounds check of
    /// any kind — nothing constrains this read at this layer, from above
    /// or below.
    #[test]
    fn composed_offset_form_reads_one_expression() {
        let layout = offset_layout(&[3, 2], add(mul(add(coord(0), lit(1)), lit(3)), coord(1)));
        let op = ops::materialize_layout_copy::MaterializeLayoutCopyDps;
        let source = generate(
            &op,
            &[slot_l(layout), slot(vec![3, 2])],
            &[slot(vec![3, 2])],
        );
        assert_contains(
            &source,
            &[
                "long long a_idx = (((c1 + 1LL) * 3LL) + c0);",
                "out[i] = a[a_idx];",
            ],
        );
        assert_no_traps(&source);
        assert!(
            !source.contains("a_idx <") && !source.contains("a_idx >="),
            "the read index is computed and used, never tested:\n{source}"
        );
        assert!(!source.contains("a_h0_0"), "no hop variables:\n{source}");
    }

    /// Reduce over a transposed input, OPTION B: ReduceSum(axis_from_end
    /// = 0) on a [2,3] value whose SLOT LAYOUT is the transpose
    /// composition over a [3,2] parent (element (c0,c1) at parent flat
    /// c1*2 + c0). The reduced coordinate is the loop variable, and the
    /// read is the layout's own offset expression — no hop chain, and
    /// no bounds checks.
    #[test]
    fn reduce_reads_through_the_slot_layout() {
        // shape [2,3]; from-end coord(0) = c1, coord(1) = c0.
        let layout = strided_layout(&[2, 3], vec![mul(coord(0), lit(2)), coord(1)]);
        let op = ops::reduce_sum::ReduceSumDps { axis: 0 };
        let source = generate(&op, &[slot_l(layout), slot(vec![2])], &[slot(vec![2])]);
        assert_contains(
            &source,
            &[
                // c0 (outside the reduced axis) rebuilt before the loop
                "long long c0 = (long long)(rem % 2ULL); rem /= 2ULL;",
                // the reduced coordinate is the loop variable
                "long long c1 = (long long)r;",
                // the layout's expression, lowered directly
                "long long a_idx = (c1 * 2LL) + c0;",
                "float v = a[a_idx];",
                "acc = acc + v;",
            ],
        );
        assert!(!source.contains("a_h0_0"), "no hop variables:\n{source}");
    }

    /// Cast keeps its conversion around the layout-expression read.
    #[test]
    fn cast_wraps_the_strided_read() {
        // A reversed rank-1 read: chain [-coord + 3] spelled as
        // ((coord0 * -1) + 3) — does not simplify, so it is lowered.
        let layout = strided_layout(&[4], vec![add(mul(coord(0), lit(-1)), lit(3))]);
        let op = ops::cast::CastDps;
        // The dtypes ride the slots' CARRIED LAYOUTS (the runtime's own
        // type), not a descriptor field.
        let operand = slot_l(layout);
        let dest = slot_dt(vec![4], PlanDtype::Int);
        let source = generate(&op, &[operand, dest.clone()], &[dest]);
        assert_contains(&source, &["out[i] = (int)a[a_idx];"]);
    }

    /// OPTION B fail-closed analogues of the old `entries: None` hop
    /// refusals: a layout the lowerer cannot spell numerically bails
    /// loudly — never identity, never a guessed extent.
    ///
    /// THE SEAM HAS MOVED. A SYMBOLIC domain now refuses one step
    /// EARLIER, at `from_descriptors`, because the slot's extents ARE
    /// its layout's domain (there is no dims field to disagree with).
    /// The domain-mismatch refusal survives only where two DIFFERENT
    /// slots' layouts disagree — here an operand whose domain is not the
    /// destination's, which is what the elementwise template reads
    /// against.
    #[test]
    fn unlowerable_layouts_refuse_loudly() {
        let op = ops::materialize_layout_copy::MaterializeLayoutCopyDps;
        // Symbolic extent in the layout's own domain — refused at ctx build.
        let layout = typed(MirrorLayout::Strided(StridedElementLayout {
            shape: ShapeTerm(vec![IntExprTerm::Var("n".to_string()), lit(2)]),
            chain: vec![coord(0), mul(coord(1), lit(2))],
            width: BitWidthTerm(32),
        }));
        let err = kernels::CodegenCtx::from_descriptors(
            "Copy",
            &[slot_l(layout), slot(vec![3, 2])],
            &[slot(vec![3, 2])],
        )
        .expect_err("symbolic layout extents must refuse");
        assert!(
            err.to_string().contains("symbolic layout extents"),
            "got: {err}"
        );
        // An operand layout whose DOMAIN is not the destination's: the
        // template reads at the dest's coordinates, so this is a real
        // incoherence and refuses in the template, never reinterprets.
        //
        // ASKED OF EVERY OPERAND NOW (ruling 2026-08-31). Both spellings
        // below denote the SAME dense [2,3] read against a [3,2] dest,
        // and both must refuse. Before the ruling only the first did:
        // the coherence check lived inside the "non-direct" branch, so a
        // right-major operand of the wrong shape sailed through the flat
        // path and silently reinterpreted 6 elements as [3,2]. That is
        // the spelling-dependence, caught in a second place.
        for (what, layout) in [
            (
                "strided spelling",
                strided_layout(&[2, 3], vec![coord(0), mul(coord(1), lit(3))]),
            ),
            ("right-major spelling", rm_layout(&[2, 3])),
        ] {
            let ctx = kernels::CodegenCtx::from_descriptors(
                "Copy",
                &[slot_l(layout), slot(vec![3, 2])],
                &[slot(vec![3, 2])],
            )
            .expect("ctx builds");
            let err = (kernels::codegen_for(&op).unwrap().codegen)(&op, &ctx)
                .expect_err("a foreign-domain operand must refuse, whatever its spelling");
            assert!(
                err.to_string().contains("differ from dest extents"),
                "{what}: the template names the incoherence: {err}"
            );
        }
    }

    // ===================================================================
    // THREE WRITE-FENCE TESTS WERE DELETED HERE (ruling 2026-09-01).
    //
    //   * result_that_does_not_write_at_the_identity_index_refuses
    //   * dest_operand_that_does_not_write_at_the_identity_index_refuses
    //   * expression_kernel_write_sides_stay_fail_closed
    //
    // Each asserted that a destination whose layout is not the flat
    // index over its dims REFUSES ("strided writes are not lowered";
    // iota's variant, "does not lower"). Austin ruled that fence out of
    // the backend entirely — "we should not have it in the codebase
    // here. delete it" — so their subject no longer exists.
    //
    // They are deleted rather than reworded because the honest
    // restatement of today's behavior would be "a strided destination
    // silently emits a kernel that writes the wrong addresses", and
    // pinning that would pin a BUG as a contract.
    //
    // WHERE THE PROPERTY LIVES NOW: `tests/view_admission.rs` asserts of
    // every compute RESULT in every searched plan that its elected
    // layout IS the flat index. That is the same property, checked where
    // it can actually hold — over real elections — and it is the
    // standing evidence that the unguarded gap is not being hit. When
    // the egglog output-layout constraint lands, that assertion becomes
    // its regression test.
    // ===================================================================

    /// THE BYTE-IDENTITY PIN — unchanged text, changed meaning.
    ///
    /// It was written to prove the flat FAST PATH fired: a
    /// `layout_is_direct` constructor match diverted these operands
    /// around the expression path entirely. That fast path is gone
    /// (ruling 2026-08-31, "there should be no special casing... it
    /// should always go through the expression pathway"). These operands
    /// now go through the SAME expression path as every other read; what
    /// this pin proves today is that the SIMPLIFIER recognizes the dense
    /// case — each read reduces to the identity over the very `i` the
    /// coordinates would have been decomposed from, so no coordinate is
    /// materialized, the prelude is dead, and the body collapses back to
    /// exactly this text.
    ///
    /// If this string ever changes for a dense operand, the simplifier
    /// is INCOMPLETE. Fix the simplifier; do not update this pin.
    #[test]
    fn flat_path_is_byte_identical() {
        let op = ops::add::AddFunctionalDps;
        let source = generate(
            &op,
            &[slot(vec![2, 3]), slot(vec![2, 3]), slot(vec![2, 3])],
            &[slot(vec![2, 3])],
        );
        assert_eq!(
            source,
            r#"extern "C" __global__ void k(const float* a, const float* b, float* out, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}"#
        );
    }

    // =======================================================================
    // THE TWO-SPELLING PROOF (ruling 2026-08-31).
    //
    // The e-graph may hand the decoder ANY spelling of a layout class —
    // all spellings of a class denote one function, and `decode_layout`
    // only states a PREFERENCE among the ones it finds. So the read
    // decision may not be made on a spelling. These tests state one dense
    // function five ways and require the emitted CUDA source to be
    // byte-identical every time.
    // =======================================================================

    /// The dense row-major read of a [2,3] value — strides [3,1] — in
    /// five spellings:
    ///   1. `RightMajor[2,3]`               (structural)
    ///   2. `Strided` chain `[c1, c0*3]`    (dense strides, canonical residues)
    ///   3. `Strided` chain `[c1*1 + 0, c0*3]` — an explicit unit stride
    ///      and a zero residue, the kind of leftover an unnormalized
    ///      composition carries
    ///   4. `ElementOffset` `(c0*6 + c1*2) / 2` — folds to the same by
    ///      exact division, the kind of thing a bit/element conversion
    ///      leaves behind
    ///   5. `BitOffset` `(c0*3 + c1) * 32` at width 32 — the bit form of
    ///      the same read
    ///
    /// Each is first checked to BE the same function by the runtime's
    /// own independent evaluator (`layouts::element_index`, which walks
    /// the mirror structs and knows nothing about codegen), then the
    /// emitted source is compared. Two statements, one answer.
    #[test]
    fn dense_spellings_all_collapse_to_the_flat_read() {
        use luminal::layouts::BitOffsetExpressionLayout;
        let dims = [2usize, 3usize];
        // from-end: coord(0) = c1 (stride 1), coord(1) = c0 (stride 3).
        let spellings: Vec<(&str, CudaLayout)> = vec![
            ("right-major", rm_layout(&[2, 3])),
            (
                "strided, dense chain",
                strided_layout(&[2, 3], vec![coord(0), mul(coord(1), lit(3))]),
            ),
            (
                "strided, unnormalized residues",
                strided_layout(
                    &[2, 3],
                    vec![add(mul(coord(0), lit(1)), lit(0)), mul(coord(1), lit(3))],
                ),
            ),
            (
                "element-offset, exact division",
                offset_layout(
                    &[2, 3],
                    IntExprTerm::TruncDiv(
                        Box::new(add(mul(coord(1), lit(6)), mul(coord(0), lit(2)))),
                        Box::new(lit(2)),
                    ),
                ),
            ),
            (
                "bit-offset at width 32",
                typed(MirrorLayout::BitOffset(BitOffsetExpressionLayout {
                    offset: mul(add(mul(coord(1), lit(3)), coord(0)), lit(32)),
                    shape: shape(&[2, 3]),
                    width: BitWidthTerm(32),
                })),
            ),
        ];

        // (a) INDEPENDENTLY: every spelling really is the same function.
        for (what, layout) in &spellings {
            for c0 in 0..dims[0] {
                for c1 in 0..dims[1] {
                    let got = luminal_cuda_lite::layouts::element_index(layout, &[c0, c1])
                        .unwrap_or_else(|e| panic!("{what} evaluates at ({c0},{c1}): {e}"));
                    assert_eq!(
                        got,
                        c0 * 3 + c1,
                        "{what}: the spellings must denote the SAME function"
                    );
                }
            }
        }

        // (b) THEREFORE: identical emitted source, and it is the flat
        //     read — the simplifier recognizes all five.
        let op = ops::materialize_layout_copy::MaterializeLayoutCopyDps;
        let want = r#"extern "C" __global__ void k(const float* a, float* out, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i];
}"#;
        for (what, layout) in &spellings {
            let source = generate(
                &op,
                &[slot_l(layout.clone()), slot(vec![2, 3])],
                &[slot(vec![2, 3])],
            );
            assert_eq!(
                source, want,
                "{what}: a dense read must emit the flat source, whatever its spelling"
            );
        }
    }

    /// The same rule from the write side: a dense STRIDED destination
    /// is written at the flat index, even though it is not spelled
    /// `RightMajor`. Under the old constructor match this refused —
    /// a capability the backend has, denied on a spelling. (There is no
    /// write FENCE any more, ruling 2026-09-01; what this pins is that
    /// the emitted source is the flat one for a dense destination
    /// however it is spelled.)
    #[test]
    fn a_dense_strided_destination_is_written_at_the_flat_index() {
        let dense_strided = strided_layout(&[2, 3], vec![coord(0), mul(coord(1), lit(3))]);
        let op = ops::materialize_layout_copy::MaterializeLayoutCopyDps;
        let source = generate(
            &op,
            &[slot(vec![2, 3]), slot_l(dense_strided.clone())],
            &[slot_l(dense_strided)],
        );
        assert_eq!(
            source,
            r#"extern "C" __global__ void k(const float* a, float* out, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i];
}"#
        );
    }

    /// The simplification is not credulous: a chain that is
    /// dense-LOOKING but permuted, offset, or scaled must NOT collapse
    /// to `i`. (The transpose/pitch/broadcast/offset cases above are the
    /// same point stated through their emitted expressions; this states
    /// it as a direct verdict, including the extent-1 rule.)
    ///
    /// This once asserted on a `reads_identity` predicate. That predicate
    /// is deleted (ruling 2026-09-01) and the behavior it named now lives
    /// inside the one lowering, so the verdicts are read off
    /// [`kernels::layout_read_index`] itself via [`reads_flat`] — same
    /// cases, aimed at the surviving subject.
    #[test]
    fn only_the_identity_collapses() {
        let cases: Vec<(&str, Vec<usize>, CudaLayout, bool)> = vec![
            (
                "transposed [3,2]",
                vec![3, 2],
                strided_layout(&[3, 2], vec![mul(coord(0), lit(3)), coord(1)]),
                false,
            ),
            (
                "pitched [4,5]",
                vec![4, 5],
                strided_layout(&[4, 5], vec![coord(0), mul(coord(1), lit(8))]),
                false,
            ),
            (
                "broadcast [2,3]",
                vec![2, 3],
                strided_layout(&[2, 3], vec![coord(0), lit(0)]),
                false,
            ),
            (
                "nonzero base [3,2]",
                vec![3, 2],
                offset_layout(&[3, 2], add(mul(add(coord(1), lit(1)), lit(2)), coord(0))),
                false,
            ),
            (
                "inexact division",
                vec![4],
                offset_layout(
                    &[4],
                    IntExprTerm::TruncDiv(Box::new(mul(coord(0), lit(3))), Box::new(lit(2))),
                ),
                false,
            ),
            ("foreign domain", vec![3, 2], rm_layout(&[2, 3]), false),
            // An extent-1 axis pins no coefficient: c is always 0, so
            // every spelling of that axis' contribution is the same
            // function. This is a fact about the function, not a licence.
            (
                "extent-1 axis, stride 0",
                vec![1, 3],
                strided_layout(&[1, 3], vec![coord(0), lit(0)]),
                true,
            ),
            (
                "extent-1 axis, wild stride",
                vec![1, 3],
                strided_layout(&[1, 3], vec![coord(0), mul(coord(1), lit(99))]),
                true,
            ),
            ("rank 0", vec![], offset_layout(&[], lit(0)), true),
            // Left-major and right-major coincide at rank 1.
            (
                "left-major rank 1",
                vec![5],
                typed(MirrorLayout::LeftMajor(
                    luminal::layouts::LeftMajorContiguousElementLayout {
                        shape: shape(&[5]),
                        width: BitWidthTerm(32),
                    },
                )),
                true,
            ),
            // ...and diverge at rank 2, where left-major is a transpose.
            (
                "left-major rank 2",
                vec![2, 3],
                typed(MirrorLayout::LeftMajor(
                    luminal::layouts::LeftMajorContiguousElementLayout {
                        shape: shape(&[2, 3]),
                        width: BitWidthTerm(32),
                    },
                )),
                false,
            ),
        ];
        for (what, dims, layout, want) in cases {
            assert_eq!(
                super::reads_flat(&layout, &dims),
                want,
                "{what}: does the lowered read simplify to the bare `i`?"
            );
        }
    }
}

/// THE LOUD-BAIL CONTRACT, descriptor-side, RESPELLED (corrected
/// contract, 2026-08-31): the descriptor has no `dims`/`dtype` fields
/// to leave empty any more — every numeric a kernel needs comes from
/// the slot's CARRIED LAYOUT. So the refusals move onto the layout: a
/// symbolic domain and a missing dtype fact both bail loudly, never a
/// silent zero-extent kernel and never a guessed representation.
#[test]
fn descriptor_ctx_bails_loudly_on_unusable_layouts() {
    use luminal::bufferize::SlotDescriptor;
    use luminal::layouts::{
        BitWidthTerm, IntExprTerm, MirrorLayout, RightMajorContiguousElementLayout, ShapeTerm,
    };
    use luminal_cuda_lite::CudaLayout;
    let rm = |shape: ShapeTerm| {
        MirrorLayout::RightMajor(RightMajorContiguousElementLayout {
            shape,
            width: BitWidthTerm(32),
        })
    };
    let lit_shape = ShapeTerm(vec![IntExprTerm::Lit(2), IntExprTerm::Lit(3)]);
    let filled = SlotDescriptor {
        value: luminal::prelude::egraph_serialize::ClassId::from("val$x"),
        buffer: luminal::bufferize::BufferId::Allocated(0),
        layout: CudaLayout {
            mirror: rm(lit_shape.clone()),
            dtype: Some(PlanDtype::F32),
        },
    };
    let symbolic = SlotDescriptor {
        layout: CudaLayout {
            mirror: rm(ShapeTerm(vec![
                IntExprTerm::Var("n".to_string()),
                IntExprTerm::Lit(3),
            ])),
            dtype: Some(PlanDtype::F32),
        },
        ..filled.clone()
    };
    let err = kernels::CodegenCtx::from_descriptors(
        "ProbeOp",
        &[symbolic],
        std::slice::from_ref(&filled),
    )
    .expect_err("symbolic layout extents must refuse");
    assert!(
        err.to_string().contains("symbolic layout extents"),
        "got: {err}"
    );
    let untyped = SlotDescriptor {
        layout: CudaLayout {
            mirror: rm(lit_shape),
            dtype: None,
        },
        ..filled.clone()
    };
    let err =
        kernels::CodegenCtx::from_descriptors("ProbeOp", std::slice::from_ref(&filled), &[untyped])
            .expect_err("a missing dtype fact must refuse");
    assert!(
        err.to_string().contains("carries no dtype fact"),
        "got: {err}"
    );
    let ok = kernels::CodegenCtx::from_descriptors(
        "ProbeOp",
        std::slice::from_ref(&filled),
        std::slice::from_ref(&filled),
    )
    .expect("filled descriptors build");
    assert_eq!(ok.operand_dims, vec![vec![2, 3]]);
    assert!(
        reads_flat(ok.operand_layout(0), &ok.operand_dims[0]),
        "a dense layout's read simplifies to the bare `i` — no chain is emitted"
    );
}

/// RE-PINNED ONCE at Phase 5 (view electability). Justification per
/// flip:
///  * NON-FOLDED nodes: the Phase-3 equality pin stands unchanged —
///    descriptor-derived codegen is string-identical to the
///    buffer-table replication.
///  * FOLDED nodes (an operand carries composed access): equality is
///    now IMPOSSIBLE BY DESIGN — the buffer table never knew the
///    folded view's map, which is exactly the Phase-3 bug class the
///    descriptors were built to fix. The pin flips to REQUIRED
///    DIVERGENCE: the descriptor route must emit a different (strided
///    read-through) source than the flat replication. The matmul
///    fixture flips from all-equal to folded (its broadcast/permute
///    movement now folds); elementwise and mul_sum stay all-equal.
#[test]
fn codegen_strings_via_descriptors_match_the_buffer_table() {
    let mut folded_seen = 0usize;
    for (name, plan) in representative_plans() {
        let via_table = sources_via_buffer_table(&plan);
        let via_descriptors = sources_via_descriptors(&plan);
        assert!(
            !via_table.is_empty(),
            "{name}: no compute kernels generated"
        );
        assert_eq!(
            via_table.len(),
            via_descriptors.len(),
            "{name}: both routes generate the same kernel sequence"
        );
        for ((t_label, t_sources), (d_label, d_sources, folded)) in
            via_table.iter().zip(&via_descriptors)
        {
            assert_eq!(
                t_label, d_label,
                "{name}: kernel order agrees between routes"
            );
            if *folded {
                folded_seen += 1;
                // Divergence, in one of its two forms: a DIFFERENT
                // kernel, or (since the one-read-path ruling) a loud
                // refusal — the replication route cannot even state a
                // coherent kernel for a folded operand's residence.
                match t_sources {
                    Err(why) => assert!(
                        why.contains("differ from dest extents"),
                        "{name}/{d_label}: the replication route must refuse for the \
                         stated reason, got: {why}"
                    ),
                    Ok(t) => assert_ne!(
                        t, d_sources,
                        "{name}/{d_label}: a folded operand must change the generated \
                         read (the buffer table cannot know the composed access)"
                    ),
                }
            } else {
                assert_eq!(
                    t_sources
                        .as_ref()
                        .expect("non-folded nodes generate on both routes"),
                    d_sources,
                    "{name}/{d_label}: descriptor-derived codegen must be \
                     string-identical to the buffer table on non-folded nodes"
                );
            }
        }
        if let Ok(dir) = std::env::var("CODEGEN_DUMP_DIR") {
            let dir = std::path::Path::new(&dir);
            std::fs::create_dir_all(dir).expect("dump dir");
            for (i, (label, sources, _)) in via_descriptors.iter().enumerate() {
                for (k, source) in sources.iter().enumerate() {
                    let file = dir.join(format!("{name}_{i:02}_{k}_{label}.cu"));
                    std::fs::write(file, source).expect("dump write");
                }
            }
        }
        let folded_here = via_descriptors.iter().filter(|(_, _, f)| *f).count();
        let refused = via_table.iter().filter(|(_, s)| s.is_err()).count();
        println!(
            "[{name}] {} nodes: {} identical via both paths, {} folded \
             (divergence required; {} of those the replication route refuses outright)",
            via_table.len(),
            via_table.len() - folded_here,
            folded_here,
            refused
        );
    }
    assert!(
        folded_seen > 0,
        "Phase 5: at least one representative plan must fold a view \
         (the matmul fixture's movement is foldable)"
    );
}
