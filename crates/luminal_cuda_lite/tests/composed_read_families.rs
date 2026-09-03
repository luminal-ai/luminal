//! TRAIN-2B: composed-access READ lowering for the three kernel
//! families that refused it in the field (A100, 2026-08-28): Gather,
//! ScatterFunctional, IndexMapApplyMaterialize. Searches were green and
//! plans built — only the CL device codegen refused folded read
//! operands ("operand N carries a composed access this kernel does not
//! lower"). Everything here is host-side: real searched plans through
//! the CUDA ladder, rendered to CUDA source strings (the
//! codegen_identity discipline), pinned for composed index arithmetic.
//! UNDER THE CORRECTED CONTRACT (2026-08-31) the hop chain is gone: each
//! read operand carries ONE composed layout (the e-graph composed it at
//! view creation) and the kernels lower that layout's own offset
//! expression. The pins below therefore show ONE index expression per
//! folded operand and NO bounds check of any kind — this runtime emits
//! no `__trap()` (same-day ruling; the record is the NO RUNTIME BOUNDS
//! TRAPS note in `luminal_cuda_lite::kernels`), which is why every pin
//! here also calls `assert_no_traps`. Numeric
//! truth for the gather family comes from the reference runtime on the
//! SAME graph (materialize-only by ruling aff22598 — flat kernels),
//! pinned against hand-computed values. WRITE sides stay fail-closed:
//! `codegen_identity::strided::expression_kernel_write_sides_stay_fail_closed`.

use luminal::buffer_tensor_ir::TypedBuffer;
use luminal::bufferize::{BufferIrGraph, BufferNode};
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::implementation_search::ImplementationSearchOptions;
use luminal::prelude::{FxHashMap, NodeIndex};
use luminal_cuda_lite::{kernels, CudaRuntime};

/// The view fixtures' search budget (mirrors `view_admission`):
/// profiling is static bytes-moved, so folds win deterministically
/// under a fixed seed. The seed is per-fixture: fold-vs-materialize
/// spellings of a movement class are cost-TIED (same bytes), so which
/// one the genome elects is sampling — a fixture that needs a specific
/// tied spelling pins the seed that elects it.
fn view_search_options(seed: u64) -> ImplementationSearchOptions {
    ImplementationSearchOptions {
        generations: 4,
        generation_size: 8,
        mutations: 4,
        trials: 1,
        seed,
        search_log: false,
    }
}

/// Load → search on the CUDA runtime; return the best plan.
fn plan_for(
    cx: &Graph,
    inputs: &[(NodeIndex, TypedBuffer)],
    seed: u64,
) -> BufferIrGraph<luminal_cuda_lite::CudaLayout> {
    let mut rt = CudaRuntime::load(cx).expect("cuda load");
    let data: FxHashMap<NodeIndex, TypedBuffer> = inputs.iter().cloned().collect();
    let outcome = rt
        .search(&data, &view_search_options(seed))
        .expect("cuda search");
    assert!(outcome.plans_profiled > 0, "no plans profiled");
    rt.plan().expect("plan loaded").clone()
}

/// Render every compute node through the REAL dispatch path
/// (descriptor ctx → codegen row). Returns
/// (label, launch sources, composed operand slots).
fn rendered(
    plan: &BufferIrGraph<luminal_cuda_lite::CudaLayout>,
) -> Vec<(String, Vec<String>, Vec<usize>)> {
    let mut out = Vec::new();
    for node in plan.dag.node_weights() {
        let BufferNode::Compute {
            op,
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
        let kernel = kernels::codegen_for(op.as_ref())
            .unwrap_or_else(|| panic!("elected op {label} has no codegen row"));
        let ctx = kernels::CodegenCtx::from_descriptors(&label, operand_info, result_info)
            .unwrap_or_else(|e| panic!("descriptor ctx for {label}: {e}"));
        // "Folded" now means: the operand's own carried LAYOUT does not
        // simplify to the bare `i` when read at coordinates that ARE `i`
        // decomposed — i.e. the one lowering had to emit a chain for it.
        // Read off the production lowering itself; there is no predicate
        // to consult any more (ruling 2026-09-01).
        let folded: Vec<usize> = (0..operand_info.len())
            .filter(|&k| {
                !kernels::layout_read_index(
                    "probe",
                    ctx.operand_layout(k),
                    &ctx.operand_dims[k],
                    kernels::Coords::FlatIndex { prefix: "c" },
                )
                .is_ok_and(|(chain, idx)| chain.is_empty() && idx == "i")
            })
            .collect();
        let sources: Vec<String> = (kernel.codegen)(op.as_ref(), &ctx)
            .unwrap_or_else(|e| panic!("codegen for {label}: {e}"))
            .into_iter()
            .map(|l| l.source)
            .collect();
        out.push((label, sources, folded));
    }
    out
}

/// The single node with `label` in the plan, rendered.
fn the_one(
    plan: &BufferIrGraph<luminal_cuda_lite::CudaLayout>,
    label: &str,
) -> (Vec<String>, Vec<usize>) {
    let hits: Vec<_> = rendered(plan)
        .into_iter()
        .filter(|(l, _, _)| l == label)
        .collect();
    assert_eq!(
        hits.len(),
        1,
        "exactly one {label} in the plan:\n{}",
        plan.summary()
    );
    let (_, sources, folded) = hits.into_iter().next().unwrap();
    (sources, folded)
}

fn assert_contains(source: &str, needles: &[&str], what: &str) {
    for needle in needles {
        assert!(
            source.contains(needle),
            "{what}: generated source missing `{needle}`:\n{source}"
        );
    }
}

/// This runtime emits no `__trap()` — not for layout reads, not for the
/// DATA-derived gather/scatter coordinates that used to be checked here
/// (ruling 2026-08-31; the record is the NO RUNTIME BOUNDS TRAPS note in
/// `luminal_cuda_lite::kernels`). Out-of-range indexing is UB at this
/// layer, and this assertion notices if a trap comes back by accident.
fn assert_no_traps(source: &str) {
    assert!(
        !source.contains("__trap"),
        "the CUDA runtime emits no traps:\n{source}"
    );
}

/// Numeric truth from the reference runtime (flat kernels /
/// materialization — it never folds), checked against hand-computed
/// values. The CL side of the differential is textual on CPU; the
/// device half is the A100 pass.
fn reference_values(cx: &Graph, inputs: &[(NodeIndex, TypedBuffer)], out: NodeIndex, want: &[f32]) {
    let reference = luminal_reference::harness::run_reference(cx, inputs);
    let got = reference.get_f32(out).expect("reference output");
    assert_eq!(
        got.as_slice(),
        want,
        "reference numerics diverge from the hand computation"
    );
}

/// EMBEDDING-STYLE GATHER, indices through a fold: data (4,3) gathered
/// at rows broadcast over the out shape — coord0 = rows(2,) expanded to
/// (2,3) (a stride-0 view the search folds), coord1 = a column iota.
/// This is the llama3/qwen3/gemma field shape in miniature; before
/// Train-2B this plan's codegen refused with "Gather: operand 1 carries
/// a composed access this kernel does not lower".
#[test]
fn gather_lowers_a_folded_coordinate_operand() {
    let mut cx = Graph::new();
    let data = cx.tensor((4usize, 3usize), DType::F32);
    let rows = cx.tensor(2usize, DType::Int);
    let cols = cx.iota((2usize, 3usize), |c| c[1]);
    let row_coord = rows.expand_dim(1, 3usize);
    let out = data.gather(&[row_coord, cols]).output();

    let data_vals: Vec<f32> = (0..12).map(|v| v as f32).collect();
    let inputs: Vec<(NodeIndex, TypedBuffer)> =
        vec![(data.id, data_vals.into()), (rows.id, vec![2i32, 0].into())];

    // Numeric truth: out[i][j] = data[rows[i]][j] with rows = [2, 0].
    reference_values(&cx, &inputs, out.id, &[6., 7., 8., 0., 1., 2.]);

    let plan = plan_for(&cx, &inputs, 0);
    let (sources, folded) = the_one(&plan, "GatherGeneric");
    assert_eq!(
        folded,
        vec![1],
        "the broadcast folds into coord0 (operand 1):\n{}",
        plan.summary()
    );
    assert_eq!(sources.len(), 1, "gather is a single launch");
    // The full rendered kernel, pinned: coord0 is read through its own
    // composed layout and coord1's expression simplifies straight back
    // to the flat `coord1[i]`. No bounds checks anywhere (2026-08-31).
    //
    // THE DATA READ MOVED (ruling 2026-09-01), and the move is the whole
    // point of that ruling. It used to be a `flat` accumulator written
    // out by the gather itself — `flat += coord * 3LL; flat += coord *
    // 1LL; ... data[flat]` — a hand-rolled duplicate of what the data
    // layout already says. Now the gathered coordinates are bound as
    // `data_c{axis}` and the DATA LAYOUT states its own address:
    // `data_idx = data_c0 * 3LL + data_c1 * 1LL`. Identical arithmetic,
    // one statement of it instead of two.
    assert_eq!(
        sources[0],
        r#"extern "C" __global__ void k(const float* data, const int* coord0, const int* coord1, float* out, unsigned long long n) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long rem = i;
    long long c1 = (long long)(rem % 3ULL); rem /= 3ULL;
    long long c0 = (long long)(rem % 2ULL); rem /= 2ULL;
    long long coord;
    long long coord0_idx = c0 + 0LL;
    coord = (long long)coord0[coord0_idx];
    long long data_c0 = coord;
    coord = (long long)coord1[i];
    long long data_c1 = coord;
    long long data_idx = data_c0 * 3LL + data_c1 * 1LL;
    out[i] = data[data_idx];
}"#
    );
    // READ THE COORD0 LINE: `c0 + 0LL` is the BROADCAST layout itself —
    // the stride-0 residue on the expanded axis and the bare coordinate
    // on the live one. Three checks used to sit in this kernel: one on
    // `coord0_idx` against the broadcast layout's disclosed span, and
    // two on the gathered `coord` against the data extents (4 and 3).
    // All three are gone; the address arithmetic around them is
    // unchanged, line for line.
}

/// GATHER, data through a fold: the data operand is a permute view, so
/// its chain is evaluated at the GATHERED coordinate values — the
/// gather's own indirection composes ON TOP of the folded chain
/// (`data_c* = coord`, then the hops).
#[test]
fn gather_lowers_a_folded_data_operand() {
    let mut cx = Graph::new();
    let base = cx.tensor((3usize, 4usize), DType::F32);
    let rows = cx.tensor(2usize, DType::Int);
    let cols = cx.iota((2usize, 3usize), |c| c[1]);
    // data = base^T, shape (4,3): data[i][j] = base[j][i].
    let data = base.permute((1, 0));
    let out = data.gather(&[rows.expand_dim(1, 3usize), cols]).output();

    let base_vals: Vec<f32> = (0..12).map(|v| v as f32).collect();
    let inputs: Vec<(NodeIndex, TypedBuffer)> =
        vec![(base.id, base_vals.into()), (rows.id, vec![2i32, 0].into())];

    // data[i][j] = base[j][i] = j*4 + i; rows = [2, 0]:
    // out row 0 = data[2][:] = [2, 6, 10]; out row 1 = data[0][:] = [0, 4, 8].
    reference_values(&cx, &inputs, out.id, &[2., 6., 10., 0., 4., 8.]);

    let plan = plan_for(&cx, &inputs, 0);
    let (sources, folded) = the_one(&plan, "GatherGeneric");
    assert!(
        folded.contains(&0),
        "the permute folds onto the data operand:\n{}",
        plan.summary()
    );
    assert_contains(
        &sources[0],
        &[
            // The gathered coordinates become the layout's coordinates
            // directly — they were checked against the data VALUE's
            // extents (4,3) here until the 2026-08-31 ruling.
            "long long data_c0 = coord;",
            "long long data_c1 = coord;",
            // The permute's COMPOSED layout, lowered once: data[i][j]
            // lives at base flat j*4 + i.
            "long long data_idx = data_c0 * 1LL + data_c1 * 4LL;",
            "out[i] = data[data_idx];",
        ],
        "gather folded-data",
    );
    assert_no_traps(&sources[0]);
    assert!(
        !sources[0].contains("data[flat]"),
        "no flat data read remains:\n{}",
        sources[0]
    );
}

/// SCATTER, coordinate operand through a fold: init (4,3), src (2,3),
/// coord0 = rows(2,) broadcast to (2,3), coord1 = a column iota — the
/// qwen3_moe field shape ("ScatterFunctional: operand 2 carries a
/// composed access") in miniature. The write address arithmetic is
/// untouched by folding; the injectivity check that used to accompany
/// it (and its `flags` scratch buffer) is gone — see the NO RUNTIME
/// BOUNDS TRAPS note in `luminal_cuda_lite::kernels`.
#[test]
fn scatter_lowers_a_folded_coordinate_operand() {
    let mut cx = Graph::new();
    let init = cx.tensor((4usize, 3usize), DType::F32);
    let src = cx.tensor((2usize, 3usize), DType::F32);
    let rows = cx.tensor(2usize, DType::Int);
    let cols = cx.iota((2usize, 3usize), |c| c[1]);
    let row_coord = rows.expand_dim(1, 3usize);
    let out = init.scatter(&[row_coord, cols], src).output();

    let inputs: Vec<(NodeIndex, TypedBuffer)> = vec![
        (init.id, vec![0.0f32; 12].into()),
        (
            src.id,
            (0..6).map(|v| 10.0 + v as f32).collect::<Vec<f32>>().into(),
        ),
        (rows.id, vec![2i32, 0].into()),
    ];

    // out = zeros(4,3); out[2][:] = src[0][:] = [10,11,12];
    // out[0][:] = src[1][:] = [13,14,15].
    reference_values(
        &cx,
        &inputs,
        out.id,
        &[13., 14., 15., 0., 0., 0., 10., 11., 12., 0., 0., 0.],
    );

    let plan = plan_for(&cx, &inputs, 0);
    let (sources, folded) = the_one(&plan, "ScatterFunctionalGeneric");
    assert_eq!(
        folded,
        vec![2],
        "the broadcast folds into coord0 (operand 2):\n{}",
        plan.summary()
    );
    assert_eq!(sources.len(), 2, "scatter is the two-launch sequence");
    // Launch 1 (init copy) has no folded operand here: byte-identical
    // to the flat template.
    assert_contains(
        &sources[0],
        &["if (i < n) out[i] = init[i];"],
        "scatter copy launch",
    );
    // Launch 2: coord0 read through its layout at the SRC coordinates;
    // the write address arithmetic is untouched by the fold.
    assert_contains(
        &sources[1],
        &[
            // src-coordinate prelude over (2,3)
            "long long c1 = (long long)(rem % 3ULL); rem /= 3ULL;",
            "long long c0 = (long long)(rem % 2ULL); rem /= 2ULL;",
            // the broadcast LAYOUT, lowered once
            "long long coord0_idx = c0 + 0LL;",
            "coord = (long long)coord0[coord0_idx];",
            // coord1 stays flat
            "coord = (long long)coord1[i];",
            "flat += coord * 3LL;",
            "flat += coord * 1LL;",
            "out[flat] = src[i];",
        ],
        "scatter write launch",
    );
    // The write is now UNCHECKED: no coordinate range check, and no
    // `atomicExch(&flags[flat], 1u)` injectivity check — so no `flags`
    // parameter either. The kernel signature ends `..., float* out,
    // unsigned long long n`.
    assert_no_traps(&sources[1]);
    for source in &sources {
        assert!(
            !source.contains("flags"),
            "the flags scratch buffer went with its only reader:\n{source}"
        );
    }
}

/// SCATTER with every read-side operand folded: init a permute view,
/// src a slice view, coord0 a broadcast view. All three fold; the
/// write side stays direct.
#[test]
fn scatter_lowers_all_read_side_folds() {
    let mut cx = Graph::new();
    let init_base = cx.tensor((3usize, 4usize), DType::F32);
    let src_base = cx.tensor((4usize, 3usize), DType::F32);
    let rows = cx.tensor(2usize, DType::Int);
    let cols = cx.iota((2usize, 3usize), |c| c[1]);
    let init = init_base.permute((1, 0)); // (4,3), init[i][j] = init_base[j][i]
    let src = src_base.slice((1..3, ..)); // (2,3), src[i][j] = src_base[i+1][j]
    let out = init
        .scatter(&[rows.expand_dim(1, 3usize), cols], src)
        .output();

    let init_vals: Vec<f32> = (0..12).map(|v| 100.0 + v as f32).collect();
    let src_vals: Vec<f32> = (0..12).map(|v| v as f32).collect();
    let inputs: Vec<(NodeIndex, TypedBuffer)> = vec![
        (init_base.id, init_vals.into()),
        (src_base.id, src_vals.into()),
        (rows.id, vec![3i32, 1].into()),
    ];

    // init^T rows: row i = [100+i, 104+i, 108+i]; src rows 1..3 of
    // src_base: [3,4,5], [6,7,8]. rows = [3,1]: out[3] = [3,4,5],
    // out[1] = [6,7,8], others = init.
    reference_values(
        &cx,
        &inputs,
        out.id,
        &[
            100., 104., 108., // init row 0
            6., 7., 8., // src row 1
            102., 106., 110., // init row 2
            3., 4., 5., // src row 0
        ],
    );

    // Seed 1: the genome that folds all three read-side movements
    // (cost-tied spellings; see view_search_options).
    let plan = plan_for(&cx, &inputs, 1);
    let (sources, folded) = the_one(&plan, "ScatterFunctionalGeneric");
    assert_eq!(
        folded,
        vec![0, 1, 2],
        "init, src, and coord0 all read through folds:\n{}",
        plan.summary()
    );
    // Launch 1: init read through its PERMUTE LAYOUT at dest coordinates
    // — init[i][j] = init_base[j][i], i.e. base flat j*4 + i.
    assert_contains(
        &sources[0],
        &[
            "long long init_idx = c0 * 1LL + c1 * 4LL;",
            "out[i] = init[init_idx];",
        ],
        "scatter all-folds copy launch",
    );
    // Launch 2: src through its SLICE layout (+1 row = the `+ 3LL`),
    // coord0 through its broadcast layout; write side untouched.
    assert_contains(
        &sources[1],
        &[
            "long long coord0_idx = c0 + 0LL;",
            "coord = (long long)coord0[coord0_idx];",
            "long long src_idx = (((c0 * 3LL) + 3LL) + c1);",
            "out[flat] = src[src_idx];",
        ],
        "scatter all-folds write launch",
    );
    // Three composed reads, three raw index computations, zero checks.
    for source in &sources {
        assert_no_traps(source);
        assert!(
            !source.contains("flags"),
            "no injectivity scratch:\n{source}"
        );
    }
}

/// MATERIALIZE with a folded input: a view chain folded onto the
/// materialize's parent operand — the whisper field shape
/// ("IndexMapApplyMaterialize: operand 0 carries a composed access")
/// in miniature. The op's own map lands on the input VALUE's
/// coordinates; the folded chain composes ON TOP down to the residence.
#[test]
fn materialize_lowers_a_folded_input_operand() {
    let mut cx = Graph::new();
    let x = cx.tensor((2usize, 3usize), DType::F32);
    // A pure movement chain into a pinned output: the planner must
    // land the result in the caller's dense buffer, so one movement
    // materializes — and the other folds onto its input operand.
    let out = x.permute((1, 0)).slice((0..2, ..)).output();

    let inputs: Vec<(NodeIndex, TypedBuffer)> =
        vec![(x.id, (0..6).map(|v| v as f32).collect::<Vec<f32>>().into())];

    // x^T rows 0..2 of (3,2): [[0,3],[1,4]].
    reference_values(&cx, &inputs, out.id, &[0., 3., 1., 4.]);

    // Materialize is COST-TIED with copy+fold here (same bytes moved), so
    // which one a searched genome elects is sampling luck — a hardcoded
    // lucky seed died of estate perturbation (the 2026-09-01
    // write-capability guard shifted the landscape without touching this
    // op's electability, which was verified against the saturated
    // e-graph: the materialize node is minted, with the RM out the guard
    // requires). So this fixture stopped gambling: the LOWERING — its
    // actual subject — is pinned by driving the op's codegen directly
    // through descriptors, exactly the plan node's shape: operand 0 the
    // permute's COMPOSED view layout over the parent domain (3,2),
    // operand 1 the DPS dest, entries = the slice's map.
    let sources: Vec<String> = {
        use luminal::layouts::{
            BitWidthTerm, IntExprTerm, MirrorLayout, ShapeTerm, StridedElementLayout,
        };
        use luminal_cuda_lite::layouts::CudaLayout;

        let dims =
            |extents: &[i64]| ShapeTerm(extents.iter().map(|&e| IntExprTerm::Lit(e)).collect());
        let coord = |axis_from_end: i64| IntExprTerm::Coord { axis_from_end };
        // x^T seen at the parent VALUE's coordinates (3,2): x is (2,3)
        // row-major, so x^T[a][b] = x flat b*3 + a — the chain
        // [c_last, c_first*3] over domain (3,2).
        let composed = CudaLayout {
            mirror: MirrorLayout::Strided(StridedElementLayout {
                shape: dims(&[3, 2]),
                // Coord counts FROM THE END: p0 (first axis) is
                // coord(1) at rank 2. x^T[p0][p1] = x flat p0*1 + p1*3.
                chain: vec![
                    IntExprTerm::Mul(Box::new(coord(1)), Box::new(IntExprTerm::Lit(1))),
                    IntExprTerm::Mul(Box::new(coord(0)), Box::new(IntExprTerm::Lit(3))),
                ],
                width: BitWidthTerm(32),
            }),
            dtype: Some(luminal::dtype::PlanDtype::F32),
        };
        let rm = |extents: &[i64]| CudaLayout {
            mirror: MirrorLayout::RightMajor(luminal::layouts::RightMajorContiguousElementLayout {
                shape: dims(extents),
                width: BitWidthTerm(32),
            }),
            dtype: Some(luminal::dtype::PlanDtype::F32),
        };
        let slot = |layout: CudaLayout| luminal::bufferize::SlotDescriptor {
            value: luminal::prelude::egraph_serialize::ClassId::from("probe"),
            buffer: luminal::bufferize::BufferId::Allocated(0),
            layout,
        };
        // The slice map: out (2,2) -> parent (3,2) coords [c0, c1].
        let op = luminal_cuda_lite::ops::index_map_apply_materialize::IndexMapApplyMaterializeDps {
            // IotaExpr::Coord also counts FROM THE END: parent_c0 = out
            // c0 is Coord(1) at rank 2 (the slice map is the identity
            // into the first two rows).
            entries: Some(vec![
                luminal::index_expr::IotaExpr::Coord(1),
                luminal::index_expr::IotaExpr::Coord(0),
            ]),
        };
        let ctx = kernels::CodegenCtx::from_descriptors(
            "IndexMapApplyMaterialize",
            &[slot(composed), slot(rm(&[2, 2]))],
            &[slot(rm(&[2, 2]))],
        )
        .expect("descriptor ctx builds");
        (kernels::codegen_for(&op).expect("codegen row").codegen)(&op, &ctx)
            .expect("materialize codegen")
            .into_iter()
            .map(|k| k.source)
            .collect()
    };
    assert_eq!(sources.len(), 1, "materialize is a single launch");
    assert_contains(
        &sources[0],
        &[
            // The op's own map lands on the input VALUE's coordinates
            // (each was checked against the value extents 3 and 2 until
            // the 2026-08-31 ruling)...
            "long long parent_c0 = idx;",
            "long long parent_c1 = idx;",
            // ...and the folded permute's COMPOSED LAYOUT is evaluated at
            // THOSE coordinates (composition on top), once: x^T[a][b] is
            // x flat b*3 + a.
            "long long parent_idx = (parent_c0 * 1LL) + (parent_c1 * 3LL);",
            "out[i] = parent[parent_idx];",
        ],
        "materialize folded input",
    );
    assert_no_traps(&sources[0]);
    assert!(
        !sources[0].contains("pflat"),
        "no flat parent read remains:\n{}",
        sources[0]
    );
}
