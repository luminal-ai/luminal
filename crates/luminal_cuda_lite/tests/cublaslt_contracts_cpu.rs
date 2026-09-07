//! Train 3, Item 5 — the CPU-runnable contract tests: the executor-owned
//! validation and call planning for the cuBLASLt host call, no device
//! required. The device-gated halves live in `tests/cublaslt_contracts.rs`.

use luminal::dtype::PlanDtype;
use luminal::egglog_utils::eclass::EgglogConstructor;
use luminal::layouts::{
    BitWidthTerm, ElementOffsetExpressionLayout, IntExprTerm, LeftMajorContiguousElementLayout,
    RightMajorContiguousElementLayout, ShapeTerm, StridedElementLayout,
};
use luminal::prelude::egraph_serialize::ClassId;
use luminal_cuda_lite::layouts::DecodedLayout;
use luminal_cuda_lite::ops::cublaslt::exec::{
    CSource, LtDesc, bind_destination, plan_call, plan_call_from_spec, validate_ld_bounds,
};
use luminal_cuda_lite::ops::cublaslt::{CuDim, CuEpilogue, CublasLt, CublasLtForm, LtMatmulSpec};

fn cid(s: &str) -> ClassId {
    ClassId::from(s)
}

/// A hand-built canonical spec: m x n = k-contracted call, contiguous
/// COL readings (ld = rows — the SPEC side keeps the frozen estate's
/// COL disclosure; the bridge re-expresses them as ROW descriptors,
/// see `exec.rs`'s ROW CONVENTION), no decoration.
fn base_spec(m: i64, n: i64, k: i64) -> LtMatmulSpec {
    LtMatmulSpec {
        form: CublasLtForm::Base,
        m: CuDim::Literal(m),
        n: CuDim::Literal(n),
        k: CuDim::Literal(k),
        trans_a: false,
        trans_b: false,
        lda: CuDim::Literal(m),
        ldb: CuDim::Literal(k),
        ldc: CuDim::Literal(m),
        ldd: CuDim::Literal(m),
        order_col: true,
        has_c: false,
        has_bias: false,
        epilogue: CuEpilogue::Default,
        logical_a: cid("a"),
        logical_b: cid("b"),
        logical_out: cid("out"),
        logical_site_out: cid("site_out"),
        desc_a_layout_tensor: cid("a_lt"),
        desc_b_layout_tensor: cid("b_lt"),
        c_tensor: None,
        bias_tensor: None,
        desc_a_buffer: None,
        desc_b_buffer: None,
        d_buffer: None,
    }
}

// ---------------------------------------------------------------------------
// Contract 4: the ld bounds validator — OUR check, because the library's
// own ld check is self-consistency only and VACUOUS at rows == 1.
// ---------------------------------------------------------------------------

#[test]
fn ld_bounds_accepts_contiguous_row_layouts() {
    // 4x3 ROW-contiguous: ld = 3 (row pitch), needs 3*3+3 = 12 elements.
    validate_ld_bounds("A", &LtDesc::row(4, 3, 3), 12).expect("exact fit");
    // Padded: ld = 6 over the same view needs 6*3+3 = 21.
    validate_ld_bounds("A", &LtDesc::row(4, 3, 6), 21).expect("padded fit");
    validate_ld_bounds("A", &LtDesc::row(4, 3, 6), 64).expect("slack");
}

#[test]
fn ld_bounds_rejects_the_rows_one_vacuous_case() {
    // THE load-bearing case (verified hardware finding): at rows == 1
    // the LIBRARY accepts any ld — its check is vacuous (in ROW order
    // a single row never dereferences ld) — so a too-small buffer
    // would be read out of bounds without a word. OUR check must
    // reject: 1x8 needs cols = 8 elements regardless of ld; the
    // buffer holds 4.
    let err = validate_ld_bounds("A", &LtDesc::row(1, 8, 1), 4)
        .expect_err("rows==1 with a short buffer must be refused");
    let msg = format!("{err:#}");
    assert!(msg.contains("refused BEFORE dispatch"), "{msg}");
    assert!(
        msg.contains("vacuous"),
        "the refusal must name the vacuous library check: {msg}"
    );
    // And the same descriptor over an adequate buffer passes.
    validate_ld_bounds("A", &LtDesc::row(1, 8, 1), 8).expect("adequate");
}

#[test]
fn ld_bounds_rejects_short_buffers_and_degenerate_geometry() {
    // ld too large for the buffer.
    validate_ld_bounds("B", &LtDesc::row(4, 3, 8), 12)
        .expect_err("ld 8 over 12 elements (needs 8*3+3 = 27)");
    // Zero/negative ld and empty geometry are refused outright.
    validate_ld_bounds("B", &LtDesc::row(4, 3, 0), 64).expect_err("ld 0");
    validate_ld_bounds("B", &LtDesc::row(0, 3, 1), 64).expect_err("rows 0");
    validate_ld_bounds("B", &LtDesc::row(4, 0, 4), 64).expect_err("cols 0");
}

#[test]
fn ld_bounds_gate_runs_for_every_descriptor_at_plan_validation() {
    let call = plan_call_from_spec(&base_spec(4, 3, 5)).expect("plan");
    // Operand element counts in Lit order [a, b]; dest = D.
    // (ROW bridge: A = 5x4 ld 4 -> 20, B = 3x5 ld 5 -> 15, D = 4x3
    // ld 3 -> 12 — numerically the same fits as the old COL pins.)
    call.validate_against(&[20, 15], 12)
        .expect("all descriptors fit");
    call.validate_against(&[19, 15], 12)
        .expect_err("A one element short");
    call.validate_against(&[20, 14], 12)
        .expect_err("B one element short");
    call.validate_against(&[20, 15], 11)
        .expect_err("D one element short");
    call.validate_against(&[20], 12)
        .expect_err("operand count != Lit arity");
}

// ---------------------------------------------------------------------------
// Contract 3: descriptor construction — the C descriptor ALWAYS exists;
// the no-C forms alias D (a valid Cdesc mirroring D — Cdesc=NULL is the
// segfault the hardware campaign found).
// ---------------------------------------------------------------------------

#[test]
fn no_c_forms_always_carry_a_valid_cdesc_aliasing_d() {
    let call = plan_call_from_spec(&base_spec(4, 3, 5)).expect("plan");
    assert_eq!(call.c_source, CSource::AliasD);
    assert_eq!(
        call.c, call.d,
        "the aliased Cdesc mirrors the D descriptor exactly"
    );
    assert!(
        !call.beta_is_one,
        "beta = 0.0f on the no-C forms: C is never read"
    );
    // D is the EXECUTOR's dense row-major dest: ROW m x n, ld = n —
    // NEVER the spec's ldd (which describes the claimed e-graph layout
    // over the recorder's buffer; consuming it was the orientation bug).
    assert_eq!(call.d, LtDesc::row(4, 3, 3));
}

#[test]
fn c_fold_forms_read_c_from_operand_two_with_structural_beta_one() {
    let mut spec = base_spec(4, 3, 5);
    spec.form = CublasLtForm::Accumulate;
    spec.has_c = true;
    spec.c_tensor = Some(cid("c_lt"));
    let call = plan_call_from_spec(&spec).expect("plan");
    assert_eq!(
        call.c_source,
        CSource::Operand(2),
        "contract order [a, b, c]"
    );
    assert!(
        call.beta_is_one,
        "beta = 1.0f is STRUCTURAL on the C-fold forms"
    );
    assert_eq!(call.c, call.d, "C rides the D layout by rule guard");
}

fn bias_spec(form: CublasLtForm) -> LtMatmulSpec {
    let mut spec = base_spec(4, 3, 5);
    spec.form = form;
    spec.has_c = form.has_c();
    spec.has_bias = true;
    if form.has_c() {
        spec.c_tensor = Some(cid("c_lt"));
    }
    spec.bias_tensor = Some(cid("bias_lt"));
    spec.epilogue = CuEpilogue::Bias;
    spec
}

/// RULING 2026-09-01: the bias forms are no longer refused at plan time.
/// The estate's bias decorators require a LeftMajor D (the premise
/// `(= ?inner_L (LeftMajorContiguousElementLayoutLit ?ishape ?d_bits2))`
/// in `egg/cublaslt_marker_decorate.egg`), so a planned bias form arrives
/// with a left-major election that `bind_destination` resolves to
/// CUBLASLT_ORDER_COL — the order the A100 library accepts for
/// BIAS/RELU_BIAS. `plan_call` therefore SUCCEEDS on a bias spec, and the
/// bias/order check is the TRIPWIRE at the end of `bind_destination`,
/// where the D order is known.
#[test]
fn bias_epilogue_forms_plan_and_bind_under_a_col_d() {
    for form in [CublasLtForm::Bias, CublasLtForm::AccumulateBias] {
        let call = plan_call_from_spec(&bias_spec(form))
            .expect("bias forms plan: the unconditional refusal is gone");
        assert_eq!(
            call.bias_operand,
            Some(if form.has_c() { 3 } else { 2 }),
            "contract order [a, b, c?, bias]"
        );
        assert_eq!(call.beta_is_one, form.has_c());
        // The plan's election for the sibling destination is LeftMajor
        // over the call frame [m, n] = [4, 3] -> COL, ld = m.
        let mut bound = call.clone();
        bind_destination(&mut bound, &left_major(&[4, 3]), "pin")
            .expect("a LeftMajor destination binds a bias form");
        assert_eq!(bound.d, LtDesc::col(4, 3, 4), "{form:?}: COL D, ld = m");
        assert_eq!(bound.c, bound.d, "{form:?}: C rides D's frame");
        // The bound frame fits the destination buffer exactly (COL reach =
        // ld*(cols-1) + rows = 4*2 + 4 = 12) and the bias buffer holds m.
        let mut elems = vec![20usize, 15];
        if form.has_c() {
            elems.push(12);
        }
        elems.push(4);
        bound
            .validate_against(&elems, 12)
            .expect("the bound bias call passes the pre-dispatch gate");
    }
}

/// THE TRIPWIRE: a bias form whose destination election is RIGHT-major
/// (ROW D) is unreachable from the estate — the decorators require
/// LeftMajor — and `bind_destination` refuses it loudly, naming the
/// measured library finding, BEFORE any descriptor is built.
#[test]
fn bias_epilogue_forms_with_a_row_d_trip_the_unreachable_fence() {
    for form in [CublasLtForm::Bias, CublasLtForm::AccumulateBias] {
        let mut call = plan_call_from_spec(&bias_spec(form)).expect("plan");
        let err = bind_destination(&mut call, &right_major(&[4, 3]), "pin")
            .expect_err("a ROW-order D under a bias form must be refused");
        let msg = format!("{err:#}");
        assert!(msg.contains("unreachable"), "{form:?}: {msg}");
        assert!(
            msg.contains("bias decorators require a LeftMajor D"),
            "{form:?}: the refusal must name the estate premise: {msg}"
        );
        assert!(
            msg.contains("RightMajorContiguousElementLayoutLit"),
            "{form:?}: the refusal must print what the class DOES hold: {msg}"
        );
        assert!(msg.contains("refused BEFORE dispatch"), "{form:?}: {msg}");
    }
    // The default forms are untouched by the fence: a ROW D binds.
    for form in [CublasLtForm::Base, CublasLtForm::Accumulate] {
        let mut spec = base_spec(4, 3, 5);
        spec.form = form;
        spec.has_c = form.has_c();
        if form.has_c() {
            spec.c_tensor = Some(cid("c_lt"));
        }
        let mut call = plan_call_from_spec(&spec).expect("plan");
        bind_destination(&mut call, &right_major(&[4, 3]), "pin")
            .expect("default-epilogue forms dispatch under either order");
        assert_eq!(call.d, LtDesc::row(4, 3, 3));
    }
}

#[test]
fn row_bridge_flips_the_spec_col_readings() {
    // THE ROW RE-EXPRESSION: a spec COL `r x c / ld` reading of the
    // operand bytes is the ROW `c x r / ld` reading of the transposed
    // matrix, so the bridge swaps dims and FLIPS the transpose op; the
    // spec's ld carries over verbatim.
    //
    // Spec N/N (COL A' = m x k = 4x5 ld 4; COL B' = k x n = 5x3 ld 5)
    // => ROW A' = 5x4 ld 4 at T, ROW B' = 3x5 ld 5 at T.
    let call = plan_call_from_spec(&base_spec(4, 3, 5)).expect("plan");
    assert!(call.trans_a && call.trans_b, "N/N spec => T/T ROW call");
    assert_eq!(call.a, LtDesc::row(5, 4, 4));
    assert_eq!(call.b, LtDesc::row(3, 5, 5));

    // Spec trans_a: COL A' stored [k, m] = 5x4 ld 5 presented as
    // op(A') = m x k => ROW A' = 4x5 ld 5 at N.
    let mut spec = base_spec(4, 3, 5);
    spec.trans_a = true;
    spec.lda = CuDim::Literal(5); // contiguous COL: ld = rows' = k
    let call = plan_call_from_spec(&spec).expect("plan");
    assert_eq!(call.a, LtDesc::row(4, 5, 5));
    assert_eq!(call.b, LtDesc::row(3, 5, 5));
    assert!(!call.trans_a && call.trans_b);
}

// ---------------------------------------------------------------------------
// Contract 2 (scalar scope): there is NO runtime scalar channel. The
// call carries no alpha at all and beta only as the structural
// `beta_is_one` bit — a compile-time property of `LtCall`'s type (no
// f32 field exists to smuggle a runtime scalar through); these tests
// pin the structural derivation per form.
// ---------------------------------------------------------------------------

#[test]
fn beta_is_structural_per_form_and_nothing_else() {
    // All four forms plan (ruling 2026-09-01: the bias forms dispatch
    // under a COL D — see `bias_epilogue_forms_plan_and_bind_under_a_col_d`).
    for form in CublasLtForm::ALL {
        let mut spec = base_spec(4, 3, 5);
        spec.form = form;
        spec.has_c = form.has_c();
        spec.has_bias = form.has_bias();
        if form.has_c() {
            spec.c_tensor = Some(cid("c_lt"));
        }
        if form.has_bias() {
            spec.bias_tensor = Some(cid("bias_lt"));
            spec.epilogue = CuEpilogue::Bias;
        }
        let call = plan_call_from_spec(&spec).expect("plan");
        assert_eq!(
            call.beta_is_one,
            form.has_c(),
            "beta is a function of the FORM alone (the C-fold decorator), never data"
        );
    }
}

// ---------------------------------------------------------------------------
// Loud bails: symbolic geometry and missing specs refuse before any
// descriptor is built.
// ---------------------------------------------------------------------------

#[test]
fn symbolic_geometry_is_a_loud_pre_dispatch_refusal() {
    let mut spec = base_spec(4, 3, 5);
    spec.k = CuDim::Symbolic(cid("k_class"));
    let err = plan_call_from_spec(&spec).expect_err("symbolic k");
    assert!(format!("{err:#}").contains("SYMBOLIC"), "{err:#}");
}

#[test]
fn an_elected_op_without_a_parsed_spec_refuses() {
    let op = CublasLt {
        form: CublasLtForm::Base,
        spec: None,
    };
    let err = plan_call(&op).expect_err("no spec");
    assert!(
        format!("{err:#}").contains("no parsed LtMatmulSpec"),
        "{err:#}"
    );
}

// ===========================================================================
// THE PLAN/CALL-FRAME COHERENCE FENCE — `exec::bind_destination`.
//
// WHY THESE PINS EXIST, ON THE CPU TIER. The destination-frame
// regression (Option B, 2026-08-31) reached the main line because the
// ONLY test that could see it was device-gated, so the landing could
// not run it: the marker's transpose-sandwich elects a LEFT-major
// destination layout for the sibling site, the executor wrote dense
// ROW-major anyway, and the product came out exact in transposed byte
// order. Everything about that agreement is decidable WITHOUT a GPU —
// it is a question about a hand-built `LtCall` and a hand-built
// layout — so it is pinned here, on the spin tier, where the next
// occurrence is caught before anyone reaches an A100.
//
// CLASSIFICATION (the taxonomy on `exec::bind_destination`): these are
// COHERENCE fences, not e-graph re-checks. No rule has ever seen an
// `LtCall`; the bridge invents it. They are not disposable.
// ===========================================================================

fn shape(dims: &[i64]) -> ShapeTerm {
    ShapeTerm(dims.iter().map(|&d| IntExprTerm::Lit(d)).collect())
}

fn right_major(dims: &[i64]) -> DecodedLayout {
    DecodedLayout::of(
        RightMajorContiguousElementLayout {
            shape: shape(dims),
            width: BitWidthTerm(32),
        },
        Some(PlanDtype::F32),
    )
}

fn left_major(dims: &[i64]) -> DecodedLayout {
    DecodedLayout::of(
        LeftMajorContiguousElementLayout {
            shape: shape(dims),
            width: BitWidthTerm(32),
        },
        Some(PlanDtype::F32),
    )
}

/// THE DEGENERATE-EXTENT CLASS as the e-graph really has it: at
/// `m == 1` or `n == 1` the two contiguous index maps over `[m, n]` are
/// the SAME function, so BOTH literals land in one class and the decoded
/// layout carries both spellings.
fn both_orders(dims: &[i64]) -> DecodedLayout {
    DecodedLayout::of_spellings(
        vec![
            RightMajorContiguousElementLayout {
                shape: shape(dims),
                width: BitWidthTerm(32),
            }
            .erase(),
            LeftMajorContiguousElementLayout {
                shape: shape(dims),
                width: BitWidthTerm(32),
            }
            .erase(),
        ],
        Some(PlanDtype::F32),
    )
}

#[test]
fn dest_frame_binds_a_right_major_election_as_row_order() {
    // The ordinary case: the elected destination is dense row-major over
    // the call's own [m, n] frame, so D is ROW with ld = n — the frame
    // `plan_call` already defaults to, now CONFIRMED against the plan
    // rather than assumed.
    let mut call = plan_call_from_spec(&base_spec(4, 3, 5)).expect("plan");
    bind_destination(&mut call, &right_major(&[4, 3]), "pin").expect("row-major destination");
    assert_eq!(call.d, LtDesc::row(4, 3, 3));
    assert_eq!(call.c, call.d, "C rides D's frame by rule guard");
}

#[test]
fn dest_frame_binds_a_left_major_election_as_col_order() {
    // THE REGRESSION PIN, in the exact shape the A100 dumped for the
    // 4x8x3 matmul. The marker elects the transpose-sandwich SIBLING:
    // D' = B^T A^T = out^T, so the call frame is [m, n] = [3, 4] and the
    // e-graph elects for that sibling value the layout that makes the
    // ORIGINAL out[4, 3] right-major over the same bytes — LeftMajor[3, 4],
    // element (i, j) at i + 3j.
    //
    // That is CUBLASLT_ORDER_COL with ld = m = 3, and nothing else.
    // Writing ROW here (the old hardcoded convention) put an exact
    // product down in transposed byte order: element 1 of the disclosed
    // out read out(1, 0) instead of out(0, 1).
    let mut call = plan_call_from_spec(&base_spec(3, 4, 8)).expect("plan");
    assert_eq!(call.d, LtDesc::row(3, 4, 4), "the spec-only default is ROW");
    bind_destination(&mut call, &left_major(&[3, 4]), "pin").expect("left-major destination");
    assert_eq!(
        call.d,
        LtDesc::col(3, 4, 3),
        "a left-major election IS a COL descriptor"
    );
    assert_eq!(call.c, call.d);
    // And the bound frame covers the sibling's 12-element buffer exactly:
    // COL reach = ld*(cols-1) + rows = 3*3 + 3 = 12.
    call.validate_against(&[24, 32], 12)
        .expect("the bound frame fits the dest buffer");
}

#[test]
fn dest_frame_refuses_a_permuted_frame() {
    // The check correction 4 deleted, restored and CPU-pinned: the
    // plan's destination spans [n, m] while the executor's call frame is
    // [m, n]. Extents alone catch this one; it is the weaker half of the
    // fence and it still must bite.
    let mut call = plan_call_from_spec(&base_spec(4, 3, 5)).expect("plan");
    let err = bind_destination(&mut call, &right_major(&[3, 4]), "pin")
        .expect_err("a permuted destination frame must be refused");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("DIVERGED"),
        "the refusal must name the divergence: {msg}"
    );
    assert!(
        msg.contains("[m, n] = [4, 3]"),
        "the refusal must print the call frame: {msg}"
    );
}

#[test]
fn dest_frame_refuses_a_rank_mismatch() {
    let mut call = plan_call_from_spec(&base_spec(4, 3, 5)).expect("plan");
    let err = bind_destination(&mut call, &right_major(&[2, 2, 3]), "pin")
        .expect_err("a rank-3 destination has no matmul frame");
    assert!(format!("{err:#}").contains("DIVERGED"), "{err:#}");
}

#[test]
fn dest_frame_refuses_layouts_this_backend_cannot_write() {
    // CAPABILITY refusal, the host-call mirror of the codegen path's
    // destination refusal: cuBLASLt has exactly two matrix orders,
    // so a strided or offset-expression destination is not writable by
    // this route. Loud, never wrong bytes.
    let strided = DecodedLayout::of(
        StridedElementLayout {
            shape: shape(&[4, 3]),
            chain: vec![
                IntExprTerm::Coord { axis_from_end: 1 },
                IntExprTerm::Coord { axis_from_end: 0 },
            ],
            width: BitWidthTerm(32),
        },
        Some(PlanDtype::F32),
    );
    let mut call = plan_call_from_spec(&base_spec(4, 3, 5)).expect("plan");
    let err = bind_destination(&mut call, &strided, "pin").expect_err("strided dest");
    let msg = format!("{err:#}");
    assert!(msg.contains("StridedElementLayoutLit"), "{msg}");
    assert!(
        msg.contains("CAPABILITY refusal"),
        "the refusal must classify itself: {msg}"
    );

    let offset = DecodedLayout::of(
        ElementOffsetExpressionLayout {
            offset: IntExprTerm::Coord { axis_from_end: 0 },
            shape: shape(&[4, 3]),
            width: BitWidthTerm(32),
        },
        Some(PlanDtype::F32),
    );
    let mut call = plan_call_from_spec(&base_spec(4, 3, 5)).expect("plan");
    let err = bind_destination(&mut call, &offset, "pin").expect_err("offset dest");
    assert!(
        format!("{err:#}").contains("ElementOffsetExpressionLayoutLit"),
        "{err:#}"
    );
}

/// `bias_spec` at an arbitrary frame, with the spec's COL readings kept
/// CONSISTENT with it (`base_spec`'s convention: lda = m, ldb = k,
/// ldc = ldd = m) so `validate_against` exercises real geometry.
fn bias_spec_at(form: CublasLtForm, m: i64, n: i64, k: i64) -> LtMatmulSpec {
    let mut spec = bias_spec(form);
    spec.m = CuDim::Literal(m);
    spec.n = CuDim::Literal(n);
    spec.k = CuDim::Literal(k);
    spec.lda = CuDim::Literal(m);
    spec.ldb = CuDim::Literal(k);
    spec.ldc = CuDim::Literal(m);
    spec.ldd = CuDim::Literal(m);
    spec
}

/// THE DEGENERATE-EXTENT COINCIDENCE (whisper on the A100, 2026-09-04).
/// At `n == 1` the right-major and left-major contiguous index maps over
/// `[m, n]` are the SAME function — `r*n + c` and `c*m + r` both address
/// `r` when the collapsed axis's coordinate can only be 0 — so the
/// e-graph puts BOTH spellings in ONE class. A preference-ordered
/// decoder handed the bridge the right-major spelling of the very class
/// whose LeftMajor spelling the estate's bias decorator matched, the
/// fence read that one spelling, and refusing it killed every whisper
/// genome at device prepare:
///
///   a bias form (LayoutTensorOpCublasLtAccumulateBias) reached the
///   executor with a Row-order D descriptor (384x1 ld 1)
///
/// (whisper tiny.en decodes ONE token, so the recorder site is
/// `x[1, 384] @ w[384, 384] + b[384]` and the sandwich sibling's call
/// frame is `m = 384`, `n = 1`.) THE FIX IS THE CLASS, NOT THE EXTENT:
/// `bind_destination` asks the destination class for the LeftMajor
/// spelling, and a class that carries both answers yes. The destination
/// below is that class — the two spellings the e-graph really holds —
/// and there is no degenerate-extent arm anywhere.
#[test]
fn degenerate_extent_binds_a_bias_form_col_in_both_orientations() {
    // n == 1, THE WHISPER SHAPE: the sibling frame is [m, n] = [384, 1].
    // ROW would be 384x1 ld 1; COL is 384x1 ld 384; both address exactly
    // the elements 0..384 of the destination, in the same order.
    for form in [CublasLtForm::Bias, CublasLtForm::AccumulateBias] {
        let spec = bias_spec_at(form, 384, 1, 384);
        let mut call = plan_call_from_spec(&spec).expect("plan");
        assert_eq!(
            call.d,
            LtDesc::row(384, 1, 1),
            "{form:?}: the spec-only default is still ROW (ld = n)"
        );
        bind_destination(&mut call, &both_orders(&[384, 1]), "pin").expect(
            "the destination class holds the LeftMajor spelling the decorators \
             require, so the bias form binds — no extent test involved",
        );
        assert_eq!(call.d, LtDesc::col(384, 1, 384), "{form:?}: COL D, ld = m");
        assert_eq!(call.c, call.d, "{form:?}: C rides D's frame");
        // Same 384 elements either way: COL reach = ld*(cols-1) + rows
        // = 384*0 + 384 = 384, exactly the ROW reach 1*383 + 1.
        // A holds m*k, B holds k*n, C (fold forms) holds m*n, bias m.
        let mut elems = vec![384 * 384usize, 384];
        if form.has_c() {
            elems.push(384);
        }
        elems.push(384);
        call.validate_against(&elems, 384)
            .expect("the bound frame fits the 384-element destination");
    }

    // m == 1, the symmetric degeneracy: [1, 384]. ROW would be
    // 1x384 ld 384; COL is 1x384 ld 1; same elements again.
    for form in [CublasLtForm::Bias, CublasLtForm::AccumulateBias] {
        let spec = bias_spec_at(form, 1, 384, 384);
        let mut call = plan_call_from_spec(&spec).expect("plan");
        assert_eq!(call.d, LtDesc::row(1, 384, 384), "the ROW default");
        bind_destination(&mut call, &both_orders(&[1, 384]), "pin")
            .expect("the m == 1 coincidence binds too");
        assert_eq!(
            call.d,
            LtDesc::col(1, 384, 1),
            "{form:?}: COL D, ld = m = 1"
        );
        assert_eq!(call.c, call.d);
        // A holds m*k = 384, B holds k*n, C (fold forms) holds m*n = 384,
        // bias m = 1.
        let mut elems = vec![384usize, 384 * 384];
        if form.has_c() {
            elems.push(384);
        }
        elems.push(1);
        call.validate_against(&elems, 384)
            .expect("the bound frame fits the 384-element destination");
    }
}

/// THE FENCE IS INTACT, AND IT IS ABOUT THE CLASS, NOT THE EXTENT. Two
/// destinations, same degenerate `[384, 1]` frame:
///
///  * a class holding ONLY the right-major spelling is REFUSED under a
///    bias form — the decorators' premise is not satisfied, whatever the
///    extents are;
///  * a class holding BOTH (the test above) binds COL.
///
/// So the fix cannot be widened by accident into "degenerate extents are
/// special": there is no extent test to widen.
#[test]
fn a_right_major_only_d_trips_the_bias_fence_at_any_extent() {
    for form in [CublasLtForm::Bias, CublasLtForm::AccumulateBias] {
        for (label, dims, m, n) in [
            // 384x2: one step off the whisper shape.
            ("non-degenerate", [384i64, 2], 384i64, 2i64),
            // THE NEGATIVE THE DESIGN MAKES TRUE: the degenerate frame
            // itself, but with a class that holds only the right-major
            // spelling. The fix is the class carrying both, never the
            // extent.
            ("degenerate, right-major only", [384, 1], 384, 1),
        ] {
            let spec = bias_spec_at(form, m, n, 384);
            let mut call = plan_call_from_spec(&spec).expect("plan");
            let err = bind_destination(&mut call, &right_major(&dims), "pin")
                .expect_err("a D class with no LeftMajor spelling must be refused");
            let msg = format!("{err:#}");
            assert!(msg.contains("unreachable"), "{form:?}/{label}: {msg}");
            assert!(
                msg.contains("bias decorators require a LeftMajor D"),
                "{form:?}/{label}: {msg}"
            );
            assert!(
                msg.contains("RightMajorContiguousElementLayoutLit"),
                "{form:?}/{label}: the refusal names what the class holds: {msg}"
            );
            assert!(
                msg.contains("refused BEFORE dispatch"),
                "{form:?}/{label}: {msg}"
            );
        }
    }
    // BIAS-FREE, and now the call site's own preference decides: a class
    // holding only the right-major spelling stays ROW...
    let mut call = plan_call_from_spec(&base_spec(384, 1, 384)).expect("plan");
    bind_destination(&mut call, &right_major(&[384, 1]), "pin").expect("no bias, no fence");
    assert_eq!(
        call.d,
        LtDesc::row(384, 1, 1),
        "a right-major-only D binds ROW whether or not an extent is 1"
    );
    // ...and a class holding BOTH binds COL, because this call site asks
    // for LeftMajor first. Same elements, same order, same reach — COL
    // 384x1 ld 384 reaches 0*384 + r, ROW 384x1 ld 1 reaches r*1 + 0
    // (behaviour change flagged in the design, §8.9).
    let mut call = plan_call_from_spec(&base_spec(384, 1, 384)).expect("plan");
    bind_destination(&mut call, &both_orders(&[384, 1]), "pin").expect("no bias, no fence");
    assert_eq!(
        call.d,
        LtDesc::col(384, 1, 384),
        "both spellings present: this call site prefers LeftMajor -> COL"
    );
}

#[test]
fn dest_frame_refuses_symbolic_extents() {
    let symbolic = DecodedLayout::of(
        RightMajorContiguousElementLayout {
            shape: ShapeTerm(vec![IntExprTerm::Var("s".into()), IntExprTerm::Lit(3)]),
            width: BitWidthTerm(32),
        },
        Some(PlanDtype::F32),
    );
    let mut call = plan_call_from_spec(&base_spec(4, 3, 5)).expect("plan");
    let err = bind_destination(&mut call, &symbolic, "pin").expect_err("symbolic dest");
    assert!(format!("{err:#}").contains("SYMBOLIC"), "{err:#}");
}

// ---------------------------------------------------------------------------
// Contract 4, ORDER-AWARENESS: the ld reach is a function of the
// descriptor's order. The ROW formula applied to a COL descriptor
// UNDERSTATES the reach whenever rows > cols, which is exactly the
// direction that lets an out-of-bounds write through.
// ---------------------------------------------------------------------------

#[test]
fn ld_bounds_reach_is_order_aware() {
    // COL 8x2 ld 8: reach = 8*(2-1) + 8 = 16. The ROW formula would say
    // 8*(8-1) + 2 = 58 (over-strict, refusing a legal call) — and the
    // mirror case below is the dangerous one.
    validate_ld_bounds("D", &LtDesc::col(8, 2, 8), 16).expect("COL exact fit");
    validate_ld_bounds("D", &LtDesc::col(8, 2, 8), 15).expect_err("COL one element short");

    // COL 2x8 ld 2: reach = 2*7 + 2 = 16. The ROW formula would say
    // 2*1 + 8 = 10 and ACCEPT a 10-element buffer — 6 elements of
    // out-of-bounds write, silently.
    validate_ld_bounds("D", &LtDesc::col(2, 8, 2), 16).expect("COL exact fit");
    let err = validate_ld_bounds("D", &LtDesc::col(2, 8, 2), 10)
        .expect_err("the ROW formula would have let this through");
    assert!(format!("{err:#}").contains("needs 16 elements"), "{err:#}");

    // The vacuous-library case has a COL twin: a single COLUMN never
    // dereferences ld in COL order, exactly as a single row does not in
    // ROW order.
    validate_ld_bounds("D", &LtDesc::col(8, 1, 1), 4)
        .expect_err("COL rows==8 cols==1 still needs 8 elements");
}
