//! The cuBLASLt HOST-CALL contract layer — CPU-side, device-free.
//!
//! Train 3 (registry + host-call dispatch): the four marker contracts
//! execute as ONE host library call (`cublasLtMatmul`), not an NVRTC
//! kernel. This module turns an elected [`CublasLt`] op (its parsed
//! [`LtMatmulSpec`]) into a fully-resolved [`LtCall`] — plain numerics
//! and finite classifications only — and carries the executor-side
//! validation the library itself does NOT provide. Everything here is
//! unit-testable without a device; the cudarc dispatch lives in
//! `device_call` (feature-gated) and consumes an `LtCall` verbatim.
//!
//! THE A100 EXECUTOR CONTRACTS (verified findings; load-bearing):
//! 1. F32-only end to end (inputs, outputs, CUBLAS_COMPUTE_32F).
//! 2. POINTER_MODE_HOST with LITERAL scalars: alpha = 1.0f always
//!    (the marker has no alpha channel); beta is STRUCTURAL — 1.0f on
//!    the C-fold (Accumulate) forms, 0.0f otherwise. There is no
//!    runtime scalar channel: [`LtCall`] has NO alpha/beta float
//!    fields, only [`LtCall::beta_is_one`], and the literals live in
//!    the dispatch site as compile-time constants.
//! 3. C = D aliasing with a VALID Cdesc on the no-C forms
//!    (Cdesc = NULL segfaults): [`LtCall::c`] is NON-OPTIONAL — every
//!    call carries a C descriptor; [`CSource::AliasD`] says "pass the
//!    D pointer" and beta = 0.0f guarantees C is never read.
//! 4. ld semantics: the library's own ld check is self-consistency
//!    only and VACUOUS at rows == 1 (in ROW order a single row never
//!    dereferences ld) — [`validate_ld_bounds`] is the REAL bounds
//!    validation (`ld*(rows-1) + cols <= element count` in ROW order),
//!    asserted loudly at dispatch for A, B, C, and D. Emitted lds are
//!    clamped to `>= 1` at plan time.
//! 5. TF32 is graph-modeled, never a flag: the dispatch sets
//!    CUBLAS_COMPUTE_32F explicitly and `device_call` carries a
//!    startup detector assertion at handle creation.
//!
//! THE ROW CONVENTION (Train-3 orientation fix; measured on the A100
//! with the 4x8x3 dump — see `tests/cublaslt_contracts.rs`):
//! every emitted layout descriptor DECLARES its order explicitly —
//! never the library's COL default, and (since the destination-frame
//! fix below) never a constant at the dispatch site either. The order
//! is a field of [`LtDesc`]. A and B are ROW; C and D are whatever the
//! plan's elected destination layout resolves to. This DECLARES
//! REALITY, in two halves:
//!
//!  * A and B: the marker spec's readings are COL views over the
//!    operand buffers' bytes (frozen estate convention R9/R10). A COL
//!    `r x c / ld` view of a byte range IS the ROW `c x r / ld` view
//!    of the transposed matrix — same bytes, same pitch — so the
//!    bridge re-expresses each operand reading as ROW by swapping the
//!    descriptor dims and FLIPPING the transpose op. The spec's lds
//!    carry over verbatim (a COL view's ld and the underlying
//!    row-major storage's row pitch are the same number, padded
//!    layouts included).
//!  * D (and C, which rides D's layout by rule guard): the PLAN'S
//!    disclosed destination layout is authoritative. What is NOT
//!    authoritative is the spec's ldd/ldc — those describe the CLAIMED
//!    e-graph layout over the RECORDER's out buffer, a buffer the
//!    executor never writes, and consuming them was the orientation
//!    bug: bytes landed COL-major at the spec's pitch while the
//!    disclosure read row-major (element 0 agreed, element 1 did not).
//!    The bridge therefore builds the D FRAME from the call
//!    (`m x n`) and takes its ORDER from the destination value's
//!    elected layout via [`bind_destination`] — ROW `ld = n` for a
//!    right-major election, COL `ld = m` for a left-major one.
//!    Hardcoding ROW here instead was the second orientation failure;
//!    see the note below.
//!
//! The CM-swap alternative (compute D^T with swapped roles under COL
//! defaults) is REJECTED: cuBLASLt's bias epilogue adds bias[i] to
//! row i of the API's D, and the marker's bias contract is per-row of
//! the call's D (length m) — a role swap would silently turn it into
//! a per-column bias.
//!
//! THE BIAS FORMS DISPATCH ONLY UNDER A COL-ORDER D (ruling 2026-09-01,
//! Austin: "adding the layout requirement to the rule is the correct
//! solution"). MEASURED on the A100 (2026-08-28): the library returns
//! CUBLAS_STATUS_NOT_SUPPORTED for CUBLASLT_EPILOGUE_BIAS / RELU_BIAS
//! whenever D is CUBLASLT_ORDER_ROW (any A/B order). The fix is in the
//! ESTATE, not here: the two bias decorators
//! (`egg/cublaslt_marker_decorate.egg`) now carry the premise
//! `(= ?inner_L (LeftMajorContiguousElementLayoutLit ?ishape ?d_bits2))`
//! — the bias form is minted only when the claimed D is provably
//! left-major contiguous over the sibling frame `[n, m]`, which is
//! byte-identical to the recorder's row-major `[m, n]`, puts the
//! per-feature vector on D's rows (the API's only bias axis), and is
//! exactly what [`bind_destination`] resolves to `LtDesc::col`. The
//! executor therefore no longer refuses the bias forms; it carries a
//! TRIPWIRE instead ([`assert_bias_destination_order`]): a bias-bearing
//! call whose D is not COL is unreachable from the estate, and reaching
//! it is a bug to bail on, never a case to handle.
//!
//! THE DESTINATION FRAME IS THE PLAN'S, NOT A CONSTANT (regression fix,
//! 2026-08-31 — see [`bind_destination`]). The paragraph above says
//! "the executor materializes every result DENSE ROW-MAJOR in the
//! value's own dims (the disclosure downstream walks exactly that)".
//! Under Option B that second clause STOPPED BEING TRUE: the plan
//! carries each value's ELECTED layout and every consumer — the
//! codegen read path, the output-slot disclosure, `dense_f32` — reads
//! through it, so "dense row-major" is no longer a law the executor
//! may assume. On the marker's own transpose-sandwich the elected
//! destination layout is LEFT-major: the site that dispatches is the
//! SIBLING (D' = B^T A^T = out^T, an `m' x n'` frame), and the e-graph
//! elects for that sibling value exactly the layout that makes the
//! ORIGINAL out right-major over the same bytes — `LeftMajor[m', n']`.
//! Writing ROW there produced an exact product in transposed byte
//! order: the SAME failure shape as the original orientation bug, one
//! convention further along.
//!
//! So the destination ORDER is resolved against the plan, not assumed:
//! [`LtOrder`] is a field of every [`LtDesc`], `plan_call` emits the
//! spec-only DEFAULT (ROW), and the executor calls [`bind_destination`]
//! with the result slot's elected layout before dispatch.

use anyhow::{Result, bail};

use super::{CuDim, CuEpilogue, CublasLt, CublasLtForm, LtMatmulSpec};

/// A descriptor's storage ORDER — the cuBLASLt
/// `CUBLASLT_MATRIX_LAYOUT_ORDER` attribute, carried EXPLICITLY on
/// every descriptor rather than hardcoded at the dispatch site. Two
/// bugs (Train-3's orientation bug and the Option-B destination-frame
/// regression) both had the same shape: an order convention that lived
/// only in prose while the bytes said otherwise. It is data now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LtOrder {
    /// `CUBLASLT_ORDER_ROW`: element `(r, c)` at `r*ld + c`; `ld` is the
    /// ROW pitch and the descriptor reaches `ld*(rows-1) + cols`.
    Row,
    /// `CUBLASLT_ORDER_COL`: element `(r, c)` at `c*ld + r`; `ld` is the
    /// COLUMN pitch and the descriptor reaches `ld*(cols-1) + rows`.
    Col,
}

/// One descriptor's geometry: `rows x cols` with leading dimension `ld`
/// and its storage [`LtOrder`], all resolved literals (elements, not
/// bytes). A and B are always [`LtOrder::Row`] — the frozen estate's
/// COL readings re-expressed, see the module doc's ROW CONVENTION; C
/// and D carry whatever the plan's elected destination layout says
/// (see [`bind_destination`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LtDesc {
    pub rows: i64,
    pub cols: i64,
    pub ld: i64,
    pub order: LtOrder,
}

impl LtDesc {
    /// A ROW-order descriptor (the operand convention).
    pub fn row(rows: i64, cols: i64, ld: i64) -> Self {
        Self {
            rows,
            cols,
            ld,
            order: LtOrder::Row,
        }
    }

    /// A COL-order descriptor.
    pub fn col(rows: i64, cols: i64, ld: i64) -> Self {
        Self {
            rows,
            cols,
            ld,
            order: LtOrder::Col,
        }
    }

    /// How many elements past the base pointer this descriptor
    /// dereferences: `ld * (major_lines - 1) + minor_extent`, where the
    /// MAJOR lines are rows in ROW order and columns in COL order.
    /// `None` on i64 overflow.
    fn reach(&self) -> Option<i64> {
        let (major, minor) = match self.order {
            LtOrder::Row => (self.rows, self.cols),
            LtOrder::Col => (self.cols, self.rows),
        };
        self.ld.checked_mul(major - 1)?.checked_add(minor)
    }
}

/// Where the C pointer comes from. The C DESCRIPTOR always exists
/// (contract 3); only the pointer source varies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CSource {
    /// No-C forms: pass the D pointer as C (beta = 0.0f, C never read).
    AliasD,
    /// C-fold forms: the Lit operand at this index (contract order
    /// `[a, b, c, bias?]` puts c at 2); beta = 1.0f.
    Operand(usize),
}

/// The fully-resolved host call: every number the dispatch needs,
/// nothing the dispatch may reinterpret. NO scalar fields beyond the
/// structural `beta_is_one` — see module doc contract 2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LtCall {
    pub form: CublasLtForm,
    pub m: i64,
    pub n: i64,
    pub k: i64,
    pub trans_a: bool,
    pub trans_b: bool,
    pub a: LtDesc,
    pub b: LtDesc,
    /// ALWAYS present (contract 3): on the no-C forms this mirrors `d`.
    pub c: LtDesc,
    pub d: LtDesc,
    pub c_source: CSource,
    /// beta is STRUCTURAL: `true` exactly on the C-fold forms.
    pub beta_is_one: bool,
    pub relu: bool,
    /// Lit operand index of the bias vector, on the bias forms.
    pub bias_operand: Option<usize>,
}

fn literal(dim: &CuDim, what: &str) -> Result<i64> {
    match dim.literal() {
        Some(v) => Ok(v),
        None => bail!(
            "cuBLASLt dispatch: {what} is SYMBOLIC — binding symbolic geometry \
             from the dyn map at dispatch is not wired in this landing (loud \
             bail, never a guess)"
        ),
    }
}

/// The REAL ld bounds validation (contract 4) — A VENDOR CHECK, not a
/// type re-check: it stays under every cleanup (see the CHECK TAXONOMY
/// on [`bind_destination`]). The library's own check is
/// self-consistency only — `ld >= minor extent` when there is more than
/// one major line — and VACUOUS at a single major line (one row in ROW
/// order, one column in COL order, never dereferences ld), so a
/// too-small buffer would be read/written out of bounds without a word.
/// This check is the one that counts: [`LtDesc::reach`] `<= elems`,
/// plus positivity. ORDER-AWARE, because the reach is (the ROW formula
/// applied to a COL descriptor understates it whenever `rows > cols`).
pub fn validate_ld_bounds(who: &str, desc: &LtDesc, elems: usize) -> Result<()> {
    if desc.rows < 1 || desc.cols < 1 {
        bail!(
            "cuBLASLt {who}: empty descriptor geometry {}x{} — refused before dispatch",
            desc.rows,
            desc.cols
        );
    }
    if desc.ld < 1 {
        bail!(
            "cuBLASLt {who}: ld {} < 1 — refused before dispatch",
            desc.ld
        );
    }
    let needed = desc.reach().ok_or_else(|| {
        anyhow::anyhow!("cuBLASLt {who}: the descriptor's element reach overflows i64 — refused")
    })?;
    if needed as i128 > elems as i128 {
        bail!(
            "cuBLASLt {who}: descriptor {}x{} ld {} ({:?} order) needs {} elements \
             but the buffer holds {} — out-of-bounds access refused BEFORE dispatch \
             (the library's own ld check is vacuous at a single major line, \
             i.e. rows==1 in ROW order)",
            desc.rows,
            desc.cols,
            desc.ld,
            desc.order,
            needed,
            elems
        );
    }
    Ok(())
}

/// THE PLAN/CALL-FRAME COHERENCE FENCE — resolve the C/D descriptors
/// against the destination value's ELECTED layout, the one the plan
/// disclosed and every downstream reader walks.
///
/// CHECK TAXONOMY (standing ruling, written here so the next cleanup
/// does not delete this again). Three kinds of executor-side check, and
/// only ONE of them is disposable:
///
///  1. E-GRAPH RE-CHECKS — re-verifying a fact a rule PREMISE already
///     guarantees (the marker's F32 end-to-end scope, C's dims and fold
///     status matching D's by rule guard). These are OUT: they restate
///     the e-graph's own postconditions in the executor's voice and
///     rot. Correction 4 of 2026-08-31 removed them; they stay removed.
///  2. VENDOR CHECKS — verifying the LIBRARY's behavior where its own
///     guarantees are absent or vacuous: the TF32 strictness detector
///     (contract 5) and [`validate_ld_bounds`] (contract 4, whose
///     library counterpart is vacuous at rows == 1). These STAY.
///  3. COHERENCE FENCES — agreement between the PLAN's vocabulary
///     (elected layouts, extents, orders) and a CALL FRAME THE EXECUTOR
///     ITSELF BUILDS from a different vocabulary (m/n/k, descriptors,
///     lds). These STAY, and this is the category correction 4 got
///     wrong: it classified the `[m, n]` frame check as (1) and deleted
///     it. It is not (1). No e-graph rule has ever seen `LtCall` — the
///     bridge in this module invents it — so nothing upstream can
///     guarantee the two agree. Deleting the fence is how an exact
///     product landed in transposed byte order twice.
///
/// This function is the fence, and it does more than the deleted check:
/// the old one compared EXTENTS only (`dest_dims == [m, n]`), which the
/// Option-B regression walked straight through — the extents were
/// `[m, n]` and the ORDER was the thing that had diverged. Resolving
/// the order here means the frame cannot silently disagree at all: the
/// only two orders cuBLASLt has are the only two layouts admitted, and
/// the descriptor carries the answer to the dispatch site.
///
/// Admitted destination layouts, both exactly expressible as a cuBLASLt
/// matrix layout over the `m x n` frame:
///  * `RightMajor[m, n]` — `(r, c)` at `r*n + c` → ROW, `ld = n`;
///  * `LeftMajor[m, n]`  — `(r, c)` at `c*m + r` → COL, `ld = m`.
///
/// Everything else is a CAPABILITY REFUSAL, the exact mirror of the
/// kernel path's identity-index write fence in
/// `kernels::CodegenCtx::from_descriptors`: this backend writes no
/// strided and no offset-expression destination, by kernel or by
/// library call. Loud, never wrong bytes.
///
/// C rides D's frame (`c == d` throughout this bridge): the marker's
/// rule guard cross-checks the C and D layout CLASSES, so a C-fold
/// form's C operand is stored the same way its D is.
pub fn bind_destination(
    call: &mut LtCall,
    dest: &luminal::layouts::DecodedLayout,
    who: &str,
) -> Result<()> {
    use luminal::layouts::MirrorLayout as M;
    let extents = dest.mirror.literal_extents().ok_or_else(|| {
        anyhow::anyhow!(
            "cuBLASLt {who}: the destination's elected layout has SYMBOLIC extents \
             — the call frame cannot be checked against it; refused before dispatch"
        )
    })?;
    if extents.len() != 2
        || extents[0] as i128 != call.m as i128
        || extents[1] as i128 != call.n as i128
    {
        bail!(
            "cuBLASLt {who}: the plan's elected destination layout spans {extents:?} \
             but the call frame is [m, n] = [{}, {}] — the plan's vocabulary and the \
             executor's call frame have DIVERGED; refused before dispatch (never \
             land bytes under a disclosure that does not describe them)",
            call.m,
            call.n
        );
    }
    let desc = match &dest.mirror {
        M::RightMajor(_) => LtDesc::row(call.m, call.n, call.n.max(1)),
        M::LeftMajor(_) => LtDesc::col(call.m, call.n, call.m.max(1)),
        other => bail!(
            "cuBLASLt {who}: the plan elected a {} destination layout; this backend \
             writes only the two dense orders cuBLASLt can express (RightMajor -> \
             CUBLASLT_ORDER_ROW, LeftMajor -> CUBLASLT_ORDER_COL). Strided and \
             offset-expression destinations are NOT lowered — a CAPABILITY refusal \
             (the host-call mirror of the codegen path's identity-index write fence), \
             never a guess. Layout: {other:?}",
            match other {
                M::Strided(_) => "STRIDED",
                M::ElementOffset(_) => "ELEMENT-OFFSET-EXPRESSION",
                M::BitOffset(_) => "BIT-OFFSET-EXPRESSION",
                _ => unreachable!("the dense orders are matched above"),
            }
        ),
    };
    call.d = desc;
    call.c = desc;
    // The bias/order tripwire runs HERE — after the D order is known,
    // never before (the spec-only default is ROW and would fire falsely).
    assert_bias_destination_order(call, who)
}

/// THE BIAS/ORDER TRIPWIRE (ruling 2026-09-01). The estate's two bias
/// decorators (`egg/cublaslt_marker_decorate.egg`) mint a bias form ONLY
/// when the claimed D carries the `LeftMajorContiguousElementLayoutLit`
/// spelling over the sibling frame, and [`bind_destination`] maps a
/// LeftMajor election to `CUBLASLT_ORDER_COL`. A bias-bearing call whose
/// D descriptor is not COL is therefore UNREACHABLE from a planned
/// dispatch — it can only mean the estate premise and this bridge have
/// drifted apart (or a hand-built call). Bail, never dispatch: the
/// library refuses BIAS/RELU_BIAS on a ROW-order D
/// (CUBLAS_STATUS_NOT_SUPPORTED, measured on the A100 2026-08-28), and
/// this check names the finding BEFORE any descriptor is built.
///
/// Classification (see the CHECK TAXONOMY on [`bind_destination`]): a
/// COHERENCE FENCE between the estate's premise vocabulary (the LeftMajor
/// literal) and the call frame the executor builds (the D order) — not an
/// e-graph re-check, and not disposable.
pub fn assert_bias_destination_order(call: &LtCall, who: &str) -> Result<()> {
    if call.bias_operand.is_some() && call.d.order != LtOrder::Col {
        bail!(
            "cuBLASLt {who}: unreachable: the bias decorators require a LeftMajor D; \
             a bias form ({}) reached the executor with a {:?}-order D descriptor \
             ({}x{} ld {}). The library refuses BIAS/RELU_BIAS on a ROW-order D \
             (CUBLAS_STATUS_NOT_SUPPORTED, measured on the A100 2026-08-28) — \
             refused BEFORE dispatch, no bytes move",
            call.form.constructor_name(),
            call.d.order,
            call.d.rows,
            call.d.cols,
            call.d.ld
        );
    }
    Ok(())
}

/// Resolve an elected [`CublasLt`] op into the host call. Loud on a
/// missing spec (an op elected without its parsed marker spec is
/// malformed) and on symbolic geometry.
pub fn plan_call(op: &CublasLt) -> Result<LtCall> {
    let Some(spec) = op.spec.as_ref() else {
        bail!(
            "cuBLASLt dispatch: elected {} carries no parsed LtMatmulSpec — \
             the marker's extract() did not resolve this site",
            op.form.constructor_name()
        );
    };
    plan_call_from_spec(spec)
}

/// [`plan_call`] over the spec alone (test seam).
pub fn plan_call_from_spec(spec: &LtMatmulSpec) -> Result<LtCall> {
    // NO BIAS REFUSAL HERE (ruling 2026-09-01). The unconditional
    // bias-form refusal that stood at the top of this function is gone:
    // the estate's bias decorators now require a LeftMajor D, so a bias
    // form arrives with a COL destination election and dispatches. The
    // D order is NOT known at this point (the spec-only default below is
    // ROW), so the bias/order coherence check cannot live here — it runs
    // in [`bind_destination`] once the plan's election has been read
    // (see [`assert_bias_destination_order`]).
    let m = literal(&spec.m, "m")?;
    let n = literal(&spec.n, "n")?;
    let k = literal(&spec.k, "k")?;

    // THE ROW CONVENTION (module doc): the spec's readings are COL
    // views (frozen estate convention); a COL `r x c / ld` view of the
    // operand bytes IS the ROW `c x r / ld` view of the transposed
    // matrix, so the bridge flips each operand's transpose op and
    // swaps the descriptor dims. The op algebra is then the same shape
    // as before, with the FLIPPED trans:
    //   A: op(A') is m x k  =>  A' is (m,k) at N, (k,m) at T
    //   B: op(B') is k x n  =>  B' is (k,n) at N, (n,k) at T
    //   D:                      m x n (no transD)
    let trans_a = !spec.trans_a;
    let trans_b = !spec.trans_b;
    let (a_rows, a_cols) = if trans_a { (k, m) } else { (m, k) };
    let (b_rows, b_cols) = if trans_b { (n, k) } else { (k, n) };
    let (d_rows, d_cols) = (m, n);

    // Operand lds carry over VERBATIM from the spec (a COL view's ld
    // and the row-major storage's row pitch are the same number,
    // padded layouts included), clamped to >= 1 (contract 4's
    // emission rule).
    let ld = |dim: &CuDim, what: &str| -> Result<i64> { Ok(literal(dim, what)?.max(1)) };
    let a = LtDesc::row(a_rows, a_cols, ld(&spec.lda, "lda")?);
    let b = LtDesc::row(b_rows, b_cols, ld(&spec.ldb, "ldb")?);
    // D is the EXECUTOR's destination, not the spec's claim. The spec's
    // ldd describes the claimed e-graph layout over the RECORDER's out
    // buffer — a buffer the executor never writes — so consuming it
    // here was the orientation bug.
    //
    // What stands here is the SPEC-ONLY DEFAULT: dense ROW `m x n`,
    // ld = n. It is the right answer for a caller with no plan (the
    // hand-built direct-dispatch contract tests) and it is NOT the
    // final word for a planned dispatch: the executor calls
    // [`bind_destination`] with the result slot's ELECTED layout, which
    // may resolve this frame to COL (the transpose-sandwich sibling
    // does). Assuming ROW here and never re-resolving it was the
    // Option-B destination-frame regression.
    let d = LtDesc::row(d_rows, d_cols, n.max(1));
    // C rides the D layout by rule guard (the marker cross-checks the
    // layout classes), so Cdesc == Ddesc geometry on EVERY form — the
    // valid-Cdesc contract for the no-C forms comes for free. On the
    // C-fold forms the executor owes a C operand buffer holding the
    // call-frame C dense row-major (the dispatch site enforces it).
    let c = d;

    let c_source = if spec.form.has_c() {
        // Contract order [a, b, c, bias?]: c is Lit operand 2.
        CSource::Operand(2)
    } else {
        CSource::AliasD
    };
    let bias_operand = spec.form.has_bias().then(|| match spec.form {
        CublasLtForm::Bias => 2,
        CublasLtForm::AccumulateBias => 3,
        _ => unreachable!("has_bias() is true only on the bias forms"),
    });
    let relu = matches!(spec.epilogue, CuEpilogue::Relu | CuEpilogue::ReluBias);

    Ok(LtCall {
        form: spec.form,
        m,
        n,
        k,
        trans_a,
        trans_b,
        a,
        b,
        c,
        d,
        c_source,
        beta_is_one: spec.form.has_c(),
        relu,
        bias_operand,
    })
}

impl LtCall {
    /// Validate every descriptor against its backing buffer's element
    /// count — the pre-dispatch gate (contract 4). `elems` in Lit
    /// operand order `[a, b, c?, bias?]`, then the destination.
    pub fn validate_against(&self, operand_elems: &[usize], dest_elems: usize) -> Result<()> {
        if operand_elems.len() != self.form.lit_arity() {
            bail!(
                "cuBLASLt {}: {} operand buffers for Lit arity {}",
                self.form.constructor_name(),
                operand_elems.len(),
                self.form.lit_arity()
            );
        }
        validate_ld_bounds("A", &self.a, operand_elems[0])?;
        validate_ld_bounds("B", &self.b, operand_elems[1])?;
        validate_ld_bounds("D", &self.d, dest_elems)?;
        match self.c_source {
            CSource::AliasD => validate_ld_bounds("C(=D)", &self.c, dest_elems)?,
            CSource::Operand(i) => {
                let Some(&elems) = operand_elems.get(i) else {
                    bail!("cuBLASLt: C operand index {i} out of operand range");
                };
                validate_ld_bounds("C", &self.c, elems)?;
            }
        }
        if let Some(i) = self.bias_operand {
            let Some(&elems) = operand_elems.get(i) else {
                bail!("cuBLASLt: bias operand index {i} out of operand range");
            };
            // The bias vector is length m (one entry per D row —
            // independent of storage order; this is why the ROW
            // convention was chosen over the CM-swap trick).
            if (elems as i128) < self.m as i128 {
                bail!(
                    "cuBLASLt bias: buffer holds {elems} elements, epilogue reads m = {} \
                     — refused before dispatch",
                    self.m
                );
            }
        }
        Ok(())
    }
}
