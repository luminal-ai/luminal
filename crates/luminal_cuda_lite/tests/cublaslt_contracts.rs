//! Train 3, Items 4+5 — the DEVICE-GATED cuBLASLt contract tests:
//! compiled under `--features device` on any host (the compile check is
//! this train's gate) and EXECUTED by the orchestrator's A100 pass.
//!
//! Coverage:
//!  * the four contract forms on real device buffers (direct
//!    `device_call::dispatch` over hand-built `LtCall`s): the
//!    DEFAULT-epilogue forms under a ROW D and — RULING 2026-09-01,
//!    NEEDS A100 RUN — the BIAS-epilogue forms under a COL D, all
//!    compared tolerance-based against the host walk; a bias form with
//!    a ROW D pins the tripwire (the measured library restriction — no
//!    BIAS/RELU_BIAS on a ROW-order D — now unreachable from the estate,
//!    whose bias decorators require a LeftMajor D);
//!  * the TF32 strictness detector assertion (contract 5);
//!  * a deliberate ld-bounds violation refused loudly BEFORE dispatch
//!    (contract 4 — including the rows==1 case the library's own check
//!    is vacuous on);
//!  * NUMERICS POLICY (Item 4): the marker-elected plan compares
//!    against the decomposed route TOLERANCE-based only — see
//!    `assert_close` for the reduction-order contract.
#![cfg(feature = "device")]

use cudarc::driver::{CudaContext, CudaSlice};
use luminal::buffer_tensor_ir::TypedBuffer;
use luminal::bufferize::BufferNode;
use luminal::prelude::{DType, FxHashMap, NodeIndex};

/// The universal escape-and-disclose readback (the device_fidelity
/// pattern): fetch the backing bytes + binding, walk each output
/// element through the disclosed layout. Dense elections walk the
/// identity, view elections the composed chain.
fn walked_dense(rt: &CudaRuntime, out: NodeIndex) -> Vec<f32> {
    let (data, binding) = rt.fetch(out).expect("escape-and-disclose fetch");
    let bytes = match data {
        TypedBuffer::F32(values) => values,
        other => panic!("output is {}, not f32", other.type_name()),
    };
    // The value's shape and read path both come from the RETURNED
    // LAYOUT; there is no `dims` field and no hop chain any more.
    luminal_cuda_lite::layouts::dense_f32(bytes, &binding.layout)
        .expect("the returned layout reads dense over its backing buffer")
}
use luminal_cuda_lite::ops::cublaslt::device_call;
use luminal_cuda_lite::ops::cublaslt::exec::{CSource, LtCall, LtDesc, LtOrder};
use luminal_cuda_lite::ops::cublaslt::CublasLtForm;
use luminal_cuda_lite::CudaRuntime;
use std::sync::Arc;

/// REDUCTION-ORDER CONTRACT (Item 4, documented at the comparison
/// site): bit-exactness between a vendor GEMM and the decomposed
/// mul+reduce route is IMPOSSIBLE IN PRINCIPLE — cublasLtMatmul's
/// reduction order is algorithm-dependent and unspecified (split-k,
/// tiling, FMA contraction), while the decomposed route reduces in
/// linear axis order. Marker-elected results are therefore compared
/// TOLERANCE-based at the device_fidelity epsilon
/// (`tol = 1e-5.max(|reference| * 1e-5)`), and NOTHING in this tree
/// may claim or test bit-equality against the decomposed route.
/// NaN is incomparable and bails (the negated-predicate idiom).
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn assert_close(want: &[f32], got: &[f32], what: &str) {
    assert_eq!(want.len(), got.len(), "{what}: length mismatch");
    for (i, (w, g)) in want.iter().zip(got).enumerate() {
        let tol = 1e-5f32.max(w.abs() * 1e-5);
        let diff = (w - g).abs();
        assert!(
            diff <= tol,
            "{what}: element {i} diverges — expected {w}, got {g} (|delta| {diff:.3e} > tol {tol:.3e})"
        );
    }
}

/// Deterministic values (the shared example seeding discipline).
fn weights(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + seed * 101 + 13) % 121) as f32 / 100.0) - 0.6)
        .collect()
}

/// The seed at which the 12x16/mut-4 search elected `CublasLtBias` on
/// the CPU pin (`tests/cublaslt_bias_premise.rs`,
/// `search_elects_the_bias_form_and_binds_a_col_d`).
const BIAS_ELECTING_SEED: u64 = 0;

/// Element index of `(r, c)` under a descriptor's own order: ROW puts it
/// at `r*ld + c`, COL at `c*ld + r`.
fn at(desc: &LtDesc, r: usize, c: usize) -> usize {
    match desc.order {
        LtOrder::Row => r * desc.ld as usize + c,
        LtOrder::Col => c * desc.ld as usize + r,
    }
}

/// Host reference for one call: A and B walked in ROW order (the bridge's
/// operand convention), C and D through THEIR descriptors' declared order
/// (ROW for the default forms; COL for the bias forms, ruling 2026-09-01)
/// — D = act(op(A)op(B) + beta*C + bias), alpha = 1 (the fixed literal),
/// bias[row] added along D's rows (the API's only bias axis, independent
/// of storage order).
fn host_reference(
    call: &LtCall,
    a: &[f32],
    b: &[f32],
    c: Option<&[f32]>,
    bias: Option<&[f32]>,
) -> Vec<f32> {
    let (m, n, k) = (call.m as usize, call.n as usize, call.k as usize);
    let mut d = vec![0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0f64;
            for kk in 0..k {
                // ROW storage with the descriptor's ld; op applied.
                let a_v = if call.trans_a {
                    a[kk * call.a.ld as usize + row] // A' is k x m
                } else {
                    a[row * call.a.ld as usize + kk] // A' is m x k
                };
                let b_v = if call.trans_b {
                    b[col * call.b.ld as usize + kk] // B' is n x k
                } else {
                    b[kk * call.b.ld as usize + col] // B' is k x n
                };
                acc += (a_v as f64) * (b_v as f64);
            }
            if let Some(c) = c {
                acc += c[at(&call.c, row, col)] as f64;
            }
            if let Some(bias) = bias {
                acc += bias[row] as f64;
            }
            let mut v = acc as f32;
            if call.relu {
                v = v.max(0.0);
            }
            d[at(&call.d, row, col)] = v;
        }
    }
    d
}

fn to_device(stream: &Arc<cudarc::driver::CudaStream>, host: &[f32]) -> CudaSlice<u8> {
    let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let mut slice = stream.alloc_zeros::<u8>(bytes.len().max(1)).expect("alloc");
    stream.memcpy_htod(&bytes, &mut slice).expect("H2D");
    slice
}

fn from_device(stream: &Arc<cudarc::driver::CudaStream>, slice: &CudaSlice<u8>) -> Vec<f32> {
    let mut host = vec![0u8; slice.len()];
    stream.memcpy_dtoh(slice, &mut host).expect("D2H");
    host.chunks_exact(4)
        .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
        .collect()
}

/// Build the canonical contiguous call for one form (m=3, n=4, k=5) —
/// dense row-major operands (ld = the row pitch = cols); C and D in the
/// order the plan would elect for the form: ROW (ld = n) for the default
/// forms, COL (ld = m) for the bias forms — the estate's bias decorators
/// require a LeftMajor D, which `bind_destination` resolves to COL.
fn call_for(form: CublasLtForm) -> LtCall {
    let (m, n, k) = (3i64, 4i64, 5i64);
    let cd = if form.has_bias() {
        LtDesc::col(m, n, m)
    } else {
        LtDesc::row(m, n, n)
    };
    LtCall {
        form,
        m,
        n,
        k,
        trans_a: false,
        trans_b: false,
        a: LtDesc::row(m, k, k),
        b: LtDesc::row(k, n, n),
        c: cd,
        d: cd,
        c_source: if form.has_c() {
            CSource::Operand(2)
        } else {
            CSource::AliasD
        },
        beta_is_one: form.has_c(),
        relu: false,
        bias_operand: form.has_bias().then(|| if form.has_c() { 3 } else { 2 }),
    }
}

/// Contract 5: the TF32 strictness detector runs (once) at handle
/// creation and must be green on the A100 (strict FP32, no TF32
/// fallback in effect).
#[test]
fn tf32_strictness_detector_is_green() {
    device_call::assert_compute_strictness().expect(
        "strict CUBLAS_COMPUTE_32F must be in effect (TF32 is graph-modeled, never a flag)",
    );
}

/// The four contract forms execute green on real buffers, compared
/// tolerance-based against the host walk (see `assert_close`'s
/// reduction-order contract): the DEFAULT-epilogue forms (Base,
/// Accumulate) under a ROW D, the BIAS-epilogue forms (Bias,
/// AccumulateBias) under a COL D.
///
/// RULING 2026-09-01 — NEEDS A100 RUN. The bias forms used to pin a
/// refusal here (the 2026-08-28 finding: CUBLAS_STATUS_NOT_SUPPORTED for
/// BIAS/RELU_BIAS whenever D is CUBLASLT_ORDER_ROW). The estate now
/// requires a LeftMajor D on the bias decorators, so a planned bias form
/// always dispatches with `CUBLASLT_ORDER_COL` — the order the probe
/// measured as SUPPORTED. This test's bias rows are the first device
/// measurement of that dispatch; the ROW-D refusal moved to
/// `bias_form_with_a_row_d_is_refused_before_dispatch`.
#[test]
fn all_four_contract_forms_execute_green() {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let stream = ctx.default_stream();
    for form in CublasLtForm::ALL {
        let call = call_for(form);
        let (m, n, k) = (call.m as usize, call.n as usize, call.k as usize);
        let a = weights(m * k, 1);
        let b = weights(k * n, 2);
        let c = form.has_c().then(|| weights(m * n, 3));
        let bias = form.has_bias().then(|| weights(m, 4));

        let dev_a = to_device(&stream, &a);
        let dev_b = to_device(&stream, &b);
        let mut operands: Vec<&CudaSlice<u8>> = vec![&dev_a, &dev_b];
        let dev_c = c.as_ref().map(|c| to_device(&stream, c));
        let dev_bias = bias.as_ref().map(|v| to_device(&stream, v));
        if let Some(dc) = dev_c.as_ref() {
            operands.push(dc);
        }
        if let Some(db) = dev_bias.as_ref() {
            operands.push(db);
        }
        let mut dest = stream.alloc_zeros::<u8>(m * n * 4).expect("dest alloc");

        // Bias forms: COL D (ld = m), the order the library supports for
        // BIAS/RELU_BIAS; the epilogue adds bias[row] along D's rows.
        assert_eq!(
            call.d.order,
            if form.has_bias() {
                LtOrder::Col
            } else {
                LtOrder::Row
            }
        );
        device_call::dispatch(&call, &operands, &mut dest, &stream)
            .unwrap_or_else(|e| panic!("{form:?} dispatch: {e:#}"));
        stream.synchronize().expect("sync");

        let got = from_device(&stream, &dest);
        let want = host_reference(&call, &a, &b, c.as_deref(), bias.as_deref());
        assert_close(&want, &got, &format!("{form:?}"));
    }
}

/// Contract 4: a deliberate ld-bounds violation is refused loudly
/// BEFORE dispatch — including the rows==1 shape whose ld the library
/// itself would happily accept (its check is vacuous there). The
/// destination stays all-zero: no bytes moved.
#[test]
fn ld_bounds_violation_refuses_before_dispatch() {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let stream = ctx.default_stream();
    // rows==1 (the shape whose ld the library never dereferences in
    // ROW order): D is 1x8 — needs 8 elements; give it 4.
    let call = LtCall {
        form: CublasLtForm::Base,
        m: 1,
        n: 8,
        k: 2,
        trans_a: false,
        trans_b: false,
        a: LtDesc::row(1, 2, 2),
        b: LtDesc::row(2, 8, 8),
        c: LtDesc::row(1, 8, 8),
        d: LtDesc::row(1, 8, 8),
        c_source: CSource::AliasD,
        beta_is_one: false,
        relu: false,
        bias_operand: None,
    };
    let dev_a = to_device(&stream, &weights(2, 1));
    let dev_b = to_device(&stream, &weights(16, 2));
    let mut dest = stream.alloc_zeros::<u8>(4 * 4).expect("short dest"); // 4 f32s, needs 8
    let err = device_call::dispatch(&call, &[&dev_a, &dev_b], &mut dest, &stream)
        .expect_err("the short D buffer must be refused BEFORE dispatch");
    let msg = format!("{err:#}");
    assert!(msg.contains("refused BEFORE dispatch"), "{msg}");
    stream.synchronize().expect("sync");
    assert!(
        from_device(&stream, &dest).iter().all(|&v| v == 0.0),
        "no bytes may move on a refused dispatch"
    );
}

/// THE TRIPWIRE ON DEVICE (ruling 2026-09-01): a bias form whose D is
/// ROW-order is unreachable from the estate (the bias decorators require
/// a LeftMajor D). A hand-built one is refused BEFORE any library call —
/// the library would return CUBLAS_STATUS_NOT_SUPPORTED (measured
/// 2026-08-28) — and no bytes move.
#[test]
fn bias_form_with_a_row_d_is_refused_before_dispatch() {
    let ctx = CudaContext::new(0).expect("CUDA device 0");
    let stream = ctx.default_stream();
    for form in [CublasLtForm::Bias, CublasLtForm::AccumulateBias] {
        let mut call = call_for(form);
        let (m, n, k) = (call.m as usize, call.n as usize, call.k as usize);
        call.c = LtDesc::row(call.m, call.n, call.n);
        call.d = call.c;
        let a = weights(m * k, 1);
        let b = weights(k * n, 2);
        let dev_a = to_device(&stream, &a);
        let dev_b = to_device(&stream, &b);
        let mut operands: Vec<&CudaSlice<u8>> = vec![&dev_a, &dev_b];
        let dev_c = form.has_c().then(|| to_device(&stream, &weights(m * n, 3)));
        let dev_bias = to_device(&stream, &weights(m, 4));
        if let Some(dc) = dev_c.as_ref() {
            operands.push(dc);
        }
        operands.push(&dev_bias);
        let mut dest = stream.alloc_zeros::<u8>(m * n * 4).expect("dest alloc");
        let err = device_call::dispatch(&call, &operands, &mut dest, &stream)
            .expect_err("a ROW-order D under a bias form must trip the fence");
        let msg = format!("{err:#}");
        assert!(msg.contains("unreachable"), "{form:?}: {msg}");
        assert!(
            msg.contains("bias decorators require a LeftMajor D"),
            "{form:?}: {msg}"
        );
        assert!(msg.contains("refused BEFORE dispatch"), "{form:?}: {msg}");
        stream.synchronize().expect("sync");
        assert!(
            from_device(&stream, &dest).iter().all(|&v| v == 0.0),
            "{form:?}: no bytes may move on a refused dispatch"
        );
    }
}

/// THE BIAS FORM END TO END (ruling 2026-09-01 — NEEDS A100 RUN): the
/// marker-elected plan for `x[4,8] @ w[8,3] + b[3]` (spelled with
/// `luminal_nn::linear`) executes through the
/// host-call arm with the BIAS epilogue under the sibling's COL D, against
/// the decomposed route, tolerance-based. The seed is the one the CPU pin
/// (`tests/cublaslt_bias_premise.rs`) measured electing `CublasLtBias`.
#[test]
fn marker_elected_bias_plan_matches_decomposed_route_tolerance_based() {
    let build = || {
        let mut cx = luminal::graph::Graph::new();
        let weight = cx.named_tensor("fc.weight", (8, 3), DType::F32);
        let bias = cx.named_tensor("fc.bias", 3, DType::F32);
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let out = luminal_nn::linear(x, weight, Some(bias)).output();
        (cx, x.id, weight.id, bias.id, out.id)
    };
    let data_for =
        |x: NodeIndex, w: NodeIndex, b: NodeIndex| -> FxHashMap<NodeIndex, TypedBuffer> {
            [
                (x, TypedBuffer::from(weights(32, 1))),
                (w, TypedBuffer::from(weights(24, 2))),
                (b, TypedBuffer::from(weights(3, 3))),
            ]
            .into_iter()
            .collect()
        };
    let options = luminal::implementation_search::ImplementationSearchOptions {
        generations: 12,
        generation_size: 16,
        mutations: 4,
        trials: 1,
        seed: BIAS_ELECTING_SEED,
        search_log: false,
    };

    let (cx, x, w, b, out) = build();
    let mut fused = CudaRuntime::load_with_cublaslt(&cx).expect("load fused");
    fused
        .search(&data_for(x, w, b), &options)
        .expect("fused search");
    let elected_bias = fused
        .plan()
        .expect("plan")
        .dag
        .node_weights()
        .any(|n| matches!(n, BufferNode::Compute { op, .. } if op.label() == "CublasLtBias"));
    assert!(
        elected_bias,
        "the fused route must elect CublasLtBias for this comparison (seed {BIAS_ELECTING_SEED} measured electing on the CPU pin; re-sweep tests/cublaslt_bias_premise.rs if this moves)"
    );
    fused.set_data(x, weights(32, 1));
    fused.set_data(w, weights(24, 2));
    fused.set_data(b, weights(3, 3));
    fused
        .execute()
        .expect("fused execute (bias epilogue under COL D)");
    let got = walked_dense(&fused, out);

    let (cx, x, w, b, out) = build();
    let mut plain = CudaRuntime::load(&cx).expect("load plain");
    plain
        .search(
            &data_for(x, w, b),
            &luminal::test_support::harness_search_options(),
        )
        .expect("plain search");
    plain.set_data(x, weights(32, 1));
    plain.set_data(w, weights(24, 2));
    plain.set_data(b, weights(3, 3));
    plain.execute().expect("plain execute");
    let want = walked_dense(&plain, out);

    assert_close(&want, &got, "marker(bias) vs decomposed 4x8x3 + b[3]");
}

/// Item 4 END TO END: the marker-elected plan (searched with the
/// marker vocabulary, executed through the host-call arm) against the
/// decomposed route (default vocabulary, NVRTC kernels), tolerance-based
/// per the reduction-order contract in `assert_close`.
#[test]
fn marker_elected_plan_matches_decomposed_route_tolerance_based() {
    let build = || {
        let mut cx = luminal::graph::Graph::new();
        let a = cx.tensor((4usize, 8usize), DType::F32);
        let b = cx.tensor((8usize, 3usize), DType::F32);
        let out = a.matmul(b).output();
        (cx, a, b, out)
    };
    let data_for = |a: NodeIndex, b: NodeIndex| -> FxHashMap<NodeIndex, TypedBuffer> {
        [
            (a, TypedBuffer::from(weights(32, 1))),
            (b, TypedBuffer::from(weights(24, 2))),
        ]
        .into_iter()
        .collect()
    };
    // The seeded budget the CPU election pin measured green (see
    // tests/cublaslt_election.rs).
    let options = luminal::implementation_search::ImplementationSearchOptions {
        generations: 12,
        generation_size: 16,
        mutations: 4,
        trials: 1,
        seed: 0,
        search_log: false,
    };

    // Marker-elected route.
    let (cx, a, b, out) = build();
    let mut fused = CudaRuntime::load_with_cublaslt(&cx).expect("load fused");
    let data = data_for(a.id, b.id);
    fused.search(&data, &options).expect("fused search");
    let elected =
        fused.plan().expect("plan").dag.node_weights().any(
            |n| matches!(n, BufferNode::Compute { op, .. } if op.label().starts_with("CublasLt")),
        );
    assert!(
        elected,
        "the fused route must actually elect the marker for this comparison"
    );
    fused.set_data(a.id, weights(32, 1));
    fused.set_data(b.id, weights(24, 2));
    fused.execute().expect("fused execute");
    // The marker-elected output is the sandwich's sibling VIEW — it
    // escapes with a composed layout, so the honest readback walks the
    // disclosed layout (get_f32 refuses non-row-major backings by design).
    let got = walked_dense(&fused, out.id);

    // Decomposed route (default vocabulary — no marker in the assembly).
    let (cx, a, b, out) = build();
    let mut plain = CudaRuntime::load(&cx).expect("load plain");
    let data = data_for(a.id, b.id);
    plain
        .search(&data, &luminal::test_support::harness_search_options())
        .expect("plain search");
    plain.set_data(a.id, weights(32, 1));
    plain.set_data(b.id, weights(24, 2));
    plain.execute().expect("plain execute");
    let want = walked_dense(&plain, out.id);

    // TOLERANCE-BASED, never bit-equality (reduction-order contract).
    assert_close(&want, &got, "marker vs decomposed 4x8x3");
}
