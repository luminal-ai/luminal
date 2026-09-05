//! EXTREME AND NON-FINITE CONSTANT LITERALS UNDER NVRTC (`device`
//! feature only). The literal-emission fix (`kernels::cuda_f64_literal`)
//! is argued from the C grammar and pinned by host string tests; this
//! suite is the other half — the same values compiled by the REAL NVRTC
//! and executed on the REAL device, so "valid C token" is proven rather
//! than asserted.
//!
//! THE PATH THE LITERAL TAKES, in order:
//!
//!  1. `Graph::constant_float(v)` (src/frontend/other.rs) records
//!     `LogicalOp::Constant(v as f64)`; the term's f64 child is written
//!     into the egglog program by `Graph`'s renderer as Rust's `{:?}`
//!     form — `-3.4028234663852886e38`, `-inf`, `NaN` — which the
//!     egglog parser accepts as f64 literals.
//!  2. `expand_rhs` records the broadcast view of that rank-0 constant,
//!     and the sum with a runtime input keeps the value opaque to any
//!     algebraic simplification: nothing in the graph can fold it away.
//!  3. Search elects this runtime's `ConstantDps`. Every test below
//!     ASSERTS the elected plan carries a `ConstantGeneric` compute
//!     node whose `value` BITS are the ones under test, so a plan that
//!     folded the constant away, or that reached codegen with some
//!     other number, fails loudly instead of passing vacuously. Steps
//!     1-3 were confirmed on a device-free host before this file
//!     landed: all four graphs plan, every elected op has a codegen
//!     row, and the plans carry `-3.4028234663852886e38`, `-inf` and
//!     `NaN` unchanged. Steps 4-5 are what only a device can settle.
//!  4. `ops::constant::codegen` formats the value through
//!     `kernels::cuda_f64_literal` into `out[i] = (float)<literal>;`.
//!  5. `execute()` hands that source to NVRTC. A literal that is not a
//!     C token fails to compile HERE (the pre-fix `-inf` / `NaN` and the
//!     39-digit integer-literal forms all did); a literal that compiles
//!     but denotes the wrong number shows up in the readback bits.
//!
//! CUMULATIVE MAX is the frontend-reachable witness: `GraphTensor::cummax`
//! seeds its window with `f32::MIN` (src/frontend/unary.rs), which `pad`
//! mints as exactly this constant (src/frontend/movement.rs `pad` ->
//! `constant_float(elem)`). It runs here as a fourth case with a host
//! reference, so the motivating caller is covered end to end and not
//! only the synthetic constant.
#![cfg(feature = "device")]

use luminal::bufferize::BufferNode;
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::prelude::{FxHashMap, NodeIndex};
use luminal_cuda_lite::CudaRuntime;
use luminal_cuda_lite::HostBuffer;
use luminal_cuda_lite::ops::constant::ConstantDps;

/// Read the device output DENSELY through its RETURNED LAYOUT
/// (escape-and-disclose, 2026-08-31) — the same universal readback
/// `tests/device_fidelity.rs` and `tests/device_view_differentials.rs`
/// use. A dense election evaluates the identity; a broadcast election
/// evaluates to the same backing element at every coordinate. Either
/// way the caller sees the value at each output coordinate, which is
/// what these assertions are about.
fn walked_dense(rt: &CudaRuntime, out: NodeIndex) -> Vec<f32> {
    let (data, binding) = rt.fetch(out).expect("escape-and-disclose fetch");
    let bytes = data
        .as_f32()
        .unwrap_or_else(|err| panic!("output is not f32: {err}"));
    luminal_cuda_lite::layouts::dense_f32(&bytes, &binding.layout)
        .expect("the returned layout reads dense over its backing buffer")
}

/// The plan must carry THIS EXACT VALUE as a device kernel. Without
/// the check the suite could pass on a plan that never reached
/// `ops::constant::codegen`, which would make every assertion below
/// vacuous with respect to the literal; the value comparison is on
/// BITS, so it is meaningful for NaN and the infinities too, and it
/// doubles as the proof that the f64 survived the egglog round trip
/// (`Graph`'s renderer writes `{:?}`; the vendored parser reads back
/// `NaN` / `-inf` / the exponent form).
fn assert_constant_reaches_the_device(rt: &CudaRuntime, expected: f64, what: &str) {
    let plan = rt.plan().expect("plan loaded");
    let elected: Vec<f64> = plan
        .dag
        .node_weights()
        .filter_map(|node| match node {
            BufferNode::Compute { op, .. } => op
                .as_any()
                .downcast_ref::<ConstantDps>()
                .map(|constant| constant.value),
            _ => None,
        })
        .collect();
    assert!(
        elected.iter().any(|v| v.to_bits() == expected.to_bits()),
        "{what}: no ConstantGeneric compute node carries {expected:?} \
         (the plan's constants are {elected:?}) — the literal never \
         reached NVRTC, so this test would prove nothing"
    );
}

/// Search under the CUDA allow list, assert the constant is really in
/// the plan, then NVRTC-compile and run it. Harness copied from
/// `tests/device_fidelity.rs::run_both`, minus the reference side:
/// non-finite outputs cannot go through its tolerance comparison
/// (`NaN != NaN`, and `inf - inf` is `NaN`), so each test states the
/// exact bits it expects instead.
fn run_on_device(
    cx: &Graph,
    inputs: &[(NodeIndex, Vec<f32>)],
    out: NodeIndex,
    expected_constant: f64,
    what: &str,
) -> Vec<f32> {
    let mut rt = CudaRuntime::load(cx).expect("cuda load");
    // THE TWO RUNTIMES TAKE DIFFERENT HOST PAYLOADS (ruling D4,
    // 2026-09-03): the CL side stages `HostBuffer` — bytes plus a dtype
    // tag, ready for an H2D copy.
    let data: FxHashMap<NodeIndex, HostBuffer> = inputs
        .iter()
        .map(|(id, v)| (*id, v.clone().into()))
        .collect();
    rt.search(&data, &luminal_cuda_lite::harness_search_options())
        .expect("cuda search");
    assert_constant_reaches_the_device(&rt, expected_constant, what);
    for (id, v) in inputs {
        rt.set_data(*id, v.clone());
    }
    rt.execute()
        .expect("device execute (NVRTC compiles the constant kernel here)");
    walked_dense(&rt, out)
}

/// `zeros + broadcast(constant(value))` over 8 elements. Adding a
/// runtime input is what forces the constant to be a real device value
/// rather than something the graph could rewrite away, and `0.0 + x`
/// is bit-exact for every finite `x` and for the infinities.
fn constant_plus_zero(value: f32) -> Vec<f32> {
    const N: usize = 8;
    let mut cx = Graph::new();
    let a = cx.tensor(N, DType::F32);
    let c = cx.constant_float(value).expand_rhs(a.dims());
    let out = (a + c).output();
    let got = run_on_device(
        &cx,
        &[(a.id, vec![0.0f32; N])],
        out.id,
        value as f64,
        &format!("constant {value:?}"),
    );
    assert_eq!(got.len(), N, "constant {value:?}: wrong output length");
    got
}

/// `f32::MIN` — the value `cummax` seeds with. Its `Display` form is a
/// 39-digit run with no decimal point and no exponent, which C reads as
/// an integer literal too large for any integer type: pre-fix this did
/// not compile. Bitwise equality is the bar, since `{:e}` must
/// round-trip the value exactly through the `double` literal and the
/// `(float)` cast.
#[test]
fn extreme_finite_constant_survives_nvrtc_bit_for_bit() {
    for (i, g) in constant_plus_zero(f32::MIN).iter().enumerate() {
        assert_eq!(
            g.to_bits(),
            f32::MIN.to_bits(),
            "element {i}: expected f32::MIN ({:e}), got {g:e}",
            f32::MIN
        );
    }
}

/// `-inf` — what attention masks fill with. Pre-fix the kernel carried
/// the bare token `-inf`, which is not C at all. `__uint_as_float`
/// takes its place, and NVRTC has the intrinsic without any header.
#[test]
fn negative_infinity_constant_survives_nvrtc() {
    for (i, g) in constant_plus_zero(f32::NEG_INFINITY).iter().enumerate() {
        assert!(
            g.is_infinite() && g.is_sign_negative(),
            "element {i}: expected -inf, got {g}"
        );
    }
}

/// `NaN` — pre-fix the kernel carried the bare token `NaN`. Only
/// NaN-ness is asserted: the payload is not contractual, and an add can
/// legally quiet or re-tag it.
#[test]
fn nan_constant_survives_nvrtc() {
    for (i, g) in constant_plus_zero(f32::NAN).iter().enumerate() {
        assert!(g.is_nan(), "element {i}: expected NaN, got {g}");
    }
}

/// THE FRONTEND-REACHABLE CASE. `cummax` pads its window with
/// `f32::MIN` (src/frontend/unary.rs), `pad` mints that as
/// `constant_float(f32::MIN)` (src/frontend/movement.rs), and the
/// windowed max then selects over it. The seed is the reduction
/// identity: it must be smaller than every real element, so a literal
/// that compiled to the wrong magnitude would show up as a wrong
/// running maximum rather than as a compile error.
///
/// The reference is the running maximum computed on the host. `max` is
/// exact selection — no arithmetic — so equality is exact, not
/// tolerant.
#[test]
fn cummax_seed_constant_survives_nvrtc() {
    let input = vec![-5.0f32, -3., -9., -1., -7., -2., -8., -4.];
    let mut cx = Graph::new();
    let a = cx.tensor(input.len(), DType::F32);
    let out = a.cummax(0).output();
    let got = run_on_device(
        &cx,
        &[(a.id, input.clone())],
        out.id,
        f32::MIN as f64,
        "cummax seed",
    );

    let mut running = f32::NEG_INFINITY;
    let want: Vec<f32> = input
        .iter()
        .map(|v| {
            running = running.max(*v);
            running
        })
        .collect();
    assert_eq!(got, want, "cummax over {input:?}");
}
