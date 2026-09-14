//! Translation tests for the port-batch-6 lowerings (attention, grouped
//! GEMM, upsample/resize) and the symbolic/dynamic translator.
//!
//! Fixtures are produced by `torch.export.save` (see the task notes) into
//! `<workspace>/target/luminal_fixtures`. Each test skips when its fixture
//! is absent so `cargo test` stays runnable without a torch environment.

use luminal::prelude::{DType, DynMap, IntExpr};
use luminal_pytorch_utils::{Translation, parse_pt2, translate};

/// A translated boundary shape is symbolic; these fixtures are static, so
/// every dim must resolve to a literal.
fn concrete(shape: &[IntExpr]) -> Vec<usize> {
    shape
        .iter()
        .map(|dim| dim.to_usize().expect("static fixture dim"))
        .collect()
}

fn fixture(name: &str) -> Option<Translation> {
    let path = format!(
        "{}/../../../target/luminal_fixtures/{name}.pt2",
        env!("CARGO_MANIFEST_DIR")
    );
    if !std::path::Path::new(&path).exists() {
        eprintln!("skipping: {path} not found");
        return None;
    }
    let parsed = parse_pt2(&path).expect("parse pt2");
    Some(translate(&parsed).expect("translate"))
}

#[test]
fn sdpa_translates() {
    let Some(t) = fixture("sdpa") else { return };
    let user: Vec<_> = t
        .inputs
        .iter()
        .filter(|i| matches!(i.kind, luminal_pytorch_utils::InputKind::UserInput { .. }))
        .collect();
    assert_eq!(user.len(), 3, "q, k, v");
    assert_eq!(t.outputs.len(), 1);
    assert_eq!(concrete(&t.outputs[0].shape), vec![2, 5, 8]);
}

#[test]
fn grouped_mm_translates() {
    let Some(t) = fixture("grouped_mm") else {
        return;
    };
    assert_eq!(t.inputs.len(), 3, "x, weight, offs");
    assert_eq!(t.outputs.len(), 1);
    assert_eq!(concrete(&t.outputs[0].shape), vec![6, 8]);
    assert_eq!(t.outputs[0].dtype, DType::Bf16);
}

#[test]
fn upsample_nearest_translates() {
    let Some(t) = fixture("up_nearest") else {
        return;
    };
    assert_eq!(t.outputs.len(), 1);
    assert_eq!(concrete(&t.outputs[0].shape), vec![1, 1, 8, 8]);
}

#[test]
fn upsample_bilinear_translates() {
    let Some(t) = fixture("up_bilinear") else {
        return;
    };
    assert_eq!(t.outputs.len(), 1);
    assert_eq!(concrete(&t.outputs[0].shape), vec![1, 1, 8, 8]);
}

#[test]
fn dynamic_dim_survives_as_a_symbol() {
    let Some(t) = fixture("dynamic") else {
        return;
    };
    // The batch symbol torch exported must be present and the output's
    // symbolic shape must retain it (the hint stays [3, 4]).
    assert!(!t.symbols.is_empty(), "dynamic fixture produced no symbols");
    assert!(!t.dims.is_empty(), "dynamic fixture produced no dim hints");
    let shape = &t.outputs[0].shape;
    assert_eq!(shape.len(), 2);
    assert!(
        shape[0].to_usize().is_none(),
        "expected batch dim to stay symbolic, got {:?}",
        shape[0]
    );
    assert_eq!(shape[1], IntExpr::from(4usize));
    // The hint torch exported for the batch symbol is still recoverable.
    let dims: DynMap = t.dims.iter().map(|(k, v)| (*k, *v)).collect();
    assert_eq!(shape[0].exec(&dims), Some(3));
}
