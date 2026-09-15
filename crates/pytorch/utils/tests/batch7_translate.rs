//! Translation tests for port batch 7 (elementwise specials / squeeze /
//! triangle) and batch 8 (any / var_mean / addmm-family).
//!
//! Fixtures live in `<workspace>/target/luminal_fixtures` and are produced
//! by `torch.export.save`; each test skips when its fixture is absent.

use luminal::prelude::IntExpr;
use luminal_pytorch_utils::{Translation, parse_pt2, translate};

/// A translated boundary shape is symbolic; these fixtures are static, so
/// every dim must resolve to a literal.
fn concrete(shape: &[IntExpr]) -> Vec<usize> {
    shape
        .iter()
        .map(|dim| dim.to_usize().expect("static fixture dim"))
        .collect()
}

fn translate_fixture(name: &str) -> Option<Translation> {
    let path = format!(
        "{}/../../../target/luminal_fixtures/{name}.pt2",
        env!("CARGO_MANIFEST_DIR")
    );
    if !std::path::Path::new(&path).exists() {
        eprintln!("skipping: {path} not found");
        return None;
    }
    let parsed = parse_pt2(&path).expect("parse pt2");
    Some(translate(&parsed).unwrap_or_else(|e| panic!("translate {name}: {e:#}")))
}

fn assert_translates(name: &str, output_shapes: &[Vec<usize>]) {
    let Some(t) = translate_fixture(name) else {
        return;
    };
    let shapes: Vec<Vec<usize>> = t.outputs.iter().map(|o| concrete(&o.shape)).collect();
    assert_eq!(shapes, output_shapes, "{name} output shapes");
}

#[test]
fn binary_specials_translate() {
    for name in [
        "b7_atan2",
        "b7_copysign",
        "b7_fmax",
        "b7_fmin",
        "b7_hypot",
        "b7_exp2",
        "b7_isnan",
        "b7_leaky",
        "b7_tril",
        "b7_triu",
    ] {
        let Some(t) = translate_fixture(name) else {
            continue;
        };
        assert_eq!(t.outputs.len(), 1, "{name}");
        assert_eq!(concrete(&t.outputs[0].shape), vec![4, 4], "{name}");
    }
}

#[test]
fn copysign_scalar_translates() {
    let Some(t) = translate_fixture("b7_copysign_scalar") else {
        return;
    };
    assert_eq!(concrete(&t.outputs[0].shape), vec![4, 4]);
}

#[test]
fn bitwise_translates() {
    for name in ["b7_bitand", "b7_bitor"] {
        let Some(t) = translate_fixture(name) else {
            continue;
        };
        assert_eq!(concrete(&t.outputs[0].shape), vec![4, 4], "{name}");
    }
}

#[test]
fn gcd_translates() {
    // gcd's select-based Euclidean unroll now translates (and plans once the
    // caller attests non-negative input ranges).
    let path = format!(
        "{}/../../../target/luminal_fixtures/b7_gcd.pt2",
        env!("CARGO_MANIFEST_DIR")
    );
    if !std::path::Path::new(&path).exists() {
        return;
    }
    let parsed = parse_pt2(&path).expect("parse pt2");
    let translation = translate(&parsed).expect("gcd translates");
    assert!(!translation.outputs.is_empty());
}

#[test]
fn log2_translates() {
    let Some(t) = translate_fixture("b7_log2") else {
        return;
    };
    assert_eq!(concrete(&t.outputs[0].shape), vec![4, 4]);
}

#[test]
fn squeeze_forms_translate() {
    assert_translates("b7_squeeze", &[vec![3, 4]]);
    assert_translates("b7_squeeze_dims", &[vec![3, 4]]);
}

#[test]
fn any_forms_translate() {
    assert_translates("b8_any", &[vec![]]);
    assert_translates("b8_any_dim", &[vec![4]]);
}

#[test]
fn var_mean_returns_two_outputs() {
    assert_translates("b8_var_mean", &[vec![4], vec![4]]);
}

#[test]
fn addmm_family_translates() {
    assert_translates("b8_addmm", &[vec![4, 4]]);
    assert_translates("b8_addbmm", &[vec![4, 4]]);
    assert_translates("b8_addmv", &[vec![4]]);
}

#[test]
fn tiny_llama_preprocessed_translates() {
    // tiny_llama_pp.pt2 is a tiny HF Llama exported after the old backend's
    // preprocessing pipeline (guards dropped, dead data-dependent ops
    // removed, decompositions applied). Translating it proves the op set
    // that pipeline leaves behind is fully covered.
    let Some(t) = translate_fixture("tiny_llama_pp") else {
        return;
    };
    assert_eq!(t.outputs.len(), 1);
    assert!(
        t.inputs
            .iter()
            .any(|i| { matches!(i.kind, luminal_pytorch_utils::InputKind::UserInput { .. }) })
    );
}
