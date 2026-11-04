//! `aten.gelu`'s `approximate` selects the function: the erf form by
//! default, PyTorch's tanh approximation when `"tanh"`.
use super::test_support::*;

fn gelu(approximate: Option<&str>) -> String {
    let mut inputs = vec![tensor_input("self", "x")];
    if let Some(approximate) = approximate {
        inputs.push(input(
            "approximate",
            crate::pt2_schema::Argument::Other(serde_json::json!({ "as_string": approximate })),
        ));
    }
    let t = translate_one(
        node("torch.ops.aten.gelu.default", inputs, &["out"]),
        &[("x", FLOAT, &[4, 8]), ("out", FLOAT, &[4, 8])],
        &["x"],
        &["out"],
    );
    t.graph.logical.render_all().unwrap()
}

/// A recorded f32 constant renders as its f64 value.
fn constant(value: f32) -> String {
    format!("(LogicalConstant {})", value as f64)
}

#[test]
fn the_default_and_none_are_the_erf_form() {
    // the Abramowitz & Stegun constant of the erf approximation
    for approximate in [None, Some("none")] {
        let rendered = gelu(approximate);
        assert!(rendered.contains(&constant(0.3275911)), "{approximate:?}: {rendered}");
        assert!(!rendered.contains(&constant(0.044715)), "{approximate:?}: {rendered}");
    }
}

#[test]
fn tanh_is_the_tanh_approximation() {
    let rendered = gelu(Some("tanh"));
    assert!(rendered.contains(&constant(0.044715)), "{rendered}");
    assert!(!rendered.contains(&constant(0.3275911)), "{rendered}");
}
