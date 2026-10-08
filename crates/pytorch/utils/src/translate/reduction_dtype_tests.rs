//! ATen's resolved result dtype is an input conversion for these operations.
//! Check the logical contract even for dtypes without reference kernels.

use luminal::prelude::*;
use luminal_reference::{ReferenceBindings, assembled_program};

use super::Translation;
use super::test_support::*;
use crate::declaration::Placement;
use crate::declared_dtype::declare_dtypes;
use crate::dtype::TorchDType;

const TARGETS: &[&str] = &[
    "sum.default",
    "sum.dim_IntList",
    "prod.default",
    "prod.dim_int",
    "cumsum.default",
    "cumprod.default",
];

fn case(
    target: &str,
    input: TorchDType,
    output: TorchDType,
    scalar: bool,
    explicit: bool,
) -> Translation {
    let scan = target.starts_with("cum");
    let full = matches!(target, "sum.default" | "prod.default");
    let mut args = vec![tensor_input("self", "x")];
    if !full {
        args.push(if target == "sum.dim_IntList" {
            ints("dim", &[-1])
        } else {
            scalar_int("dim", -1)
        });
        if !scan {
            args.push(scalar_bool("keepdim", true));
        }
    }
    // Omitted dtype, rather than an explicit cast, is the reported FX graph.
    if explicit {
        args.push(scalar_type("dtype", output.code()));
    }
    let in_shape: &[i64] = if scalar { &[] } else { &[2, 3] };
    let out_shape: &[i64] = if scalar || full {
        &[]
    } else if scan {
        in_shape
    } else {
        &[2, 1]
    };
    translate_one(
        node(&format!("torch.ops.aten.{target}"), args, &["y"]),
        &[
            ("x", input.code(), in_shape),
            ("y", output.code(), out_shape),
        ],
        &["x"],
        &["y"],
    )
}

fn assert_declared_dtypes_agree(t: &Translation) {
    for row in &t.declared_dtypes {
        assert_eq!(
            t.graph.logical.petgraph()[row.tensor].dtype,
            row.dtype,
            "{}",
            row.graph_name
        );
    }
    let outputs: Vec<_> = t.outputs.iter().map(|output| output.tensor).collect();
    let mut bindings = ReferenceBindings::dense(&t.graph.logical, &outputs);
    declare_dtypes(&mut bindings, t, Placement::Beginning);
    let bound = bindings.bind(&t.graph.logical).expect("bind");
    let program = format!("{}\n{}", assembled_program(), bound.text());
    luminal::egglog_snippet::new_egraph()
        .parse_and_run_program(None, &program)
        .expect("propagation and strict declared dtype facts must agree");
}

#[test]
fn default_reduction_and_scan_dtypes_include_scalars() {
    use TorchDType::*;
    for target in TARGETS {
        for input in [Bool, Byte, Char, Short, Int, Long, Float, Half, BFloat16] {
            let output = match input {
                Bool | Byte | Char | Short | Int | Long => Long,
                dtype => dtype,
            };
            for scalar in [false, true] {
                assert_declared_dtypes_agree(&case(target, input, output, scalar, false));
            }
        }
    }
}

#[test]
fn explicit_reduction_and_scan_dtypes_override_default_promotion() {
    use TorchDType::*;
    for target in TARGETS {
        for (input, output) in [
            (Bool, Float),
            (Int, Float),
            (Float, Int),
            (Long, Int),
            (Float, BFloat16),
        ] {
            for scalar in [false, true] {
                assert_declared_dtypes_agree(&case(target, input, output, scalar, true));
            }
        }
    }
    for target in &TARGETS[..4] {
        for scalar in [false, true] {
            assert_declared_dtypes_agree(&case(target, Float, Bool, scalar, true));
        }
    }
}

#[test]
fn default_floating_scans_do_not_add_opmath_widening() {
    for target in ["cumsum.default", "cumprod.default"] {
        for dtype in [TorchDType::Float, TorchDType::Half, TorchDType::BFloat16] {
            let t = case(target, dtype, dtype, false, false);
            assert!(
                !t.graph
                    .logical
                    .petgraph()
                    .node_weights()
                    .any(|node| matches!(node.op, LogicalOp::Cast(_)))
            );
        }
    }
}

#[test]
fn argmax_int64_result_does_not_convert_float_input_to_int64() {
    let t = translate_one(
        node(
            "torch.ops.aten.argmax.default",
            vec![tensor_input("self", "x"), scalar_int("dim", -1)],
            &["y"],
        ),
        &[("x", FLOAT, &[3]), ("y", TorchDType::Long.code(), &[])],
        &["x"],
        &["y"],
    );
    assert_declared_dtypes_agree(&t);
    assert!(
        !t.graph
            .logical
            .petgraph()
            .node_weights()
            .any(|node| matches!(node.op, LogicalOp::TruncCast(_)))
    );
}
