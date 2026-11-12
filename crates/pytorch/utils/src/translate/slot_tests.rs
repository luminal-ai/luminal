//! The program's arguments as slots. A symbolic integer is an argument of
//! its own, bound by value: no tensor has to carry it.

use serde_json::json;

use super::test_support::*;
use crate::boundary::SlotKind;
use crate::pt2_schema::{Argument, ScalarName, SymIntArg, SymIntValue, TensorRef};

fn int_ref(name: &str) -> TensorRef {
    TensorRef {
        as_tensor: None,
        as_tensors: None,
        as_sym_int: Some(ScalarName {
            as_name: name.to_string(),
        }),
        as_sym_float: None,
        as_sym_bool: None,
    }
}

fn int_operand(name: &str, value: &str) -> crate::pt2_schema::NodeInput {
    input(
        name,
        Argument::SymInt(SymIntArg {
            as_sym_int: SymIntValue {
                as_name: value.to_string(),
            },
        }),
    )
}

/// `t * n` with `n` an integer argument: torch hands the backend a `SymInt`
/// placeholder whose expression is a symbol no tensor axis carries.
fn times_an_integer(expr: &str) -> crate::pt2_parser::ParsedPT2 {
    let mut parsed = parsed_nodes(
        vec![node(
            "torch.ops.aten.mul.Tensor",
            vec![tensor_input("self", "x"), int_operand("other", "n")],
            &["out"],
        )],
        &[("x", FLOAT, &[4, 3]), ("out", FLOAT, &[4, 3])],
        &["x"],
        &["out"],
    );
    let graph = &mut parsed.program.graph_module.graph;
    graph.inputs.push(int_ref("n"));
    graph.sym_int_values.insert(
        "n".to_string(),
        json!({"as_expr": {"expr_str": expr, "hint": null}}),
    );
    parsed
}

#[test]
fn an_integer_argument_is_a_slot_that_defines_its_symbol() {
    let parsed = times_an_integer("Symbol('u0', integer=True)");
    let translation = crate::translate::translate(&parsed).expect("an integer argument translates");

    assert_eq!(
        translation.boundary.input_rows(),
        vec![
            ("tensor".to_string(), "x".to_string(), true, None),
            (
                "int".to_string(),
                "n".to_string(),
                true,
                Some("u0".to_string())
            ),
        ],
        "one slot per argument, in call order"
    );
    assert_eq!(
        translation.inputs.len(),
        1,
        "an integer argument has no storage"
    );
    assert_eq!(
        translation.boundary.output_rows(),
        vec![("tensor".to_string(), "program".to_string(), Some(0))]
    );

    let symbol = translation.symbols["u0"];
    assert_eq!(
        translation.int_binding("n", 5).expect("5 is a value"),
        Some((symbol, 5))
    );
    let refused = translation
        .int_binding("n", -3)
        .expect_err("a symbol's value is unsigned");
    let text = format!("{refused:#}");
    assert!(
        text.contains("\"n\"") && text.contains("-3") && text.contains("LUM-1337"),
        "{text}"
    );
    assert!(
        translation.int_binding("x", 1).is_err(),
        "a tensor argument is not an integer argument"
    );
}

/// A compound integer argument states no symbol: its value is not read, and
/// the symbols it names have to be defined by other arguments.
#[test]
fn a_compound_integer_argument_defines_no_symbol() {
    let mut parsed = times_an_integer("Symbol('u0', integer=True)");
    let graph = &mut parsed.program.graph_module.graph;
    graph.inputs.push(int_ref("twice"));
    graph.sym_int_values.insert(
        "twice".to_string(),
        json!({"as_expr": {"expr_str": "Mul(Integer(2), Symbol('u1', integer=True))"}}),
    );
    let translation = crate::translate::translate(&parsed).expect("translates");

    let slot = &translation.boundary.inputs[2];
    assert_eq!(
        (slot.kind, slot.read, &slot.binds),
        (SlotKind::Int, false, &None)
    );
    assert!(
        !translation.symbols.contains_key("u1"),
        "a compound argument defines no symbol"
    );
    assert_eq!(
        translation.int_binding("twice", -8).expect("not read"),
        None,
        "a value that defines nothing is not read, whatever it is"
    );
}

fn scalar_ref(name: &str, kind: SlotKind) -> TensorRef {
    let scalar = Some(ScalarName {
        as_name: name.to_string(),
    });
    TensorRef {
        as_tensor: None,
        as_tensors: None,
        as_sym_int: None,
        as_sym_float: if kind == SlotKind::Float {
            scalar.clone()
        } else {
            None
        },
        as_sym_bool: if kind == SlotKind::Bool { scalar } else { None },
    }
}

/// Every placeholder and every output leaf is a slot, whatever its kind.
/// An argument no node reads keeps its slot and has no storage in the
/// program: nothing states its layout or its device. A leaf that is an
/// argument returns that argument, a symbolic scalar leaf is an
/// expression, and a `None` leaf is a literal.
#[test]
fn every_placeholder_and_output_leaf_is_a_slot() {
    let mut parsed = times_an_integer("Symbol('u0', integer=True)");
    let graph = &mut parsed.program.graph_module.graph;
    // An unread tensor on another device, a saved float and a saved bool.
    graph.inputs.push(tensor_ref("elsewhere"));
    graph.tensor_values.insert(
        "elsewhere".to_string(),
        crate::pt2_schema::TensorMeta {
            dtype: FLOAT,
            sizes: sizes(&[2]),
            strides: None,
            storage_offset: None,
            layout: None,
            device: Some(crate::pt2_schema::Device {
                kind: "cuda".to_string(),
                index: Some(0),
            }),
        },
    );
    graph.inputs.push(scalar_ref("half", SlotKind::Float));
    graph.sym_float_values.insert(
        "half".to_string(),
        json!({"as_expr": {"expr_str": "FloatTrueDiv(ToFloat(Symbol('u0', integer=True)), Float('2.0', precision=53))"}}),
    );
    graph.inputs.push(scalar_ref("big", SlotKind::Bool));
    graph.sym_bool_values.insert(
        "big".to_string(),
        json!({"as_expr": {"expr_str": "StrictGreaterThan(Symbol('u0', integer=True), Integer(3))"}}),
    );
    // Leaves: the computed tensor, the tensor argument, the integer
    // argument, the saved float, and a missing gradient.
    graph.outputs = vec![
        tensor_ref("out"),
        tensor_ref("x"),
        int_ref("n"),
        scalar_ref("half", SlotKind::Float),
        tensor_ref(""),
    ];
    graph.outputs[4].as_tensor = None;
    let user_output = |arg: serde_json::Value| {
        serde_json::from_value(json!({"user_output": {"arg": arg}})).expect("an output spec")
    };
    parsed.program.graph_module.signature.output_specs = vec![
        user_output(json!({"as_tensor": {"name": "out"}})),
        user_output(json!({"as_tensor": {"name": "x"}})),
        user_output(json!({"as_sym_int": {"as_name": "n"}})),
        user_output(json!({"as_sym_float": {"as_name": "half"}})),
        user_output(json!({"as_none": true})),
    ];

    let translation = crate::translate::translate(&parsed).expect("translates");
    let rows = translation.boundary.input_rows();
    assert_eq!(
        rows.iter()
            .map(|(kind, name, read, _)| (kind.as_str(), name.as_str(), *read))
            .collect::<Vec<_>>(),
        vec![
            ("tensor", "x", true),
            ("int", "n", true),
            ("tensor", "elsewhere", false),
            ("float", "half", false),
            ("bool", "big", false),
        ]
    );
    assert_eq!(
        translation
            .inputs
            .iter()
            .map(|input| input.graph_name.as_str())
            .collect::<Vec<_>>(),
        vec!["x"],
        "only the tensor the program reads has storage"
    );
    assert_eq!(
        translation.boundary.output_rows(),
        vec![
            ("tensor".to_string(), "program".to_string(), Some(0)),
            ("tensor".to_string(), "input".to_string(), Some(0)),
            ("int".to_string(), "input".to_string(), Some(1)),
            ("float".to_string(), "input".to_string(), Some(3)),
            ("object".to_string(), "literal".to_string(), None),
        ]
    );
    assert_eq!(
        translation.outputs.len(),
        1,
        "only the computed leaf is written"
    );
}

/// A symbolic bool argument is the expression torch recorded for it, so a
/// node that reads it reads that expression. A symbolic float argument has
/// no expression this translator reads yet, and the node that reads it is
/// refused by name.
#[test]
fn a_symbolic_scalar_argument_is_read_as_its_expression() {
    let program = |operand: &str, kind: SlotKind, expr: &str| {
        let mut parsed = parsed_nodes(
            vec![node(
                "torch.ops.aten.mul.Tensor",
                vec![
                    tensor_input("self", "x"),
                    input(
                        "other",
                        if kind == SlotKind::Bool {
                            Argument::SymBool(crate::pt2_schema::SymBoolArg {
                                as_sym_bool: ScalarName {
                                    as_name: operand.to_string(),
                                },
                            })
                        } else {
                            Argument::SymFloat(crate::pt2_schema::SymFloatArg {
                                as_sym_float: ScalarName {
                                    as_name: operand.to_string(),
                                },
                            })
                        },
                    ),
                ],
                &["out"],
            )],
            &[("x", FLOAT, &[4, 3]), ("out", FLOAT, &[4, 3])],
            &["x"],
            &["out"],
        );
        let graph = &mut parsed.program.graph_module.graph;
        graph.inputs.push(int_ref("n"));
        graph.sym_int_values.insert(
            "n".to_string(),
            json!({"as_expr": {"expr_str": "Symbol('u0', integer=True)"}}),
        );
        graph.inputs.push(scalar_ref(operand, kind));
        let value = json!({"as_expr": {"expr_str": expr}});
        if kind == SlotKind::Bool {
            graph.sym_bool_values.insert(operand.to_string(), value);
        } else {
            graph.sym_float_values.insert(operand.to_string(), value);
        }
        parsed
    };

    let parsed = program(
        "big",
        SlotKind::Bool,
        "StrictGreaterThan(Symbol('u0', integer=True), Integer(3))",
    );
    let translation = crate::translate::translate(&parsed).expect("a bool argument is read");
    let slot = &translation.boundary.inputs[2];
    assert_eq!((slot.kind, slot.read), (SlotKind::Bool, true));

    let parsed = program(
        "half",
        SlotKind::Float,
        "FloatTrueDiv(ToFloat(Symbol('u0', integer=True)), Float('2.0', precision=53))",
    );
    let refused = crate::translate::translate(&parsed)
        .err()
        .expect("a float argument has no expression yet");
    let text = format!("{refused:#}");
    assert!(text.contains("half"), "{text}");
}

/// torch asserts a condition at run time with `_assert_scalar` over a
/// symbolic bool. One the declared range decides needs no check; one the
/// user wrote becomes a check of the plan, evaluated from the call's
/// symbol values with torch's own message.
#[test]
fn a_scalar_assertion_is_a_runtime_check() {
    let assertion = |name: &str, message: &str| {
        node(
            "torch.ops.aten._assert_scalar.default",
            vec![
                input(
                    "self",
                    Argument::SymBool(crate::pt2_schema::SymBoolArg {
                        as_sym_bool: ScalarName {
                            as_name: name.to_string(),
                        },
                    }),
                ),
                input("message", Argument::Other(json!({"as_string": message}))),
            ],
            &[],
        )
    };
    let mut parsed = times_an_integer("Symbol('u0', integer=True)");
    parsed.program.range_constraints.insert(
        "u0".to_string(),
        crate::pt2_schema::RangeConstraint {
            min_val: Some(2),
            max_val: Some(8),
        },
    );
    let graph = &mut parsed.program.graph_module.graph;
    graph.sym_bool_values.insert(
        "in_range".to_string(),
        json!({"as_expr": {"expr_str": "GreaterThan(Symbol('u0', integer=True), Integer(2))"}}),
    );
    graph.sym_bool_values.insert(
        "not_five".to_string(),
        json!({"as_expr": {"expr_str": "Unequality(Symbol('u0', integer=True), Integer(5))"}}),
    );
    graph.nodes.push(assertion("in_range", "u0 >= 2"));
    graph.nodes.push(assertion("not_five", "Ne(u0, 5)"));

    let translation = crate::translate::translate(&parsed).expect("translates");
    let checks = &translation.boundary.checks;
    assert_eq!(
        checks
            .iter()
            .map(|c| c.message.as_str())
            .collect::<Vec<_>>(),
        vec!["Ne(u0, 5)"],
        "the range already decides `u0 >= 2`"
    );

    let symbol = translation.symbols["u0"];
    let at = |value: usize| -> luminal::prelude::DynMap { [(symbol, value)].into_iter().collect() };
    assert!(translation.boundary.failed_check(&at(4)).is_none());
    assert_eq!(
        translation
            .boundary
            .failed_check(&at(5))
            .map(|check| check.message.as_str()),
        Some("Ne(u0, 5)")
    );
    assert!(
        translation
            .boundary
            .failed_check(&Default::default())
            .is_none(),
        "a check over a symbol the call did not bind is not evaluated"
    );
}
