//! Translator boundary table for a mutated graph input.
//!
//! AOTAutograd lifts a module's buffers into user inputs, so a mutated
//! buffer reaches the translator as a `user_input_mutation` of the input it
//! was lifted into. These are hand-written programs, small enough to state
//! the whole signature inline, read through the in-process hand-off.

use luminal_pytorch_utils::{ParsedPT2, parse_program, translate};

/// `self.cache.add_(x); return self.cache * 1.0` after AOTAutograd: the
/// buffer is the user input `cache`, and the mutation leaves as an extra
/// graph output whose spec is substituted in.
const PROGRAM: &str = r#"{
  "graph_module": {
    "graph": {
      "inputs": [{"as_tensor":{"name":"cache"}},{"as_tensor":{"name":"x"}}],
      "outputs": [{"as_tensor":{"name":"add_1"}},{"as_tensor":{"name":"mul"}}],
      "nodes": [
        {"target":"torch.ops.aten.add.Tensor",
         "inputs":[{"name":"self","arg":{"as_tensor":{"name":"cache"}},"kind":1},
                   {"name":"other","arg":{"as_tensor":{"name":"x"}},"kind":1}],
         "outputs":[{"as_tensor":{"name":"add_1"}}]},
        {"target":"torch.ops.aten.mul.Tensor",
         "inputs":[{"name":"self","arg":{"as_tensor":{"name":"add_1"}},"kind":1},
                   {"name":"other","arg":{"as_float":1.0},"kind":1}],
         "outputs":[{"as_tensor":{"name":"mul"}}]}
      ],
      "tensor_values": {
        "cache":{"dtype":7,"sizes":[{"as_int":4}]},
        "x":{"dtype":7,"sizes":[{"as_int":4}]},
        "add_1":{"dtype":7,"sizes":[{"as_int":4}]},
        "mul":{"dtype":7,"sizes":[{"as_int":4}]}
      }
    },
    "signature": {
      "input_specs": [
        {"user_input":{"arg":{"as_tensor":{"name":"cache"}}}},
        {"user_input":{"arg":{"as_tensor":{"name":"x"}}}}
      ],
      "output_specs": [
        MUTATION_SPEC,
        {"user_output":{"arg":{"as_tensor":{"name":"mul"}}}}
      ]
    }
  }
}"#;

/// `self.cache.add_(x); return self.cache` after AOTAutograd: the mutated
/// value leaves twice — once as the input mutation, once as the user output.
const RETURNS_MUTATED: &str = r#"{
  "graph_module": {
    "graph": {
      "inputs": [{"as_tensor":{"name":"cache"}},{"as_tensor":{"name":"x"}}],
      "outputs": [{"as_tensor":{"name":"add_1"}},{"as_tensor":{"name":"add_1"}}],
      "nodes": [
        {"target":"torch.ops.aten.add.Tensor",
         "inputs":[{"name":"self","arg":{"as_tensor":{"name":"cache"}},"kind":1},
                   {"name":"other","arg":{"as_tensor":{"name":"x"}},"kind":1}],
         "outputs":[{"as_tensor":{"name":"add_1"}}]}
      ],
      "tensor_values": {
        "cache":{"dtype":7,"sizes":[{"as_int":4}]},
        "x":{"dtype":7,"sizes":[{"as_int":4}]},
        "add_1":{"dtype":7,"sizes":[{"as_int":4}]}
      }
    },
    "signature": {
      "input_specs": [
        {"user_input":{"arg":{"as_tensor":{"name":"cache"}}}},
        {"user_input":{"arg":{"as_tensor":{"name":"x"}}}}
      ],
      "output_specs": [
        {"user_input_mutation":{"arg":{"name":"add_1"},"user_input_name":"cache"}},
        {"user_output":{"arg":{"as_tensor":{"name":"add_1"}}}}
      ]
    }
  }
}"#;

/// A `user_input_mutation` spec writing `add_1` into the graph input
/// `target`.
fn input_mutation(target: &str) -> String {
    format!(
        r#"{{"user_input_mutation":{{"arg":{{"name":"add_1"}},"user_input_name":"{target}"}}}}"#
    )
}

fn parse(json: &str) -> ParsedPT2 {
    parse_program(json.as_bytes()).expect("fixture must parse")
}

fn program(mutation_spec: &str) -> ParsedPT2 {
    parse(&PROGRAM.replace("MUTATION_SPEC", mutation_spec))
}

fn output<'a>(
    translation: &'a luminal_pytorch_utils::Translation,
    name: &str,
) -> &'a luminal_pytorch_utils::TranslatedOutput {
    translation
        .outputs
        .iter()
        .find(|output| output.graph_name == name)
        .unwrap_or_else(|| panic!("no output named {name}"))
}

/// The mutation writes the input's storage, so its target is the input's
/// graph name — and it is a writeback, not a tensor the caller gets back.
#[test]
fn input_mutation_output_targets_the_mutated_input() {
    let translation = translate(&program(&input_mutation("cache"))).expect("translate");

    let mutation = output(&translation, "add_1");
    assert_eq!(mutation.mutation_target.as_deref(), Some("cache"));
    assert!(!mutation.returned);
    let returned = output(&translation, "mul");
    assert!(returned.mutation_target.is_none());
    assert!(returned.returned);
}

/// A mutation naming no graph input has no storage to write into.
#[test]
fn input_mutation_naming_an_unknown_input_refuses() {
    let Err(error) = translate(&program(&input_mutation("other_cache"))) else {
        panic!("unknown input must refuse");
    };
    assert!(format!("{error:#}").contains("other_cache"), "{error:#}");
}

/// A mutated input the model also returns is ONE boundary entry that both
/// writes the input's storage and is returned, so the caller gets the
/// mutated tensor itself rather than a second copy of it.
#[test]
fn a_returned_input_mutation_is_one_output() {
    let translation = translate(&parse(RETURNS_MUTATED)).expect("translate");

    assert_eq!(
        translation.outputs.len(),
        1,
        "one boundary entry, not a writeback plus a copy"
    );
    let mutation = output(&translation, "add_1");
    assert_eq!(mutation.mutation_target.as_deref(), Some("cache"));
    assert!(mutation.returned);
}

/// The `torch.export` spelling of the same program — the cache held as a
/// `buffer` input and written by a `buffer_mutation` output — is refused by
/// name, never read as a user input and a returned user output.
#[test]
fn the_export_spelling_of_a_buffer_mutation_is_refused_by_name() {
    let held = PROGRAM
        .replace(
            r#"{"user_input":{"arg":{"as_tensor":{"name":"cache"}}}}"#,
            r#"{"buffer":{"arg":{"name":"cache"},"buffer_name":"cache","persistent":true}}"#,
        )
        .replace("MUTATION_SPEC", &input_mutation("cache"));
    let error = parse_program(held.as_bytes()).expect_err("a held buffer is refused");
    assert!(format!("{error:#}").contains("`buffer`"), "{error:#}");

    let written = PROGRAM.replace(
        "MUTATION_SPEC",
        r#"{"buffer_mutation":{"arg":{"name":"add_1"},"buffer_name":"cache"}}"#,
    );
    let error = parse_program(written.as_bytes()).expect_err("a buffer mutation is refused");
    assert!(
        format!("{error:#}").contains("`buffer_mutation`"),
        "{error:#}"
    );
}
