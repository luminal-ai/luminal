//! Translator boundary table for a mutated module buffer.
//!
//! The other translator tests read `.pt2` fixtures produced by a torch
//! environment; this one is a hand-written PT2 program, small enough to state
//! the whole signature inline.

use luminal_pytorch_utils::{ParsedPT2, translate};

/// `self.cache.add_(x); return self.cache * 1.0` after functionalization: the
/// mutation leaves as an extra graph output whose spec is substituted in.
const PROGRAM: &str = r#"{
  "graph_module": {
    "graph": {
      "inputs": [{"as_tensor":{"name":"b_cache"}},{"as_tensor":{"name":"x"}}],
      "outputs": [{"as_tensor":{"name":"add_1"}},{"as_tensor":{"name":"mul"}}],
      "nodes": [
        {"target":"torch.ops.aten.add.Tensor",
         "inputs":[{"name":"self","arg":{"as_tensor":{"name":"b_cache"}},"kind":1},
                   {"name":"other","arg":{"as_tensor":{"name":"x"}},"kind":1}],
         "outputs":[{"as_tensor":{"name":"add_1"}}]},
        {"target":"torch.ops.aten.mul.Tensor",
         "inputs":[{"name":"self","arg":{"as_tensor":{"name":"add_1"}},"kind":1},
                   {"name":"other","arg":{"as_float":1.0},"kind":1}],
         "outputs":[{"as_tensor":{"name":"mul"}}]}
      ],
      "tensor_values": {
        "b_cache":{"dtype":7,"sizes":[{"as_int":4}]},
        "x":{"dtype":7,"sizes":[{"as_int":4}]},
        "add_1":{"dtype":7,"sizes":[{"as_int":4}]},
        "mul":{"dtype":7,"sizes":[{"as_int":4}]}
      }
    },
    "signature": {
      "input_specs": [
        {"buffer":{"arg":{"name":"b_cache"},"buffer_name":"cache","persistent":true}},
        {"user_input":{"arg":{"as_tensor":{"name":"x"}}}}
      ],
      "output_specs": [
        MUTATION_SPEC,
        {"user_output":{"arg":{"as_tensor":{"name":"mul"}}}}
      ]
    }
  }
}"#;

/// `self.cache.add_(x); return self.cache` after functionalization: the mutated
/// value leaves twice — once as the buffer mutation, once as the user output.
const RETURNS_MUTATED: &str = r#"{
  "graph_module": {
    "graph": {
      "inputs": [{"as_tensor":{"name":"b_cache"}},{"as_tensor":{"name":"x"}}],
      "outputs": [{"as_tensor":{"name":"add_1"}},{"as_tensor":{"name":"add_1"}}],
      "nodes": [
        {"target":"torch.ops.aten.add.Tensor",
         "inputs":[{"name":"self","arg":{"as_tensor":{"name":"b_cache"}},"kind":1},
                   {"name":"other","arg":{"as_tensor":{"name":"x"}},"kind":1}],
         "outputs":[{"as_tensor":{"name":"add_1"}}]}
      ],
      "tensor_values": {
        "b_cache":{"dtype":7,"sizes":[{"as_int":4}]},
        "x":{"dtype":7,"sizes":[{"as_int":4}]},
        "add_1":{"dtype":7,"sizes":[{"as_int":4}]}
      }
    },
    "signature": {
      "input_specs": [
        {"buffer":{"arg":{"name":"b_cache"},"buffer_name":"cache","persistent":true}},
        {"user_input":{"arg":{"as_tensor":{"name":"x"}}}}
      ],
      "output_specs": [
        {"buffer_mutation":{"arg":{"name":"add_1"},"buffer_name":"cache"}},
        {"user_output":{"arg":{"as_tensor":{"name":"add_1"}}}}
      ]
    }
  }
}"#;

/// A `buffer_mutation` spec, naming the buffer by module FQN (`cache`) — not
/// by its graph input name (`b_cache`).
fn buffer_mutation(buffer_name: &str) -> String {
    format!(r#"{{"buffer_mutation":{{"arg":{{"name":"add_1"}},"buffer_name":"{buffer_name}"}}}}"#)
}

fn parse(json: &str) -> ParsedPT2 {
    ParsedPT2 {
        program: serde_json::from_str(json).expect("fixture must parse"),
        constants_config: None,
        weights_config: None,
        archive_prefix: String::new(),
        pt2_path: String::new(),
    }
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

/// The mutation writes the buffer's storage, so its target is the buffer's
/// graph input name, resolved from the FQN the spec carries — and it is a
/// writeback, not a tensor the caller gets back.
#[test]
fn buffer_mutation_output_targets_the_buffer_input() {
    let translation = translate(&program(&buffer_mutation("cache"))).expect("translate");

    let mutation = output(&translation, "add_1");
    assert_eq!(mutation.mutation_target.as_deref(), Some("b_cache"));
    assert!(!mutation.returned);
    let returned = output(&translation, "mul");
    assert!(returned.mutation_target.is_none());
    assert!(returned.returned);
}

/// A buffer mutation is a user-input mutation whose target is spelled by FQN:
/// the two specs must produce the same boundary treatment.
#[test]
fn buffer_mutation_is_treated_like_a_user_input_mutation() {
    let buffer = translate(&program(&buffer_mutation("cache"))).expect("translate buffer");
    let user = translate(&program(
        r#"{"user_input_mutation":{"arg":{"name":"add_1"},"user_input_name":"x"}}"#,
    ))
    .expect("translate user input");

    let buffer = output(&buffer, "add_1");
    let user = output(&user, "add_1");
    assert_eq!(buffer.mutation_target.as_deref(), Some("b_cache"));
    assert_eq!(user.mutation_target.as_deref(), Some("x"));
    assert!(!buffer.returned);
    assert!(!user.returned);
    assert_eq!(buffer.dtype, user.dtype);
}

/// A buffer FQN with no matching input spec has no storage to write into.
#[test]
fn buffer_mutation_naming_an_unknown_buffer_refuses() {
    let Err(error) = translate(&program(&buffer_mutation("other_cache"))) else {
        panic!("unknown buffer must refuse");
    };
    assert!(format!("{error:#}").contains("other_cache"), "{error:#}");
}

/// A mutated buffer the model also returns is ONE boundary entry that both
/// writes the buffer's storage and is returned, so the caller gets the
/// mutated tensor itself rather than a second copy of it.
#[test]
fn a_returned_buffer_mutation_is_one_output() {
    let translation = translate(&parse(RETURNS_MUTATED)).expect("translate");

    assert_eq!(
        translation.outputs.len(),
        1,
        "one boundary entry, not a writeback plus a copy"
    );
    let mutation = output(&translation, "add_1");
    assert_eq!(mutation.mutation_target.as_deref(), Some("b_cache"));
    assert!(mutation.returned);
}
