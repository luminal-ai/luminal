//! Torch's per-tensor `device`. A translated program lives on one device:
//! its inputs must agree, and no value may sit anywhere else, which is how a
//! device move inside the graph is refused.

use std::collections::HashMap;
use std::fmt;

use anyhow::{Result, bail};

use crate::pt2_parser::ParsedPT2;
use crate::pt2_schema::Device;

/// A torch device, `cpu` or `cuda:0`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TorchDevice {
    pub kind: String,
    pub index: Option<i64>,
}

impl From<&Device> for TorchDevice {
    fn from(device: &Device) -> Self {
        Self {
            kind: device.kind.clone(),
            index: device.index,
        }
    }
}

impl fmt::Display for TorchDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.index {
            Some(index) => write!(f, "{}:{index}", self.kind),
            None => f.write_str(&self.kind),
        }
    }
}

/// The one device the program's tensors live on, or `None` when the export
/// states no device. Inputs that disagree, or any value on another device,
/// are refused by name.
pub fn program_device(parsed: &ParsedPT2) -> Result<Option<TorchDevice>> {
    let graph = &parsed.program.graph_module.graph;
    let device_of = |name: &str| -> Option<TorchDevice> {
        graph
            .tensor_values
            .get(name)
            .and_then(|meta| meta.device.as_ref())
            .map(TorchDevice::from)
    };

    let mut program: Option<(String, TorchDevice)> = None;
    for input in &graph.inputs {
        let Some(name) = input.value_name() else {
            continue;
        };
        let Some(device) = device_of(name) else {
            continue;
        };
        match &program {
            None => program = Some((name.to_string(), device)),
            Some((first, expected)) if *expected != device => bail!(
                "input {first:?} is on {expected} but input {name:?} is on {device}; a program's \
                 inputs must share one device"
            ),
            Some(_) => {}
        }
    }
    let Some((_, expected)) = program else {
        return Ok(None);
    };

    let mut producers: HashMap<&str, &str> = HashMap::new();
    for node in &graph.nodes {
        for output in &node.outputs {
            if let Some(name) = output.value_name() {
                producers.insert(name, node.target.as_str());
            }
        }
    }
    let mut names: Vec<&String> = graph.tensor_values.keys().collect();
    names.sort();
    for name in names {
        if let Some(device) = device_of(name)
            && device != expected
        {
            let producer = producers
                .get(name.as_str())
                .map_or_else(|| "a graph input".to_string(), |t| format!("`{t}`"));
            bail!(
                "tensor {name:?} ({producer}) is on {device} while the program is on {expected}; \
                 moving tensors between devices is not translated"
            );
        }
    }
    Ok(Some(expected))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pt2_schema::{DimInt, DimSize, ExportedProgram, GraphModule, Signature, TensorMeta};
    use crate::translate::test_support::*;

    fn device(kind: &str, index: Option<i64>) -> Device {
        Device {
            kind: kind.to_string(),
            index,
        }
    }

    /// `a + b -> y`, each tensor on the given device (`None` = unstated).
    fn program(devices: [Option<Device>; 3]) -> ParsedPT2 {
        let mut tensor_values = std::collections::HashMap::new();
        for (name, dev) in ["a", "b", "y"].into_iter().zip(devices) {
            tensor_values.insert(
                name.to_string(),
                TensorMeta {
                    dtype: FLOAT,
                    sizes: vec![DimSize::Int(DimInt { as_int: 4 })],
                    layout: Some(7),
                    device: dev,
                },
            );
        }
        ParsedPT2 {
            program: ExportedProgram {
                graph_module: GraphModule {
                    graph: crate::pt2_schema::Graph {
                        inputs: vec![tensor_ref("a"), tensor_ref("b")],
                        outputs: vec![tensor_ref("y")],
                        nodes: vec![node(
                            "torch.ops.aten.add.Tensor",
                            vec![tensor_input("self", "a"), tensor_input("other", "b")],
                            &["y"],
                        )],
                        tensor_values,
                        sym_int_values: std::collections::HashMap::new(),
                    },
                    signature: Signature {
                        input_specs: Vec::new(),
                        output_specs: Vec::new(),
                    },
                },
                range_constraints: std::collections::HashMap::new(),
            },
            constants_config: None,
            weights_config: None,
            archive_prefix: "test".to_string(),
            pt2_path: String::new(),
        }
    }

    #[test]
    fn a_program_on_one_device_reports_it() {
        let cuda = || Some(device("cuda", Some(0)));
        let found = program_device(&program([cuda(), cuda(), cuda()])).unwrap();
        assert_eq!(found.map(|d| d.to_string()), Some("cuda:0".to_string()));
    }

    #[test]
    fn an_export_that_states_no_device_reports_none() {
        assert_eq!(program_device(&program([None, None, None])).unwrap(), None);
    }

    #[test]
    fn inputs_on_different_devices_are_refused_by_name() {
        let err = program_device(&program([
            Some(device("cpu", None)),
            Some(device("cuda", Some(0))),
            None,
        ]))
        .unwrap_err()
        .to_string();
        assert!(
            err.contains("\"a\" is on cpu") && err.contains("\"b\" is on cuda:0"),
            "{err}"
        );
    }

    #[test]
    fn two_cuda_indices_are_different_devices() {
        let err = program_device(&program([
            Some(device("cuda", Some(0))),
            Some(device("cuda", Some(1))),
            None,
        ]))
        .unwrap_err()
        .to_string();
        assert!(err.contains("cuda:0") && err.contains("cuda:1"), "{err}");
    }

    #[test]
    fn a_value_moved_to_another_device_is_refused_with_its_producer() {
        let cpu = || Some(device("cpu", None));
        let err = program_device(&program([cpu(), cpu(), Some(device("cuda", Some(0)))]))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("\"y\"")
                && err.contains("torch.ops.aten.add.Tensor")
                && err.contains("cuda:0"),
            "{err}"
        );
    }

    #[test]
    fn translation_refuses_a_device_move() {
        let cpu = || Some(device("cpu", None));
        let err = match crate::translate::translate(&program([
            cpu(),
            cpu(),
            Some(device("cuda", Some(0))),
        ])) {
            Ok(_) => panic!("a device move must refuse"),
            Err(err) => err.to_string(),
        };
        assert!(
            err.contains("moving tensors between devices is not translated"),
            "{err}"
        );
    }
}
