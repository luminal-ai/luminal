//! Torch's per-tensor `device`. A translated program lives on one device:
//! its inputs must agree, and no value may sit anywhere else, which is how a
//! device move inside the graph is refused.

use std::collections::{HashMap, HashSet};
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
    let is_cpu_scalar = |name: &str| {
        graph.tensor_values.get(name).is_some_and(|meta| {
            meta.sizes.is_empty()
                && meta
                    .device
                    .as_ref()
                    .is_some_and(|device| device.kind == "cpu")
        })
    };
    let mut host_scalar_carriers: HashSet<&str> = graph
        .inputs
        .iter()
        .filter_map(|input| input.value_name())
        .filter(|name| is_cpu_scalar(name))
        .collect();
    // Export may preserve the scalar's CPU placement through value-only
    // carrier operations before it is consumed by a CUDA op. These operations
    // do not compute new scalar semantics; the translator preserves the exact
    // value while the backend stages its boundary source.
    for node in &graph.nodes {
        let carries_value = matches!(
            node.target.as_str(),
            "torch.ops.aten.clone.default"
                | "torch.ops.aten.alias.default"
                | "torch.ops.aten.detach.default"
                | "torch.ops.aten.lift_fresh_copy.default"
        ) && node.inputs.iter().any(|input| {
            input
                .arg
                .as_value_name()
                .is_some_and(|name| host_scalar_carriers.contains(name))
        });
        if carries_value {
            host_scalar_carriers.extend(
                node.outputs
                    .iter()
                    .filter_map(|output| output.value_name())
                    .filter(|name| is_cpu_scalar(name)),
            );
        }
    }

    // PyTorch permits a rank-zero CPU tensor as a scalar operand of a CUDA
    // operation. It is a boundary staging exception, not the program's
    // execution device, so choose the device from ordinary inputs first.
    let mut program: Option<(String, TorchDevice)> = None;
    for input in graph.inputs.iter().filter(|input| {
        input
            .value_name()
            .is_none_or(|name| !host_scalar_carriers.contains(name))
    }) {
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
    if program.is_none() {
        program = graph.inputs.iter().find_map(|input| {
            let name = input.value_name()?;
            Some((name.to_string(), device_of(name)?))
        });
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
            if expected.kind == "cuda" && host_scalar_carriers.contains(name.as_str()) {
                continue;
            }
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
                    strides: None,
                    storage_offset: None,
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
    fn rank_zero_cpu_input_is_a_cuda_scalar_operand_not_the_program_device() {
        let mut parsed = program([
            Some(device("cpu", None)),
            Some(device("cuda", Some(0))),
            Some(device("cuda", Some(0))),
        ]);
        parsed
            .program
            .graph_module
            .graph
            .tensor_values
            .get_mut("a")
            .unwrap()
            .sizes
            .clear();

        let found = program_device(&parsed).unwrap();
        assert_eq!(found.map(|d| d.to_string()), Some("cuda:0".to_string()));
    }

    #[test]
    fn value_only_carriers_keep_a_staged_cpu_scalar_legal() {
        let mut parsed = program([
            Some(device("cpu", None)),
            Some(device("cuda", Some(0))),
            Some(device("cuda", Some(0))),
        ]);
        let graph = &mut parsed.program.graph_module.graph;
        graph.tensor_values.get_mut("a").unwrap().sizes.clear();
        graph.tensor_values.insert(
            "a_clone".into(),
            TensorMeta {
                dtype: FLOAT,
                sizes: vec![],
                strides: None,
                storage_offset: None,
                layout: Some(7),
                device: Some(device("cpu", None)),
            },
        );
        graph.nodes = vec![
            node(
                "torch.ops.aten.clone.default",
                vec![tensor_input("input", "a")],
                &["a_clone"],
            ),
            node(
                "torch.ops.aten.add.Tensor",
                vec![tensor_input("self", "b"), tensor_input("other", "a_clone")],
                &["y"],
            ),
        ];

        let found = program_device(&parsed).unwrap();
        assert_eq!(found.map(|d| d.to_string()), Some("cuda:0".to_string()));
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
