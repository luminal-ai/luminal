//! Torch's tensor `layout`: the storage kind. Only `torch.strided` describes
//! memory the way this frontend reads it (sizes, strides, storage offset), so
//! every other kind is refused before translation.

use std::fmt;

/// The `layout` enum of torch's export schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TorchLayout {
    Unknown,
    SparseCoo,
    SparseCsr,
    SparseCsc,
    SparseBsr,
    SparseBsc,
    Mkldnn,
    Strided,
}

impl TorchLayout {
    /// The layout for a schema code; `Err` carries a code the schema does not
    /// define.
    pub fn from_code(code: u32) -> Result<Self, u32> {
        Ok(match code {
            0 => Self::Unknown,
            1 => Self::SparseCoo,
            2 => Self::SparseCsr,
            3 => Self::SparseCsc,
            4 => Self::SparseBsr,
            5 => Self::SparseBsc,
            6 => Self::Mkldnn,
            7 => Self::Strided,
            other => return Err(other),
        })
    }
}

impl fmt::Display for TorchLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Unknown => "unknown layout",
            Self::SparseCoo => "torch.sparse_coo",
            Self::SparseCsr => "torch.sparse_csr",
            Self::SparseCsc => "torch.sparse_csc",
            Self::SparseBsr => "torch.sparse_bsr",
            Self::SparseBsc => "torch.sparse_bsc",
            Self::Mkldnn => "torch._mkldnn",
            Self::Strided => "torch.strided",
        })
    }
}

/// Refuse a tensor whose declared layout is not `torch.strided`. An export
/// that states no layout is read as strided.
pub fn check_strided(name: &str, layout: Option<u32>) -> anyhow::Result<()> {
    let Some(code) = layout else {
        return Ok(());
    };
    match TorchLayout::from_code(code) {
        Ok(TorchLayout::Strided) => Ok(()),
        Ok(layout) => anyhow::bail!(
            "tensor {name:?} is exported with layout {layout} (code {code}); only torch.strided \
             tensors translate"
        ),
        Err(code) => anyhow::bail!(
            "tensor {name:?} is exported with layout code {code}, which torch's export schema \
             does not define; only torch.strided tensors translate"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_schema_code_maps_and_only_seven_is_strided() {
        for code in 0..=7u32 {
            let layout = TorchLayout::from_code(code).unwrap();
            assert_eq!(layout == TorchLayout::Strided, code == 7, "{layout}");
        }
        assert_eq!(TorchLayout::from_code(8), Err(8));
    }

    #[test]
    fn strided_and_unstated_layouts_pass() {
        check_strided("x", Some(7)).unwrap();
        check_strided("x", None).unwrap();
    }

    #[test]
    fn a_sparse_layout_is_refused_by_name() {
        let err = check_strided("weights", Some(2)).unwrap_err().to_string();
        assert!(
            err.contains("\"weights\"") && err.contains("torch.sparse_csr"),
            "{err}"
        );
    }

    #[test]
    fn translation_refuses_a_program_with_a_sparse_tensor() {
        use crate::pt2_schema::{ExportedProgram, GraphModule, Signature, TensorMeta};
        use crate::translate::test_support::*;
        use std::collections::HashMap;
        let mut tensor_values = HashMap::new();
        tensor_values.insert(
            "a".to_string(),
            TensorMeta {
                dtype: FLOAT,
                sizes: sizes(&[4]),
                layout: Some(1),
                device: None,
            },
        );
        tensor_values.insert(
            "z".to_string(),
            TensorMeta {
                dtype: FLOAT,
                sizes: sizes(&[4]),
                layout: Some(7),
                device: None,
            },
        );
        let program = ExportedProgram {
            graph_module: GraphModule {
                graph: crate::pt2_schema::Graph {
                    inputs: vec![tensor_ref("a")],
                    outputs: vec![tensor_ref("z")],
                    nodes: vec![node(
                        "torch.ops.aten.neg.default",
                        vec![tensor_input("self", "a")],
                        &["z"],
                    )],
                    tensor_values,
                    sym_int_values: HashMap::new(),
                },
                signature: Signature {
                    input_specs: Vec::new(),
                    output_specs: Vec::new(),
                },
            },
            range_constraints: HashMap::new(),
        };
        let parsed = crate::pt2_parser::ParsedPT2 {
            program,
            constants_config: None,
            weights_config: None,
            archive_prefix: "test".to_string(),
            pt2_path: String::new(),
        };
        let err = match crate::translate::translate(&parsed) {
            Ok(_) => panic!("a sparse tensor must refuse"),
            Err(err) => err.to_string(),
        };
        assert!(
            err.contains("\"a\"") && err.contains("torch.sparse_coo"),
            "{err}"
        );
    }

    #[test]
    fn an_undefined_code_is_refused() {
        let err = check_strided("x", Some(42)).unwrap_err().to_string();
        assert!(
            err.contains("code 42") && err.contains("does not define"),
            "{err}"
        );
    }
}
