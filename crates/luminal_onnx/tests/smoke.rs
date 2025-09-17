use luminal::prelude::*;
use luminal_onnx::import_onnx;
use luminal_onnx::onnx::proto as onnx;

fn build_linear_softmax_model(n: i64, in_dim: i64, out_dim: i64) -> onnx::ModelProto {
    use onnx::*;
    // Shapes
    let x_vi = ValueInfoProto {
        name: "X".into(),
        r#type: Some(TypeProto {
            value: Some(type_proto::Value::TensorType(TypeProtoTensor {
                elem_type: onnx::tensor_proto::DataType::Float as i32,
                shape: Some(TensorShapeProto {
                    dim: vec![
                        TensorShapeProtoDimension {
                            value: Some(onnx::tensor_shape_proto_dimension::Value::DimValue(n)),
                        },
                        TensorShapeProtoDimension {
                            value: Some(onnx::tensor_shape_proto_dimension::Value::DimValue(
                                in_dim,
                            )),
                        },
                    ],
                }),
            })),
        }),
        doc_string: String::new(),
        metadata_props: vec![],
    };
    let y_vi = ValueInfoProto {
        name: "Y".into(),
        r#type: Some(TypeProto {
            value: Some(type_proto::Value::TensorType(TypeProtoTensor {
                elem_type: onnx::tensor_proto::DataType::Float as i32,
                shape: Some(TensorShapeProto {
                    dim: vec![
                        TensorShapeProtoDimension {
                            value: Some(onnx::tensor_shape_proto_dimension::Value::DimValue(n)),
                        },
                        TensorShapeProtoDimension {
                            value: Some(onnx::tensor_shape_proto_dimension::Value::DimValue(
                                out_dim,
                            )),
                        },
                    ],
                }),
            })),
        }),
        doc_string: String::new(),
        metadata_props: vec![],
    };

    // Initializers W [in,out], B [out]
    let mut w_vals = Vec::with_capacity((in_dim * out_dim) as usize);
    for i in 0..(in_dim * out_dim) {
        w_vals.push((i as f32 * 0.01).sin());
    }
    let w = TensorProto {
        data_type: onnx::tensor_proto::DataType::Float as i32,
        name: "W".into(),
        dims: vec![in_dim, out_dim],
        float_data: w_vals.clone(),
        int32_data: vec![],
        int64_data: vec![],
        raw_data: vec![],
    };
    let mut b_vals = Vec::with_capacity(out_dim as usize);
    for i in 0..out_dim {
        b_vals.push((i as f32 * 0.1).cos());
    }
    let b = TensorProto {
        data_type: onnx::tensor_proto::DataType::Float as i32,
        name: "B".into(),
        dims: vec![out_dim],
        float_data: b_vals.clone(),
        int32_data: vec![],
        int64_data: vec![],
        raw_data: vec![],
    };

    // Nodes: Y0=MatMul(X,W); Y1=Add(Y0,B); Y=Softmax(Y1)
    let mm = NodeProto {
        input: vec!["X".into(), "W".into()],
        output: vec!["Y0".into()],
        name: String::new(),
        op_type: "MatMul".into(),
        domain: String::new(),
        attribute: vec![],
        doc_string: String::new(),
    };
    let add = NodeProto {
        input: vec!["Y0".into(), "B".into()],
        output: vec!["Y1".into()],
        name: String::new(),
        op_type: "Add".into(),
        domain: String::new(),
        attribute: vec![],
        doc_string: String::new(),
    };
    let sm = NodeProto {
        input: vec!["Y1".into()],
        output: vec!["Y".into()],
        name: String::new(),
        op_type: "Softmax".into(),
        domain: String::new(),
        attribute: vec![],
        doc_string: String::new(),
    };

    let graph = GraphProto {
        name: "linear_sm".into(),
        node: vec![mm, add, sm],
        initializer: vec![w, b],
        input: vec![x_vi],
        output: vec![y_vi],
        value_info: vec![],
    };
    ModelProto {
        ir_version: onnx::Version::IrVersion2019122 as i64,
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 13,
            extension: String::new(),
        }],
        graph: Some(graph),
    }
}

#[test]
fn test_import_and_run() {
    let n = 2;
    let in_dim = 4;
    let out_dim = 3;
    let model = build_linear_softmax_model(n, in_dim, out_dim);
    // Write to temp file
    let mut tmpf = tempfile::NamedTempFile::new().unwrap();
    use prost::Message as _;
    let mut buf = Vec::new();
    model.encode(&mut buf).unwrap();
    std::io::Write::write_all(&mut tmpf, &buf).unwrap();
    let path = tmpf.path().to_path_buf();

    // Import
    let mut res = import_onnx(&path).expect("import ok");

    // Build inputs
    let mut x_data = vec![0f32; (n * in_dim) as usize];
    for (i, v) in x_data.iter_mut().enumerate() {
        *v = ((i as f32) * 0.2).sin();
    }
    let _x = res
        .inputs
        .get("X")
        .copied()
        .expect("X exists")
        .set(x_data.clone());

    // Execute imported graph
    res.graph.execute();
    let y = res.outputs.get("Y").copied().expect("Y exists").data();

    // Reference using luminal directly
    let mut g2 = Graph::new();
    // The same initializers W,B must be reconstructed from the onnx model initializers used above
    let mut w_vals = Vec::with_capacity((in_dim * out_dim) as usize);
    for i in 0..(in_dim * out_dim) {
        w_vals.push((i as f32 * 0.01).sin());
    }
    let mut b_vals = Vec::with_capacity(out_dim as usize);
    for i in 0..out_dim {
        b_vals.push((i as f32 * 0.1).cos());
    }
    let x2 = g2.tensor((n as usize, in_dim as usize)).set(x_data);
    let w2 = g2.tensor((in_dim as usize, out_dim as usize)).set(w_vals);
    let b2 = g2.tensor((out_dim as usize,)).set(b_vals);
    let y2 = (x2.matmul(w2) + b2.expand_dim(0, n as usize))
        .softmax(1)
        .retrieve();
    g2.execute();
    let y_ref = y2.data();

    assert_eq!(y.len(), y_ref.len());
    for (a, b) in y.iter().zip(y_ref.iter()) {
        assert!((a - b).abs() < 1e-3, "{a} vs {b}");
    }
}
