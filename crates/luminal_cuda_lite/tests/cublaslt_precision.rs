//! CPU inspection of the graph -> elected cuBLASLt configuration -> API call.
//! No device timing or bitwise floating-point equivalence is assumed here.

use luminal::dtype::{DType, PlanDtype};
use luminal::egglog_utils::eclass::EGraphView;
use luminal::graph::Graph;
use luminal::layout_ir::{ExtractionSite, SerializedIndex};
use luminal::layouts::DecodedLayout;
use luminal::prelude::egraph_serialize::ClassId;
use luminal_cuda_lite::CudaRuntime;
use luminal_cuda_lite::ops::cublaslt::exec::{LtOrder, bind_destination, plan_call_from_spec};
use luminal_cuda_lite::ops::cublaslt::{
    CuEpilogue, CublasLtForm, LtComputeType, LtMatmulSpec, parse_spec,
};

fn logical_of(site: &ExtractionSite<'_>, lt: &ClassId) -> ClassId {
    let node = site
        .nodes_in_class_value(lt, "LayoutTensorLit")
        .next()
        .unwrap();
    site.class_of_child(node, 0).unwrap()
}

fn inspect(cx: &Graph) -> Vec<LtMatmulSpec> {
    let rt = CudaRuntime::load_with_registry(cx, luminal_cuda_lite::cuda_registry()).expect("load");
    let egraph = rt.saturated_egraph(&Default::default()).expect("saturate");
    let index = SerializedIndex::new(&egraph);
    let view = EGraphView::new(&egraph, rt.decoders());
    let mut specs = Vec::new();
    for (node_id, node) in &egraph.nodes {
        let Some(form) = CublasLtForm::ALL
            .into_iter()
            .find(|form| node.op == form.constructor_name())
        else {
            continue;
        };
        let site = ExtractionSite {
            egraph: &egraph,
            node_id,
            node,
            index: &index,
        };
        let spec = parse_spec(&site, form).expect("every candidate decodes");
        let config = spec.config;
        // Compare the API types with the actual dataflow payloads, including
        // C and bias. A successful descriptor decode alone cannot detect a
        // half pointer incorrectly described as F32 (or the converse).
        for (logical, dtype) in [
            (
                logical_of(&site, &spec.desc_a_layout_tensor),
                config.a_dtype,
            ),
            (
                logical_of(&site, &spec.desc_b_layout_tensor),
                config.b_dtype,
            ),
            (spec.logical_out.clone(), config.d_dtype),
        ] {
            assert_eq!(site.logical_dtype(&logical), Some(dtype), "{spec:?}");
        }
        for (lt, dtype) in [
            (&spec.c_tensor, config.c_dtype),
            (&spec.bias_tensor, config.bias_dtype),
        ] {
            if let Some(lt) = lt {
                assert_eq!(
                    site.logical_dtype(&logical_of(&site, lt)),
                    Some(dtype),
                    "{spec:?}"
                );
            }
        }
        let mut call = plan_call_from_spec(&spec).expect("plan call");
        let d_class = site.child_class(3);
        let desc_d = site
            .nodes_in_class_value(&d_class, "CublasLtOutputDDescriptor")
            .next()
            .unwrap();
        let output_lt = site.class_of_child(desc_d, 1).unwrap();
        let output = site
            .nodes_in_class_value(&output_lt, "LayoutTensorLit")
            .next()
            .unwrap();
        let layout_class = site.class_of_child(output, 1).unwrap();
        let layout =
            DecodedLayout::from_class(&view.class(&layout_class), Some(config.d_dtype)).unwrap();
        assert_eq!(layout.width_bits(), config.d_dtype.egglog_bits());
        bind_destination(&mut call, &layout, "precision inspection").unwrap();
        assert_eq!(call.d.order, LtOrder::Col);
        assert_eq!(call.c, call.d);
        assert_eq!(call.d.ld, call.m);
        // Geometry is the transposed recorder frame: op(A) is m*k,
        // op(B) is k*n, with C/D sharing the elected column-major layout.
        assert_eq!(
            if call.trans_a {
                (call.a.cols, call.a.rows)
            } else {
                (call.a.rows, call.a.cols)
            },
            (call.m, call.k)
        );
        assert_eq!(
            if call.trans_b {
                (call.b.cols, call.b.rows)
            } else {
                (call.b.rows, call.b.cols)
            },
            (call.k, call.n)
        );
        assert_eq!(call.config, config);
        assert_eq!(config.compute_type, LtComputeType::F32);
        assert_eq!(config.scale_dtype, PlanDtype::F32);
        assert_eq!(config.c_dtype, config.d_dtype);
        assert_eq!(call.beta_is_one, form.has_c());
        assert_eq!(call.bias_operand.is_some(), form.has_bias());
        assert_eq!(
            call.relu,
            matches!(spec.epilogue, CuEpilogue::Relu | CuEpilogue::ReluBias)
        );
        specs.push(spec);
    }
    specs
}

fn graph_with_epilogues(dtype: DType, m: usize, transposed_weight: bool) -> Graph {
    let mut cx = Graph::new();
    let a = cx.tensor(vec![m, 8], dtype).cast(DType::F32);
    let b = if transposed_weight {
        cx.tensor(vec![3, 8], dtype)
            .cast(DType::F32)
            .permute(vec![1, 0])
    } else {
        cx.tensor(vec![8, 3], dtype).cast(DType::F32)
    };
    let c = cx.tensor(vec![m, 3], dtype).cast(DType::F32);
    let bias = cx.tensor(vec![3], dtype).cast(DType::F32).expand_dim(0, m);
    let product = a.matmul(b);
    for value in [product, product + c, product + bias, (product + c) + bias] {
        let _out = value.cast(dtype);
        let _relu_out = value.relu().cast(dtype);
    }
    cx
}

#[test]
fn explicit_precision_and_epilogues_reach_all_five_api_configs() {
    for (dtype, storage) in [
        (DType::F32, PlanDtype::F32),
        (DType::F16, PlanDtype::F16),
        (DType::Bf16, PlanDtype::Bf16),
    ] {
        let specs = inspect(&graph_with_epilogues(dtype, 4, true));
        for output in [PlanDtype::F32, storage] {
            for form in CublasLtForm::ALL {
                for relu in [false, true] {
                    let wanted_epilogue = match (form.has_bias(), relu) {
                        (false, false) => CuEpilogue::Default,
                        (false, true) => CuEpilogue::Relu,
                        (true, false) => CuEpilogue::Bias,
                        (true, true) => CuEpilogue::ReluBias,
                    };
                    let spec = specs.iter().find(|spec| {
                        spec.config.a_dtype == storage && spec.config.b_dtype == storage
                            && spec.config.d_dtype == output && spec.form == form
                            && spec.epilogue == wanted_epilogue
                    }).unwrap_or_else(|| panic!("missing {storage:?} -> {output:?}, {form:?}, {wanted_epilogue:?}; found {:?}", specs.iter().map(|s| (s.config, s.form, s.epilogue)).collect::<Vec<_>>()));
                    assert_eq!(spec.mnk_lits(), (3, 4, 8));
                    println!(
                        "verified {storage:?} -> {output:?}: {form:?}, {wanted_epilogue:?}, F32 compute/scalars, COL destination"
                    );
                }
            }
        }
    }
}

#[test]
fn decode_shape_preserves_low_precision_storage() {
    let specs = inspect(&graph_with_epilogues(DType::Bf16, 1, false));
    assert!(
        specs
            .iter()
            .any(|spec| spec.config.a_dtype == PlanDtype::Bf16
                && spec.config.d_dtype == PlanDtype::Bf16
                && spec.form == CublasLtForm::Base)
    );
}

#[test]
fn cast_after_low_precision_arithmetic_is_not_f32_compute() {
    for dtype in [DType::F16, DType::Bf16] {
        let mut cx = Graph::new();
        let a = cx.tensor(vec![4, 8], dtype);
        let b = cx.tensor(vec![8, 3], dtype);
        let _out = a.matmul(b).cast(DType::F32);
        assert!(inspect(&cx).is_empty());
    }
}

#[test]
fn mixed_input_storage_is_not_an_enumerated_config() {
    let mut cx = Graph::new();
    let a = cx.tensor(vec![4, 8], DType::F16).cast(DType::F32);
    let b = cx.tensor(vec![8, 3], DType::Bf16).cast(DType::F32);
    let _out = a.matmul(b);
    // The ordinary F32 call is legal after materializing both casts. Reading
    // the F16/BF16 buffers together as a mixed cuBLASLt call is not.
    let specs = inspect(&cx);
    assert!(!specs.is_empty());
    assert!(
        specs
            .iter()
            .all(|spec| spec.config.a_dtype == PlanDtype::F32
                && spec.config.b_dtype == PlanDtype::F32)
    );
}

#[test]
fn epilogue_does_not_cross_an_intermediate_rounding_cast() {
    let mut cx = Graph::new();
    let a = cx.tensor(vec![4, 8], DType::Bf16).cast(DType::F32);
    let b = cx.tensor(vec![8, 3], DType::Bf16).cast(DType::F32);
    let c = cx.tensor(vec![4, 3], DType::F32);
    let rounded = a.matmul(b).cast(DType::Bf16).cast(DType::F32);
    let _out = (rounded + c).relu();
    let specs = inspect(&cx);
    assert!(
        specs
            .iter()
            .any(|spec| spec.config.d_dtype == PlanDtype::Bf16)
    );
    assert!(
        specs
            .iter()
            .all(|spec| spec.form == CublasLtForm::Base && spec.epilogue == CuEpilogue::Default)
    );
}

#[test]
fn narrow_output_does_not_reinterpret_f32_residual_or_bias() {
    let mut cx = Graph::new();
    let a = cx.tensor(vec![4, 8], DType::F16).cast(DType::F32);
    let b = cx.tensor(vec![8, 3], DType::F16).cast(DType::F32);
    let c = cx.tensor(vec![4, 3], DType::F32);
    let bias = cx.tensor(vec![3], DType::F32).expand_dim(0, 4usize);
    let product = a.matmul(b);
    let _out_c = (product + c).cast(DType::F16);
    let _out_bias = (product + bias).cast(DType::F16);
    let specs = inspect(&cx);
    for form in [CublasLtForm::Accumulate, CublasLtForm::Bias] {
        assert!(specs.iter().any(|spec| spec.form == form
            && spec.config.a_dtype == PlanDtype::F16
            && spec.config.d_dtype == PlanDtype::F32));
        assert!(
            !specs
                .iter()
                .any(|spec| spec.form == form && spec.config.d_dtype == PlanDtype::F16)
        );
    }
}
