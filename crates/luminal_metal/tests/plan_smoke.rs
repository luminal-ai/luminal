use luminal::bufferize::BufferNode;
use luminal::dtype::DType;
use luminal::prelude::FxHashMap;
use luminal_metal::{MetalRuntime, kernels};

#[test]
fn search_produces_a_codegen_complete_plan() {
    let mut cx = luminal::graph::Graph::new();
    let a = cx.tensor((2usize, 3usize), DType::F32);
    let b = cx.tensor((2usize, 3usize), DType::F32);
    let _out = ((a + b) * a).output();

    let mut rt = MetalRuntime::load(&cx).expect("load");
    let data: FxHashMap<_, _> = [
        (a.id, vec![1.0f32, 2., 3., 4., 5., 6.].into()),
        (b.id, vec![10.0f32, 20., 30., 40., 50., 60.].into()),
    ]
    .into_iter()
    .collect();
    let outcome = rt
        .search(&data, &luminal_metal::harness_search_options())
        .expect("search under the Metal allow list");
    assert!(outcome.plans_profiled > 0, "no plans profiled");

    let plan = rt.plan().expect("plan loaded");
    let mut computes = 0usize;
    for node in plan.dag.node_weights() {
        if let BufferNode::Compute { op, .. } = node {
            computes += 1;
            let label = op.label();
            if label == "BufferAlloc" || label == "BufferFree" {
                continue;
            }
            assert!(
                luminal_metal::as_kernel_op(op.as_ref()).is_some(),
                "elected op {label} has no kernel interface"
            );
        }
    }
    assert!(computes > 0, "plan has no compute nodes");

    rt.set_data(a.id, vec![1.0f32, 2., 3., 4., 5., 6.]);
    rt.set_data(b.id, vec![10.0f32, 20., 30., 40., 50., 60.]);
    #[cfg(not(target_os = "macos"))]
    {
        let err = rt
            .execute()
            .expect_err("execute must refuse without a device");
        assert!(
            err.to_string().contains("device"),
            "refusal must name the missing feature: {err}"
        );
    }
    #[cfg(target_os = "macos")]
    {
        rt.execute().expect("device execute");
        let got = rt.get_f32(_out.id).expect("output payload");
        assert_eq!(got, vec![11.0f32, 44., 99., 176., 275., 396.]);
    }
}

#[test]
fn codegen_emits_wellformed_sources() {
    use luminal::dtype::PlanDtype;
    use luminal::layouts::{
        BitWidthTerm, IntExprTerm, RightMajorContiguousElementLayout, ShapeTerm,
    };
    fn rm_layout(dims: &[i64]) -> luminal_metal::layouts::DecodedLayout {
        luminal_metal::layouts::DecodedLayout::of(
            RightMajorContiguousElementLayout {
                shape: ShapeTerm(dims.iter().map(|&d| IntExprTerm::Lit(d)).collect()),
                width: BitWidthTerm(32),
            },
            Some(PlanDtype::F32),
        )
    }
    let ctx = kernels::CodegenCtx {
        operand_dims: vec![
            vec![2usize.into(), 3usize.into()],
            vec![2usize.into(), 3usize.into()],
            vec![2usize.into(), 3usize.into()],
        ],
        operand_dtypes: vec![PlanDtype::F32, PlanDtype::F32, PlanDtype::F32],
        dest_dims: vec![vec![2usize.into(), 3usize.into()]],
        dest_dtypes: vec![PlanDtype::F32],
        operand_layouts: vec![rm_layout(&[2, 3]), rm_layout(&[2, 3]), rm_layout(&[2, 3])],
    };
    let add = luminal_metal::ops::add::AddFunctionalDps;
    let kernel = luminal_metal::as_kernel_op(&add).expect("add implements KernelOp");
    let launches = kernel.codegen(&ctx).expect("codegen");
    assert_eq!(launches.len(), 1);
    assert_eq!(launches[0].n.literal(), Some(6));
    assert!(launches[0].source.contains("kernel void k("));
    assert!(launches[0].source.contains("a[i] + b[i]"));
}
