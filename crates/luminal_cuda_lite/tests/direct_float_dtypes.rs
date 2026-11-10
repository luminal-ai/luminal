//! Every CUDA-lite physical op directly selected from a same-dtype floating
//! logical op must remain executable in the four public float formats.
#![cfg(feature = "device")]

mod support;

use half::{bf16, f16};
use luminal::dtype::PlanDtype;
use luminal::prelude::*;
use luminal_cuda_lite::{CudaRuntime, HostBuffer};
use support::TestTransfers;

fn plan_dtype(dtype: DType) -> PlanDtype {
    match dtype {
        DType::F32 => PlanDtype::F32,
        DType::F64 => PlanDtype::F64,
        DType::F16 => PlanDtype::F16,
        DType::Bf16 => PlanDtype::Bf16,
        other => panic!("not one of the four floating test dtypes: {other:?}"),
    }
}

fn payload(dtype: DType, values: &[f32]) -> HostBuffer {
    let bytes = match dtype {
        DType::F32 => values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect(),
        DType::F64 => values
            .iter()
            .flat_map(|value| f64::from(*value).to_ne_bytes())
            .collect(),
        DType::F16 => values
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_bits().to_ne_bytes())
            .collect(),
        DType::Bf16 => values
            .iter()
            .flat_map(|value| bf16::from_f32(*value).to_bits().to_ne_bytes())
            .collect(),
        other => panic!("not one of the four floating test dtypes: {other:?}"),
    };
    HostBuffer::new(plan_dtype(dtype), bytes).unwrap()
}

fn decoded_f64(buffer: &HostBuffer) -> Vec<f64> {
    match buffer.dtype {
        PlanDtype::F32 => buffer
            .bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()) as f64)
            .collect(),
        PlanDtype::F64 => buffer.as_f64().unwrap(),
        PlanDtype::F16 => buffer
            .bytes
            .chunks_exact(2)
            .map(|bytes| f16::from_bits(u16::from_ne_bytes(bytes.try_into().unwrap())).to_f64())
            .collect(),
        PlanDtype::Bf16 => buffer
            .bytes
            .chunks_exact(2)
            .map(|bytes| bf16::from_bits(u16::from_ne_bytes(bytes.try_into().unwrap())).to_f64())
            .collect(),
        other => panic!("unexpected output dtype {other:?}"),
    }
}

#[test]
fn direct_float_matches_execute_for_f32_f64_f16_and_bf16() {
    for dtype in [DType::F32, DType::F64, DType::F16, DType::Bf16] {
        let mut graph = Graph::new();
        let x = graph.tensor([4], dtype);
        let y = graph.tensor([4], dtype);

        // These two logical constructors normally have equivalent frontend
        // spellings (`exp2(x * log2(e))`, `x * recip(y)`). Keep raw leaves so
        // the direct backend match itself must also compile on the device.
        let dims = x.dims();
        let _direct_exp =
            graph
                .logical
                .op(LogicalOp::Exp, &[(x.id, dims.clone())], dims.clone(), dtype);
        let _direct_div = graph.logical.op(
            LogicalOp::Div,
            &[(x.id, dims.clone()), (y.id, dims.clone())],
            dims.clone(),
            dtype,
        );
        let _direct_trunc_cast = graph.logical.op(
            LogicalOp::TruncCast(DType::I64),
            &[(x.id, dims.clone())],
            dims,
            DType::I64,
        );

        // Direct elementwise logical matches, including every libdevice-style
        // math call and the ordinary typed CUDA operators.
        let math = x.sqrt()
            + x.exp()
            + x.exp2()
            + x.log2()
            + x.sin()
            + x.tanh()
            + x.floor()
            + x.ceil()
            + x.trunc()
            + x.round()
            + (x % y)
            + x.reciprocal();
        let arithmetic = (((x + y) * y) / y) + x.lt(y).select(x, y);

        // Direct reduction and sequential-scan matches use the same typed
        // scalar representation, but different kernel templates.
        let reductions = x.sum(&[0]) + x.max(&[0]);
        let scans = x.cumsum(&[0]).sum(&[0]) + x.cumprod(&[0]).sum(&[0]) + x.cummax(&[0]).sum(&[0]);
        let output = math.sum(&[0]) + arithmetic.sum(&[0]) + reductions + scans;

        let inputs: FxHashMap<_, _> = [
            (x.id, payload(dtype, &[0.25, 0.5, 1.0, 2.0])),
            (y.id, payload(dtype, &[3.0, 3.0, 3.0, 3.0])),
        ]
        .into_iter()
        .collect();
        let mut runtime = CudaRuntime::load(&graph).unwrap();
        runtime
            .search(
                &Default::default(),
                &Default::default(),
                &inputs,
                &luminal_cuda_lite::harness_search_options(),
            )
            .unwrap_or_else(|error| panic!("{dtype:?} search failed: {error:#}"));

        let labels: std::collections::BTreeSet<_> = runtime
            .plan()
            .unwrap()
            .dag
            .node_weights()
            .filter_map(|node| match node {
                luminal::bufferize::BufferNode::Compute { op, .. } => Some(op.label()),
                _ => None,
            })
            .collect();
        for required in [
            "SqrtFunctionalGeneric",
            "ExpFunctionalGeneric",
            "Exp2FunctionalGeneric",
            "Log2FunctionalGeneric",
            "SinFunctionalGeneric",
            "TanhFunctionalGeneric",
            "FloorFunctionalGeneric",
            "CeilFunctionalGeneric",
            "TruncFunctionalGeneric",
            "TruncCastGeneric",
            "RoundFunctionalGeneric",
            "ModFunctionalGeneric",
            "RecipFunctionalGeneric",
            "AddFunctionalGeneric",
            "MulFunctionalGeneric",
            "DivFunctionalGeneric",
            "LessThanGeneric",
            "SelectFunctionalGeneric",
            "ReduceSumGeneric",
            "ReduceMaxGeneric",
            "LeftSequentialScanSum",
            "LeftSequentialScanProd",
            "LeftSequentialScanMax",
        ] {
            assert!(
                labels.contains(required),
                "{dtype:?} plan did not select {required}; labels: {labels:?}"
            );
        }

        let mut arena = support::allocate_arena(&runtime).unwrap();
        let mut staging = runtime.allocate_staging().unwrap();
        runtime
            .upload(&mut arena, x.id, inputs[&x.id].clone())
            .unwrap();
        runtime
            .upload(&mut arena, y.id, inputs[&y.id].clone())
            .unwrap();
        runtime.execute(arena.arena(), &mut staging).unwrap();
        let result = runtime.download(&arena, output.id).unwrap().0;
        assert_eq!(result.dtype, plan_dtype(dtype));
        let values = decoded_f64(&result);
        assert_eq!(values.len(), 1, "{dtype:?} output is not scalar");
        assert!(values[0].is_finite(), "{dtype:?} output is {values:?}");
    }
}
