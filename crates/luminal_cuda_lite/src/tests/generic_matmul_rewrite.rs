use half::bf16;
use luminal::prelude::*;

use crate::runtime::CudaRuntime;

use super::utilities::{
    ForcedExtractionConfig, assert_close,
    extract_forced_kernel_llir as extract_forced_kernel_llir_with_config, get_cuda_stream,
    gpu_supports_dtype, llir_kernel_names,
};

#[test]
fn generic_matmul_covers_noncontiguous_merged_head_projection() {
    let mut cx = Graph::default();
    let heads = 3;
    let seq = 4;
    let head_dim = 5;
    let hidden = heads * head_dim;
    let out_dim = 7;

    let attn = cx.tensor((heads, seq, head_dim));
    let weight = cx.tensor((out_dim, hidden));
    let merged = attn.transpose(0, 1).merge_dims(1, 2);
    merged.matmul(weight.t()).output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let llir = extract_forced_kernel_llir(&mut cx, "GenericMatmul");
    let names = llir_kernel_names(&llir);

    assert!(
        names.contains(&"GenericMatmul"),
        "expected generic matmul fallback, kernels: {names:?}"
    );
    assert!(
        !names.contains(&"Mul") && !names.contains(&"SumReduce"),
        "a forced GenericMatmul extraction should select that implementation atomically, kernels: {names:?}"
    );
}

#[test]
fn generic_matmul_executes_noncontiguous_merged_head_projection() {
    let mut cx = Graph::default();
    let heads = 3;
    let seq = 4;
    let head_dim = 5;
    let hidden = heads * head_dim;
    let out_dim = 7;

    let attn = cx.tensor((heads, seq, head_dim));
    let weight = cx.tensor((out_dim, hidden));
    let merged = attn.transpose(0, 1).merge_dims(1, 2);
    let output = merged.matmul(weight.t()).output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let llir = extract_forced_kernel_llir(&mut cx, "GenericMatmul");
    let stream = get_cuda_stream().expect("CUDA device required for GenericMatmul execution test");
    let mut rt = CudaRuntime::initialize(stream);

    let attn_data = seeded_data(heads * seq * head_dim, 0.19, -0.09);
    let weight_data = seeded_data(out_dim * hidden, 0.14, -0.06);
    rt.set_data(attn, attn_data.as_slice());
    rt.set_data(weight, weight_data.as_slice());

    rt.load_llir(&llir);
    assert!(
        rt.kernel_names().contains(&"GenericMatmul"),
        "expected the forced GenericMatmul plan, kernels: {:?}",
        rt.kernel_names()
    );

    rt.execute(&cx.dyn_map);
    let result = rt.get_f32(output.id);

    let mut expected = vec![0.0; seq * out_dim];
    for token in 0..seq {
        for out_col in 0..out_dim {
            let mut sum = 0.0;
            for inner in 0..hidden {
                let head = inner / head_dim;
                let dim = inner % head_dim;
                let attn_idx = head * seq * head_dim + token * head_dim + dim;
                sum += attn_data[attn_idx] * weight_data[out_col * hidden + inner];
            }
            expected[token * out_dim + out_col] = sum;
        }
    }

    assert_close(&result, &expected, 1e-5, 1e-5);
}

#[test]
fn generic_matmul_fp8_fallback_accumulates_and_outputs_f32() {
    const K: usize = 17;
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    if !gpu_supports_dtype(DType::F8E4M3) {
        return;
    }

    let mut cx = Graph::default();
    let a = cx.tensor((1, K)).as_dtype(DType::F8E4M3);
    let b_storage = cx.tensor((1, K)).as_dtype(DType::F8E4M3);
    // Keep the spelling used by existing FP8 model code. The trailing cast is
    // now a no-op because GraphTensor::matmul itself returns F32.
    let output = a.matmul(b_storage.t()).cast(DType::F32).output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let llir = extract_forced_kernel_llir(&mut cx, "GenericMatmul");
    assert!(
        llir_kernel_names(&llir).contains(&"GenericMatmul"),
        "the materialized F32-cast fallback must remain extractable"
    );

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    // 1.0 is exactly 0x38 in E4M3. The dot is 17.0, which is not E4M3
    // representable at this magnitude; an F8 output would round it.
    rt.set_data(a, vec![0x38u8; K]);
    rt.set_data(b_storage, vec![0x38u8; K]);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(output.id), &[17.0], 0.0, 0.0);
}

#[test]
fn kernel_gemv_f8_absorbs_promoted_casts_and_reads_raw_fp8() {
    const K: usize = 32;
    const N: usize = 8;
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    if !gpu_supports_dtype(DType::F8E4M3) {
        return;
    }

    let mut cx = Graph::default();
    let x = cx.tensor((1, K)).as_dtype(DType::Bf16);
    let weight = cx.tensor((N, K)).as_dtype(DType::F8E4M3);
    let input_scale = cx.tensor(());
    let weight_scale = cx.tensor(());
    let x_f32 = x.cast(DType::F32);
    let quantized = (x_f32 / input_scale.expand_rhs(x_f32.dims())).cast(DType::F8E4M3);
    let matmul = quantized.matmul(weight.t());
    let output = (matmul * (input_scale * weight_scale).expand_rhs(matmul.dims()))
        .cast(DType::Bf16)
        .cast(DType::F32)
        .output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let llir = extract_forced_kernel_llir(&mut cx, "KernelGemvF8");
    assert!(
        llir_kernel_names(&llir).contains(&"GemvF8"),
        "the FP8 GEMV backend must remain deterministically extractable"
    );

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(x, vec![bf16::from_f32(0.5); K]);
    rt.set_data(weight, vec![0x38u8; N * K]);
    rt.set_data(input_scale, vec![0.5f32]);
    rt.set_data(weight_scale, vec![0.25f32]);
    rt.execute(&cx.dyn_map);

    // q=1, w=1, dot=32, and the dequant scale is 0.5*0.25.
    assert_close(&rt.get_f32(output.id), &[4.0; N], 0.0, 0.0);
}

fn seeded_data(len: usize, scale: f32, bias: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = ((i * 37 + 11) % 97) as f32 / 97.0;
            x * scale + bias
        })
        .collect()
}

fn extract_forced_kernel_llir(cx: &mut Graph, egglog_kind: &str) -> LLIRGraph {
    let runtime_kernel_name = match egglog_kind {
        "KernelGemvF8" => "GemvF8",
        other => other,
    };
    extract_forced_kernel_llir_with_config(
        cx,
        egglog_kind,
        runtime_kernel_name,
        ForcedExtractionConfig::new(0x9EEE_0000)
            .attempts_per_node(128)
            .node_seed_stride(1 << 16),
        false,
    )
}
