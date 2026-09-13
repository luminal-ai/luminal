use super::*;
use crate::runtime::CudaRuntime;
use cudarc::driver::{CudaContext, DevicePtr, LaunchConfig, PushKernelArg};
use half::{bf16, f16};
use luminal::prelude::*;

fn round(x: f32, dtype: DType) -> f32 {
    match dtype {
        DType::Bf16 => bf16::from_f32(x).to_f32(),
        DType::F16 => f16::from_f32(x).to_f32(),
        _ => x,
    }
}

fn bytes(data: &[f32], dtype: DType) -> Vec<u8> {
    data.iter()
        .flat_map(|&x| match dtype {
            DType::Bf16 => bf16::from_f32(x).to_bits().to_le_bytes().to_vec(),
            DType::F16 => f16::from_f32(x).to_bits().to_le_bytes().to_vec(),
            _ => x.to_le_bytes().to_vec(),
        })
        .collect()
}

fn floats(data: &[u8], dtype: DType) -> Vec<f32> {
    data.chunks_exact(dtype.bits() / 8)
        .map(|v| match dtype {
            DType::Bf16 => bf16::from_bits(u16::from_le_bytes(v.try_into().unwrap())).to_f32(),
            DType::F16 => f16::from_bits(u16::from_le_bytes(v.try_into().unwrap())).to_f32(),
            _ => f32::from_le_bytes(v.try_into().unwrap()),
        })
        .collect()
}

// Read the real saturated graph and run every extracted normalization choice.
// This also verifies the public helper no longer hides its operation from search.
fn choices(rows: usize, cols: usize, dtype: DType, dynamic: bool) -> Vec<(usize, DType, LLIROp)> {
    let mut cx = Graph::default();
    let r = if dynamic {
        's'.into()
    } else {
        Expression::from(rows)
    };
    cx.set_dim('s', rows);
    let x = cx.tensor((r, cols));
    let w = cx.tensor(cols);
    fused_rms_norm(x.cast(dtype), w, 1e-12).output();
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let egraph = cx.egraph().unwrap();
    let mut choices = Vec::new();
    for (_, fields) in egraph
        .enodes
        .values()
        .filter(|(name, _)| name == "FusedRMSNorm")
    {
        let children = fields
            .iter()
            .map(|c| &egraph.eclasses[c].1[0])
            .collect::<Vec<_>>();
        let threads =
            luminal::egglog_utils::extract_expr(egraph, children[6], &mut FxHashMap::default())
                .unwrap()
                .to_usize()
                .unwrap();
        let input_dtype = luminal::egglog_utils::extract_dtype(egraph, children[5]);
        let (op, _) = RMSNormKernel::default().extract(
            egraph,
            &children,
            vec![],
            &mut FxHashMap::default(),
            &mut FxHashMap::default(),
        );
        choices.push((threads, input_dtype, op));
    }
    choices.sort_by_key(|(threads, input, _)| (*threads, input.bits()));
    let expected_count = if dtype == DType::F32 { 4 } else { 8 };
    assert_eq!(
        choices.len(),
        expected_count,
        "dtype={dtype:?}, dynamic={dynamic}, cols={cols}"
    );
    for threads in [128, 256, 512, 1024] {
        assert!(choices.iter().any(|(t, d, _)| *t == threads && *d == dtype));
        assert!(
            choices
                .iter()
                .any(|(t, d, _)| *t == threads && *d == DType::F32)
        );
    }
    choices
}

#[test]
fn searchable_rmsnorm_all_choices_preserve_cast_and_match_scalar_reference() {
    let stream = CudaContext::new(0).unwrap().default_stream();
    let mut cache = FxHashMap::default();
    let rows = 3;
    for dtype in [DType::F32, DType::F16, DType::Bf16] {
        for (cols, dynamic) in [(131, false), (136, true), (2880, false)] {
            let x: Vec<_> = (0..rows * cols)
                .map(|i| {
                    let scale = if i < cols { 1e-6 } else { 1.0 };
                    scale * (((i * 1543 % 2003) as f32) / 1001. - 1.)
                })
                .collect();
            let w: Vec<_> = (0..cols).map(|i| (i % 19) as f32 / 13. - 0.4).collect();
            let mut expected = Vec::new();
            for row in x.chunks_exact(cols) {
                let sum: f64 = row.iter().map(|&v| (round(v, dtype) as f64).powi(2)).sum();
                let inv = (sum / cols as f64 + 1e-12_f32 as f64).sqrt().recip();
                expected.extend(row.iter().zip(&w).map(|(&v, &w)| {
                    round((round(v, dtype) as f64 * inv * w as f64) as f32, dtype)
                }));
            }
            let xf = stream.clone_htod(&bytes(&x, DType::F32)).unwrap();
            let xn = stream.clone_htod(&bytes(&x, dtype)).unwrap();
            let wg = stream.clone_htod(&w).unwrap();
            let out = stream
                .alloc_zeros::<u8>(rows * cols * dtype.bits() / 8)
                .unwrap();
            let mut paired = std::collections::BTreeMap::<usize, Vec<Vec<u8>>>::new();
            for (threads, input_dtype, op) in choices(rows, cols, dtype, dynamic) {
                let kernel = op.to_dialect::<dyn KernelOp>().unwrap();
                let (f, _m, _, _, _, _, _) = kernel.compile(&stream, &mut cache);
                let xp = if input_dtype == DType::F32 {
                    xf.device_ptr(&stream).0
                } else {
                    xn.device_ptr(&stream).0
                };
                let wp = wg.device_ptr(&stream).0;
                let yp = out.device_ptr(&stream).0;
                unsafe {
                    stream
                        .launch_builder(&f)
                        .arg(&yp)
                        .arg(&xp)
                        .arg(&wp)
                        .launch(LaunchConfig {
                            grid_dim: (rows as u32, 1, 1),
                            block_dim: (threads as u32, 1, 1),
                            shared_mem_bytes: 0,
                        })
                        .unwrap();
                }
                let actual_bytes = stream.clone_dtoh(&out).unwrap();
                let actual = floats(&actual_bytes, dtype);
                let tolerance = match dtype {
                    DType::Bf16 => 0.008,
                    DType::F16 => 0.001,
                    _ => 2e-6,
                };
                for (i, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
                    assert!(
                        (a - e).abs() <= tolerance * e.abs().max(1e-6),
                        "dtype={dtype:?}, cols={cols}, threads={threads}, input={input_dtype:?}, i={i}: {a} != {e}"
                    );
                }
                paired.entry(threads).or_default().push(actual_bytes);
            }
            // Absorbing the input cast must retain every bit at a fixed reduction schedule.
            if dtype != DType::F32 {
                for outputs in paired.values() {
                    assert_eq!(outputs[0], outputs[1]);
                }
            }
        }
    }
}

#[test]
fn searchable_rmsnorm_materializes_transposed_input() {
    let stream = CudaContext::new(0).unwrap().default_stream();
    let mut cx = Graph::default();
    let x = cx.tensor((5, 3));
    let w = cx.tensor((5, 2));
    let weight_view = w.slice((.., 0..1)).squeeze(1);
    let y = fused_rms_norm(x.t(), weight_view, 0.0).output();
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let data: Vec<f32> = (0..15).map(|i| i as f32 / 7. - 1.).collect();
    let weights = vec![1f32, 9., 1., 8., 1., 7., 1., 6., 1., 5.];
    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(x, data.clone());
    rt.set_data(w, weights.clone());
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(8));
    rt.set_data(x, data.clone());
    rt.set_data(w, weights);
    rt.execute(&cx.dyn_map);
    let actual = rt.get_f32(y.id);
    for row in 0..3 {
        let sum: f64 = (0..5).map(|col| (data[col * 3 + row] as f64).powi(2)).sum();
        let inv = (sum / 5.).sqrt().recip();
        for col in 0..5 {
            assert!((actual[row * 5 + col] as f64 - data[col * 3 + row] as f64 * inv).abs() < 2e-6);
        }
    }
}

#[test]
#[ignore = "performance experiment; requires an otherwise idle GPU"]
fn benchmark_searchable_rmsnorm_cast_choices() {
    use crate::kernel::CudaGraphHandle;
    use cudarc::driver::sys::CUevent_flags;
    let stream = CudaContext::new(0).unwrap().new_stream().unwrap();
    let mut cache = FxHashMap::default();
    let cols = 2880;
    eprintln!("rmsnorm_bench,rows,cols,threads,fused_input_cast,trial,microseconds");
    for rows in [1, 2, 4, 8, 16, 32, 64] {
        let x = stream.clone_htod(&vec![0.1234567f32; rows * cols]).unwrap();
        let cast = stream.alloc_zeros::<u16>(rows * cols).unwrap();
        let w = stream.clone_htod(&vec![1f32; cols]).unwrap();
        let y = stream.alloc_zeros::<u16>(rows * cols).unwrap();
        let xp = x.device_ptr(&stream).0;
        let cp = cast.device_ptr(&stream).0;
        let wp = w.device_ptr(&stream).0;
        let yp = y.device_ptr(&stream).0;
        let cast_source = format!(
            r#"#include <cuda_bf16.h>
            extern "C" __global__ void cast_input(__nv_bfloat16 *y, const float *x) {{
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < {}) y[i] = __float2bfloat16_rn(x[i]);
            }}"#,
            rows * cols
        );
        let module = stream
            .context()
            .load_module(
                compile_module_image_for_current_device(stream.context(), &cast_source).unwrap(),
            )
            .unwrap();
        let cast_f = module.load_function("cast_input").unwrap();
        for (threads, input_dtype, op) in choices(rows, cols, DType::Bf16, false) {
            let kernel = op.to_dialect::<dyn KernelOp>().unwrap();
            let (f, _module, _, _, _, _, _) = kernel.compile(&stream, &mut cache);
            let fused = input_dtype == DType::F32;
            let norm_x = if fused { xp } else { cp };
            stream.synchronize().unwrap();
            CudaGraphHandle::begin_standalone_capture(&stream).unwrap();
            for _ in 0..64 {
                unsafe {
                    if !fused {
                        stream
                            .launch_builder(&cast_f)
                            .arg(&cp)
                            .arg(&xp)
                            .launch(LaunchConfig {
                                grid_dim: ((rows * cols).div_ceil(256) as u32, 1, 1),
                                block_dim: (256, 1, 1),
                                shared_mem_bytes: 0,
                            })
                            .unwrap();
                    }
                    stream
                        .launch_builder(&f)
                        .arg(&yp)
                        .arg(&norm_x)
                        .arg(&wp)
                        .launch(LaunchConfig {
                            grid_dim: (rows as u32, 1, 1),
                            block_dim: (threads as u32, 1, 1),
                            shared_mem_bytes: 0,
                        })
                        .unwrap();
                }
            }
            let graph = CudaGraphHandle::end_standalone_capture(&stream).unwrap();
            let exec = graph.instantiate().unwrap();
            for _ in 0..3 {
                exec.launch(&stream).unwrap();
            }
            for trial in 0..5 {
                let start = stream
                    .context()
                    .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                    .unwrap();
                let end = stream
                    .context()
                    .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                    .unwrap();
                start.record(&stream).unwrap();
                for _ in 0..10 {
                    exec.launch(&stream).unwrap();
                }
                end.record(&stream).unwrap();
                end.synchronize().unwrap();
                let us = start.elapsed_ms(&end).unwrap() * 1000. / 640.;
                eprintln!("rmsnorm_bench,{rows},{cols},{threads},{fused},{trial},{us:.6}");
            }
        }
    }
}

#[test]
fn searchable_rmsnorm_cast_fusion_survives_shared_residual() {
    let mut cx = Graph::default();
    cx.set_dim('s', 8);
    let x = cx.tensor(('s', 2880));
    let residual = cx.tensor(('s', 2880));
    let w = cx.tensor(2880);
    let sum = x + residual;
    sum.output();
    fused_rms_norm(sum.cast(DType::Bf16), w, 1e-5).output();
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let egraph = cx.egraph().unwrap();
    let fused = egraph
        .enodes
        .values()
        .filter(|(name, _)| name == "FusedRMSNorm")
        .filter(|(_, fields)| {
            egraph.eclasses[&fields[5]]
                .1
                .iter()
                .any(|id| egraph.enodes[id].0 == "F32")
        })
        .count();
    assert_eq!(
        fused, 4,
        "every block size should retain an absorbed-cast alternative after elementwise fusion"
    );
}
