use luminal::{
    egglog_utils::{
        NodeId, SerializedEGraph, egglog_to_llir, random_initial_choice, validate_choice_set,
    },
    prelude::*,
};
use rand::{SeedableRng, rngs::StdRng};

use crate::{kernel::KernelOp, runtime::CudaRuntime};

use super::utilities::{assert_close, get_cuda_stream};

fn conv2d_bias_hlir(
    x: GraphTensor,
    weight: GraphTensor,
    bias: GraphTensor,
    kernel_h: usize,
    kernel_w: usize,
) -> GraphTensor {
    let unfolded = x.unfold(
        vec![1usize, kernel_h, kernel_w],
        vec![1usize, 1, 1],
        vec![1usize, 1, 1],
    );
    let output_spatial_dims = unfolded.dims()[1..3].to_vec();

    let mut patches = unfolded.squeeze(3).permute(&[1, 2, 0, 3, 4]);
    while patches.dims().len() > 3 {
        let last = patches.dims().len();
        patches = patches.merge_dims(last - 2, last - 1);
    }
    let patches = patches.merge_dims(0, 1);

    let out = patches.matmul(weight.t());
    let out = out
        .split_dims(0, output_spatial_dims[1])
        .permute(&[2, 0, 1]);
    let out_dims = out.dims();
    out + bias.expand_dim(1, out_dims[1]).expand_dim(2, out_dims[2])
}

fn build_conv_graph() -> (Graph, GraphTensor, GraphTensor, GraphTensor, GraphTensor) {
    let mut cx = Graph::new();
    let x = cx.tensor((2usize, 5usize, 6usize));
    let weight = cx.tensor((3usize, 2usize * 3 * 2));
    let bias = cx.tensor(3usize);
    let out = conv2d_bias_hlir(x, weight, bias, 3, 2).output();
    (cx, x, weight, bias, out)
}

fn conv2d_bias_padded_hlir(
    x: GraphTensor,
    weight: GraphTensor,
    bias: GraphTensor,
    kernel: usize,
    padding: usize,
) -> GraphTensor {
    let zero = Expression::from(0);
    let pad = Expression::from(padding);
    let padded = x.pad(vec![(zero, zero), (pad, pad), (pad, pad)], 0.0);
    conv2d_bias_hlir(padded, weight, bias, kernel, kernel)
}

fn build_padded_conv_graph() -> (Graph, GraphTensor, GraphTensor, GraphTensor, GraphTensor) {
    let mut cx = Graph::new();
    let x = cx.tensor((2usize, 4usize, 5usize));
    let weight = cx.tensor((3usize, 2usize * 3 * 3));
    let bias = cx.tensor(3usize);
    let out = conv2d_bias_padded_hlir(x, weight, bias, 3, 1).output();
    (cx, x, weight, bias, out)
}

fn nearest_upsample_2x_hlir(x: GraphTensor) -> GraphTensor {
    let stage1 = x.expand_dim(2, 2usize).merge_dims(1, 2);
    stage1.expand_dim(3, 2usize).merge_dims(2, 3)
}

fn build_upsample_conv_graph() -> (Graph, GraphTensor, GraphTensor, GraphTensor, GraphTensor) {
    let mut cx = Graph::new();
    let x = cx.tensor((2usize, 3usize, 4usize));
    let weight = cx.tensor((3usize, 2usize * 3 * 3));
    let bias = cx.tensor(3usize);
    let up = nearest_upsample_2x_hlir(x);
    let out = conv2d_bias_padded_hlir(up, weight, bias, 3, 1).output();
    (cx, x, weight, bias, out)
}

fn conv1x1_bias_hlir(x: GraphTensor, weight: GraphTensor, bias: GraphTensor) -> GraphTensor {
    let dims = x.dims();
    let h = dims[1];
    let w = dims[2];
    let xt = x.permute(&[1, 2, 0]).merge_dims(0, 1);
    let out = xt.matmul(weight.t());
    let out = out.split_dims(0, w).permute(&[2, 0, 1]);
    out + bias.expand_dim(1, h).expand_dim(2, w)
}

fn build_conv1x1_graph() -> (Graph, GraphTensor, GraphTensor, GraphTensor, GraphTensor) {
    let mut cx = Graph::new();
    let x = cx.tensor((2usize, 4usize, 5usize));
    let weight = cx.tensor((3usize, 2usize));
    let bias = cx.tensor(3usize);
    let out = conv1x1_bias_hlir(x, weight, bias).output();
    (cx, x, weight, bias, out)
}

fn conv2d_matmul_without_conv_output_shape(
    x: GraphTensor,
    weight: GraphTensor,
    bias: GraphTensor,
    kernel_h: usize,
    kernel_w: usize,
) -> GraphTensor {
    let unfolded = x.unfold(
        vec![1usize, kernel_h, kernel_w],
        vec![1usize, 1, 1],
        vec![1usize, 1, 1],
    );

    let mut patches = unfolded.squeeze(3).permute(&[1, 2, 0, 3, 4]);
    while patches.dims().len() > 3 {
        let last = patches.dims().len();
        patches = patches.merge_dims(last - 2, last - 1);
    }
    let patches = patches.merge_dims(0, 1);

    let out = patches.matmul(weight.t());
    let out_dims = out.dims();
    out + bias.expand_dim(0, out_dims[0])
}

#[test]
fn generic_conv2d_rewrite_matches_unfold_matmul_bias() {
    let (mut cx, _, _, _, _) = build_conv_graph();
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let egraph = cx.egraph().expect("search space should have an e-graph");

    assert!(
        !op_ir_nodes(egraph, "KernelConv2D").is_empty(),
        "expected generic conv2d rewrite candidate"
    );
    assert!(
        op_ir_nodes(egraph, "Add").is_empty(),
        "generic conv2d cleanup should prune the final bias Add fallback"
    );
}

#[test]
fn generic_conv2d_rewrite_matches_conv1x1_matmul_bias() {
    let (mut cx, _, _, _, _) = build_conv1x1_graph();
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let egraph = cx.egraph().expect("search space should have an e-graph");

    assert!(
        !op_ir_nodes(egraph, "KernelConv2D").is_empty(),
        "expected generic conv2d rewrite candidate for 1x1 conv"
    );
}

#[test]
fn generic_conv2d_rewrite_requires_conv_output_shape() {
    let mut cx = Graph::new();
    let x = cx.tensor((2usize, 5usize, 6usize));
    let weight = cx.tensor((3usize, 2usize * 3 * 2));
    let bias = cx.tensor(3usize);
    conv2d_matmul_without_conv_output_shape(x, weight, bias, 3, 2).output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let egraph = cx.egraph().expect("search space should have an e-graph");

    assert!(
        op_ir_nodes(egraph, "KernelConv2D").is_empty(),
        "matmul+bias without [C_out,H_out,W_out] conv output shape should not match KernelConv2D"
    );
}

#[test]
fn generic_conv2d_candidate_executes_unfold_matmul_bias() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let (mut cx, x, weight, bias, out) = build_conv_graph();
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let llir = extract_forced_kernel_llir(&mut cx, "GenericConv2D");

    let input: Vec<f32> = (0..2 * 5 * 6).map(|i| i as f32 * 0.03 - 0.4).collect();
    let weights: Vec<f32> = (0..3 * 2 * 3 * 2)
        .map(|i| (i as f32 % 11.0) * 0.04 - 0.2)
        .collect();
    let biases = vec![0.25_f32, -0.15, 0.05];
    let expected = reference_conv2d(
        &input,
        &weights,
        &biases,
        ConvCase {
            c_in: 2,
            h: 5,
            w: 6,
            c_out: 3,
            kh: 3,
            kw: 2,
            padding_h: 0,
            padding_w: 0,
        },
    );

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(x, input);
    rt.set_data(weight, weights);
    rt.set_data(bias, biases);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
}

#[test]
fn generic_conv2d_candidate_executes_conv1x1_matmul_bias() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let (mut cx, x, weight, bias, out) = build_conv1x1_graph();
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let llir = extract_forced_kernel_llir(&mut cx, "GenericConv2D");

    let input: Vec<f32> = (0..2 * 4 * 5).map(|i| i as f32 * 0.07 - 1.0).collect();
    let weights: Vec<f32> = (0..3 * 2).map(|i| (i as f32 % 5.0) * 0.11 - 0.2).collect();
    let biases = vec![0.2_f32, -0.1, 0.4];
    let expected = reference_conv2d(
        &input,
        &weights,
        &biases,
        ConvCase {
            c_in: 2,
            h: 4,
            w: 5,
            c_out: 3,
            kh: 1,
            kw: 1,
            padding_h: 0,
            padding_w: 0,
        },
    );

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(x, input);
    rt.set_data(weight, weights);
    rt.set_data(bias, biases);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
}

#[test]
fn generic_conv2d_candidate_executes_padded_unfold_matmul_bias() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let (mut cx, x, weight, bias, out) = build_padded_conv_graph();
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let llir = extract_forced_kernel_llir(&mut cx, "GenericConv2D");

    let input: Vec<f32> = (0..2 * 4 * 5).map(|i| i as f32 * 0.05 - 0.5).collect();
    let weights: Vec<f32> = (0..3 * 2 * 3 * 3)
        .map(|i| (i as f32 % 13.0) * 0.03 - 0.17)
        .collect();
    let biases = vec![0.15_f32, -0.25, 0.35];
    let expected = reference_conv2d(
        &input,
        &weights,
        &biases,
        ConvCase {
            c_in: 2,
            h: 4,
            w: 5,
            c_out: 3,
            kh: 3,
            kw: 3,
            padding_h: 1,
            padding_w: 1,
        },
    );

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(x, input);
    rt.set_data(weight, weights);
    rt.set_data(bias, biases);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
}

#[test]
fn generic_conv2d_candidate_executes_upsample_view_input() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let (mut cx, x, weight, bias, out) = build_upsample_conv_graph();
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let llir = extract_forced_kernel_llir(&mut cx, "GenericConv2D");

    let input: Vec<f32> = (0..2 * 3 * 4).map(|i| i as f32 * 0.09 - 0.8).collect();
    let weights: Vec<f32> = (0..3 * 2 * 3 * 3)
        .map(|i| (i as f32 % 17.0) * 0.025 - 0.2)
        .collect();
    let biases = vec![0.05_f32, -0.1, 0.2];
    let upsampled = reference_nearest_upsample_2x(&input, 2, 3, 4);
    let expected = reference_conv2d(
        &upsampled,
        &weights,
        &biases,
        ConvCase {
            c_in: 2,
            h: 6,
            w: 8,
            c_out: 3,
            kh: 3,
            kw: 3,
            padding_h: 1,
            padding_w: 1,
        },
    );

    let mut rt = CudaRuntime::initialize(stream);
    rt.load_llir(&llir);
    rt.set_data(x, input);
    rt.set_data(weight, weights);
    rt.set_data(bias, biases);
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(out.id), &expected, 1e-5, 1e-5);
}

struct ConvCase {
    c_in: usize,
    h: usize,
    w: usize,
    c_out: usize,
    kh: usize,
    kw: usize,
    padding_h: usize,
    padding_w: usize,
}

fn reference_nearest_upsample_2x(input: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; c * h * 2 * w * 2];
    for ci in 0..c {
        for y in 0..h {
            for x in 0..w {
                let value = input[ci * h * w + y * w + x];
                for dy in 0..2 {
                    for dx in 0..2 {
                        let oy = y * 2 + dy;
                        let ox = x * 2 + dx;
                        out[ci * h * 2 * w * 2 + oy * w * 2 + ox] = value;
                    }
                }
            }
        }
    }
    out
}

fn reference_conv2d(input: &[f32], weight: &[f32], bias: &[f32], case: ConvCase) -> Vec<f32> {
    let ConvCase {
        c_in,
        h,
        w,
        c_out,
        kh,
        kw,
        padding_h,
        padding_w,
    } = case;
    let h_out = h + 2 * padding_h - kh + 1;
    let w_out = w + 2 * padding_w - kw + 1;
    let mut out = vec![0.0; c_out * h_out * w_out];
    for co in 0..c_out {
        for oh in 0..h_out {
            for ow in 0..w_out {
                let mut acc = bias[co];
                for ci in 0..c_in {
                    for r in 0..kh {
                        for s in 0..kw {
                            let Some(ih) = (oh + r).checked_sub(padding_h) else {
                                continue;
                            };
                            let Some(iw) = (ow + s).checked_sub(padding_w) else {
                                continue;
                            };
                            if ih >= h || iw >= w {
                                continue;
                            }
                            let input_idx = ci * h * w + ih * w + iw;
                            let weight_idx = co * c_in * kh * kw + (ci * kh + r) * kw + s;
                            acc += input[input_idx] * weight[weight_idx];
                        }
                    }
                }
                out[co * h_out * w_out + oh * w_out + ow] = acc;
            }
        }
    }
    out
}

fn extract_forced_kernel_llir(cx: &mut Graph, kernel_name: &str) -> LLIRGraph {
    let egraph = cx.egraph().expect("search space should have an e-graph");
    let ops = cx
        .egglog_ops()
        .expect("search space should have registered egglog ops");
    let kernel_nodes = op_ir_nodes(egraph, "KernelConv2D");
    assert!(
        !kernel_nodes.is_empty(),
        "expected at least one {kernel_name} candidate"
    );

    for (idx, kernel_node) in kernel_nodes.iter().enumerate() {
        let mut rng = StdRng::seed_from_u64(0xC0_2D00 + idx as u64);
        let mut choices = random_initial_choice(egraph, &mut rng);
        let kernel_class = &egraph.node_to_class[*kernel_node];
        choices.insert(kernel_class, kernel_node);

        if validate_choice_set(egraph, &choices, ops).is_err() {
            continue;
        }

        let mut list_cache = FxHashMap::default();
        let mut expr_cache = FxHashMap::default();
        let llir = egglog_to_llir(
            egraph,
            choices,
            ops,
            &cx.custom_ops,
            &mut list_cache,
            &mut expr_cache,
            None,
        );
        if llir_kernel_names(&llir).contains(&kernel_name) {
            return llir;
        }
    }

    panic!("could not extract a valid {kernel_name} candidate");
}

fn llir_kernel_names(llir: &LLIRGraph) -> Vec<&'static str> {
    llir.node_indices()
        .filter_map(|node| {
            llir[node]
                .to_dialect::<dyn KernelOp>()
                .map(|kernel| kernel.kernel_name())
        })
        .collect()
}

fn op_ir_nodes<'a>(egraph: &'a SerializedEGraph, kind_label: &str) -> Vec<&'a NodeId> {
    let op_kind_classes = egraph
        .enodes
        .iter()
        .filter(|(_, (label, _))| label == kind_label)
        .map(|(node, _)| egraph.node_to_class[node].clone())
        .collect::<Vec<_>>();

    egraph
        .enodes
        .iter()
        .filter_map(|(node, (label, children))| {
            (label == "Op"
                && children
                    .first()
                    .is_some_and(|kind| op_kind_classes.contains(kind)))
            .then_some(node)
        })
        .collect()
}
