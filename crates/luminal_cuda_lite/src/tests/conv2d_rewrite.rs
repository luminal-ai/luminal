use luminal::{egglog_utils::SerializedEGraph, prelude::*};

use crate::runtime::CudaRuntime;

use super::utilities::{
    ForcedExtractionConfig, assert_close,
    extract_forced_kernel_llir as extract_forced_kernel_llir_with_config, get_cuda_stream,
    llir_kernel_names, op_ir_nodes, try_extract_forced_nodes_llir_where,
};

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

fn conv2d_bias_same_shape_wrong_indices(
    x: GraphTensor,
    weight: GraphTensor,
    bias: GraphTensor,
    kernel_h: usize,
    kernel_w: usize,
) -> GraphTensor {
    let dims = x.dims();
    let h_out = dims[1] - kernel_h;
    let h_out = h_out + 1;
    let w_out = dims[2] - kernel_w;
    let w_out = w_out + 1;
    let index_shape = vec![
        dims[0],
        h_out,
        w_out,
        1.into(),
        kernel_h.into(),
        kernel_w.into(),
    ];

    // Same shape, range and contiguous index layout as unfold, but a cyclic
    // linear mapping instead of the sliding-window address expression.
    let input_elements = dims.iter().copied().product::<Expression>();
    let indexes = x
        .graph()
        .iota(Expression::from('z') % input_elements, index_shape);
    let gathered = x.gather(indexes);

    let mut patches = gathered.squeeze(3).permute(&[1, 2, 0, 3, 4]);
    while patches.dims().len() > 3 {
        let last = patches.dims().len();
        patches = patches.merge_dims(last - 2, last - 1);
    }
    let patches = patches.merge_dims(0, 1);

    let out = patches.matmul(weight.t());
    let out = out.split_dims(0, w_out).permute(&[2, 0, 1]);
    out + bias.expand_dim(1, h_out).expand_dim(2, w_out)
}

fn conv2d_bias_wrong_patch_axis_order(
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

    // Flatten K as [KH,Cin,KW], rather than the convolution contract's
    // [Cin,KH,KW]. All dimensions still multiply to the same M and K.
    let mut patches = unfolded.squeeze(3).permute(&[1, 2, 3, 0, 4]);
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
        op_kinds_share_class(egraph, "KernelConv2D", "FusionEnd"),
        "direct Conv2D and its lowered decomposition should coexist in one e-class"
    );
}

#[test]
fn generic_conv2d_decomposed_candidate_is_extractable() {
    let (mut cx, _, _, _, _) = build_conv_graph();
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let llir = extract_forced_decomposed_llir(&mut cx);

    assert!(
        !llir_kernel_names(&llir).contains(&"GenericConv2D"),
        "forcing the fallback output choice must extract a non-Conv2D plan"
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
fn generic_conv2d_rewrite_rejects_same_shape_non_unfold_gather() {
    let mut cx = Graph::new();
    let x = cx.tensor((2usize, 5usize, 6usize));
    let weight = cx.tensor((3usize, 2usize * 3 * 2));
    let bias = cx.tensor(3usize);
    conv2d_bias_same_shape_wrong_indices(x, weight, bias, 3, 2).output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let egraph = cx.egraph().expect("search space should have an e-graph");

    assert!(
        op_ir_nodes(egraph, "KernelConv2D").is_empty(),
        "same Gather shape with a non-unfold Iota mapping must not match KernelConv2D"
    );
}

#[test]
fn generic_conv2d_rewrite_rejects_wrong_patch_axis_order() {
    let mut cx = Graph::new();
    let x = cx.tensor((2usize, 5usize, 6usize));
    let weight = cx.tensor((3usize, 2usize * 3 * 2));
    let bias = cx.tensor(3usize);
    conv2d_bias_wrong_patch_axis_order(x, weight, bias, 3, 2).output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let egraph = cx.egraph().expect("search space should have an e-graph");

    assert!(
        op_ir_nodes(egraph, "KernelConv2D").is_empty(),
        "same M and K with a non-convolution patch-axis order must not match KernelConv2D"
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
fn generic_conv2d_decomposed_candidate_executes_unfold_matmul_bias() {
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let (mut cx, x, weight, bias, out) = build_conv_graph();
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let llir = extract_forced_decomposed_llir(&mut cx);

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
    extract_forced_kernel_llir_with_config(
        cx,
        "KernelConv2D",
        kernel_name,
        ForcedExtractionConfig::new(0xC0_2D00),
        false,
    )
}

fn extract_forced_decomposed_llir(cx: &mut Graph) -> LLIRGraph {
    let egraph = cx.egraph().expect("search space should have an e-graph");
    let conv_classes = op_ir_nodes(egraph, "KernelConv2D")
        .into_iter()
        .map(|node| egraph.node_to_class[node].clone())
        .collect::<FxHashSet<_>>();
    let fallback_nodes = op_ir_nodes(egraph, "FusionEnd")
        .into_iter()
        .filter(|node| conv_classes.contains(&egraph.node_to_class[*node]))
        .collect::<Vec<_>>();
    try_extract_forced_nodes_llir_where(
        cx,
        &fallback_nodes,
        ForcedExtractionConfig::new(0xDEC0_2D00)
            .attempts_per_node(32)
            .node_seed_stride(32),
        |llir| !llir_kernel_names(llir).contains(&"GenericConv2D"),
    )
    .unwrap_or_else(|error| panic!("could not extract the decomposed Conv2D fallback: {error}"))
}

fn op_kinds_share_class(egraph: &SerializedEGraph, a: &str, b: &str) -> bool {
    let a_classes = op_ir_nodes(egraph, a)
        .into_iter()
        .map(|node| &egraph.node_to_class[node])
        .collect::<FxHashSet<_>>();
    op_ir_nodes(egraph, b)
        .into_iter()
        .any(|node| a_classes.contains(&egraph.node_to_class[node]))
}
