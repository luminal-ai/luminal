use luminal::graph::Graph as LuminalGraph;
use luminal::prelude::*;
use pyo3::prelude::*;
use std::collections::HashMap;

#[cfg(feature = "cuda")]
use luminal_cuda_lite::cudarc::driver::CudaContext;
#[cfg(feature = "cuda")]
use luminal_cuda_lite::runtime::CudaRuntime;

use crate::compiled_graph::CompiledGraph;
use crate::pt2_parser;
use crate::pt2_schema;
use crate::runtime::RuntimeBackend;
use crate::translator;
use crate::util::DimParamMap;

fn resolve_dim_sizes(
    sizes: &[pt2_schema::DimSize],
    sym_to_char: &HashMap<String, char>,
) -> Vec<Expression> {
    sizes
        .iter()
        .map(|s| match s {
            pt2_schema::DimSize::Int(i) => Expression::from(i.as_int as usize),
            pt2_schema::DimSize::Expr(e) => {
                if let Some(sym) = pt2_parser::extract_symbol_name_pub(&e.as_expr.expr_str) {
                    if let Some(c) = sym_to_char.get(&sym) {
                        Expression::from(*c)
                    } else {
                        Expression::from(1usize)
                    }
                } else {
                    Expression::from(1usize)
                }
            }
        })
        .collect()
}

#[pyfunction]
pub fn compile_pt2(
    pt2_path: &str,
    weights_path: &str,
    backend: &str,
    search_iters: usize,
) -> PyResult<CompiledGraph> {
    compile_pt2_inner(pt2_path, weights_path, backend, search_iters)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))
}

fn compile_pt2_inner(
    pt2_path: &str,
    weights_path: &str,
    backend: &str,
    search_iters: usize,
) -> anyhow::Result<CompiledGraph> {
    let parsed = pt2_parser::parse_pt2(pt2_path)?;
    let translated = translator::translate(&parsed)?;
    let mut graph = translated.graph;

    for (sym_name, c) in &translated.sym_map.sym_to_char {
        if let Some(rc) = translated.sym_map.ranges.get(sym_name) {
            graph.set_dim(*c, rc.min_val as usize);
        }
    }

    let output_shape_exprs: Vec<Vec<Expression>> = translated
        .output_ids
        .iter()
        .map(|(name, _id)| {
            parsed
                .tensor_meta(name)
                .map(|meta| resolve_dim_sizes(&meta.sizes, &translated.sym_map.sym_to_char))
                .unwrap_or_default()
        })
        .collect();

    let input_names: Vec<String> = translated
        .user_input_ids
        .iter()
        .map(|(name, _)| name.clone())
        .collect();
    let output_names: Vec<String> = translated
        .output_ids
        .iter()
        .map(|(name, _)| name.clone())
        .collect();

    let input_shape_exprs: Vec<Vec<Expression>> = translated
        .user_input_ids
        .iter()
        .map(|(name, _id)| {
            parsed
                .tensor_meta(name)
                .map(|meta| resolve_dim_sizes(&meta.sizes, &translated.sym_map.sym_to_char))
                .unwrap_or_default()
        })
        .collect();

    let user_input_sizes: Vec<(NodeIndex, usize)> = translated
        .user_input_ids
        .iter()
        .map(|(name, id)| {
            let meta = parsed.tensor_meta(name);
            let n_elements = meta
                .map(|m| {
                    m.sizes
                        .iter()
                        .map(|s| s.hint().unwrap_or(1) as usize)
                        .product()
                })
                .unwrap_or(1);
            (*id, n_elements)
        })
        .collect();

    let runtime = match backend {
        "cpu" | "native" => {
            graph.build_search_space::<NativeRuntime>();
            let mut rt = graph.search(NativeRuntime::default(), search_iters);
            if !weights_path.is_empty() {
                load_safetensors_native(&mut rt, &graph, weights_path)?;
            }
            load_constants_native(&mut rt, &graph, &parsed)?;
            RuntimeBackend::Native(rt)
        }
        "cuda" | "gpu" => init_cuda_runtime(
            &mut graph,
            weights_path,
            &parsed,
            &user_input_sizes,
            search_iters,
        )?,
        other => {
            anyhow::bail!("Unknown backend: {other}. Use 'cpu' or 'cuda'.");
        }
    };

    // Build tensor_ids from user inputs and outputs
    let mut tensor_ids: HashMap<String, NodeIndex> = HashMap::new();
    for (name, id) in &translated.user_input_ids {
        tensor_ids.insert(name.clone(), *id);
    }
    for (name, id) in &translated.output_ids {
        tensor_ids.insert(name.clone(), *id);
    }

    // Resolve concrete output shapes
    let output_shapes: Vec<Vec<usize>> = output_shape_exprs
        .iter()
        .map(|exprs| exprs.iter().map(|e| e.to_usize().unwrap_or(1)).collect())
        .collect();

    // Build dim_param_map from sym_map
    let dim_param_map: DimParamMap = translated.sym_map.sym_to_char;

    Ok(CompiledGraph {
        graph,
        runtime,
        tensor_ids,
        input_names,
        output_names,
        output_shapes,
        output_shape_exprs,
        input_shape_exprs,
        dim_param_map,
    })
}

#[cfg(feature = "cuda")]
fn init_cuda_runtime(
    graph: &mut LuminalGraph,
    weights_path: &str,
    parsed: &pt2_parser::ParsedPT2,
    user_input_sizes: &[(NodeIndex, usize)],
    search_iters: usize,
) -> anyhow::Result<RuntimeBackend> {
    let cuda_ctx =
        CudaContext::new(0).map_err(|e| anyhow::anyhow!("CUDA context init failed: {e}"))?;
    let stream = cuda_ctx.default_stream();

    graph.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);

    // Phase 1: Set ALL input nodes to safe dummy data (1.0) for search profiling.
    // Real weights/constants may contain -inf (e.g. causal attention mask) which
    // produce NaN in intermediate computations (e.g. -inf - (-inf) = NaN in softmax
    // decomposition), causing the search's has_nan_outputs check to reject ALL
    // candidates. We load real data only AFTER the search completes.
    set_all_inputs_dummy_cuda(&mut rt, graph, weights_path, parsed, user_input_sizes)?;

    let mut rt = graph.search(rt, search_iters);

    if !weights_path.is_empty() {
        load_safetensors_cuda(&mut rt, graph, weights_path)?;
    }
    load_constants_cuda(&mut rt, graph, parsed)?;

    Ok(RuntimeBackend::Cuda(Box::new(rt)))
}

#[cfg(not(feature = "cuda"))]
fn init_cuda_runtime(
    _graph: &mut LuminalGraph,
    _weights_path: &str,
    _parsed: &pt2_parser::ParsedPT2,
    _user_input_sizes: &[(NodeIndex, usize)],
    _search_iters: usize,
) -> anyhow::Result<RuntimeBackend> {
    anyhow::bail!("CUDA support not compiled. Rebuild with --features cuda")
}

// ---------------------------------------------------------------------------
// Weight loading
// ---------------------------------------------------------------------------

fn load_safetensors_impl(
    cx: &LuminalGraph,
    file_path: &str,
    mut set_data: impl FnMut(NodeIndex, Vec<f32>),
) -> anyhow::Result<()> {
    use memmap2::MmapOptions;
    use safetensors::SafeTensors;
    use std::fs::File;

    let f = File::open(file_path)?;
    let mmap = unsafe { MmapOptions::new().map(&f)? };
    let st = SafeTensors::deserialize(&mmap)
        .map_err(|e| anyhow::anyhow!("SafeTensors deserialize error: {e}"))?;

    for node in cx.graph.node_indices() {
        if let Some(input) = (*cx.graph[node])
            .as_any()
            .downcast_ref::<luminal::hlir::Input>()
            && let Ok(tensor) = st.tensor(&input.label)
        {
            let f32s = bytes_to_f32(tensor.data(), safetensors_dtype_to_pt2(tensor.dtype()));
            set_data(node, f32s);
        }
    }

    Ok(())
}

fn load_safetensors_native(
    rt: &mut NativeRuntime,
    cx: &LuminalGraph,
    file_path: &str,
) -> anyhow::Result<()> {
    load_safetensors_impl(cx, file_path, |node, data| rt.set_data(node, data))
}

#[cfg(feature = "cuda")]
fn load_safetensors_cuda(
    rt: &mut CudaRuntime,
    cx: &LuminalGraph,
    file_path: &str,
) -> anyhow::Result<()> {
    load_safetensors_impl(cx, file_path, |node, data| rt.set_data(node, data))
}

/// Set ALL input nodes to dummy 1.0 data for safe CUDA search profiling.
#[cfg(feature = "cuda")]
fn set_all_inputs_dummy_cuda(
    rt: &mut CudaRuntime,
    cx: &LuminalGraph,
    weights_path: &str,
    parsed: &pt2_parser::ParsedPT2,
    user_input_sizes: &[(NodeIndex, usize)],
) -> anyhow::Result<()> {
    use memmap2::MmapOptions;
    use safetensors::SafeTensors;
    use std::fs::File;

    let mut label_sizes: HashMap<String, usize> = HashMap::new();

    if !weights_path.is_empty() {
        let f = File::open(weights_path)?;
        let mmap = unsafe { MmapOptions::new().map(&f)? };
        let st = SafeTensors::deserialize(&mmap)
            .map_err(|e| anyhow::anyhow!("SafeTensors deserialize error: {e}"))?;
        for (name, info) in st.tensors() {
            let n: usize = info.shape().iter().product();
            label_sizes.insert(name.to_string(), n);
        }
    }

    if let Some(cc) = &parsed.constants_config {
        for (name, entry) in &cc.config {
            let n: usize = entry
                .tensor_meta
                .sizes
                .iter()
                .map(|s| s.hint().unwrap_or(1) as usize)
                .product();
            label_sizes.insert(name.clone(), n);
        }
    }

    for node_id in cx.graph.node_indices() {
        if let Some(input) = (*cx.graph[node_id])
            .as_any()
            .downcast_ref::<luminal::hlir::Input>()
        {
            if let Some(&n) = label_sizes.get(&input.label) {
                if n > 0 {
                    rt.set_data(node_id, vec![1.0f32; n]);
                }
            }
        }
    }

    for &(id, n_elements) in user_input_sizes {
        rt.set_data(id, vec![1.0f32; n_elements]);
    }

    Ok(())
}

/// Convert safetensors Dtype to PT2 dtype number.
fn safetensors_dtype_to_pt2(dtype: safetensors::Dtype) -> u32 {
    match dtype {
        safetensors::Dtype::BOOL => 12,
        safetensors::Dtype::U8 => 1,
        safetensors::Dtype::I8 => 2,
        safetensors::Dtype::I16 => 3,
        safetensors::Dtype::I32 => 4,
        safetensors::Dtype::I64 => 5,
        safetensors::Dtype::F16 => 6,
        safetensors::Dtype::F32 => 7,
        safetensors::Dtype::F64 => 8,
        safetensors::Dtype::BF16 => 13,
        _ => 7, // default to f32
    }
}

/// Convert raw bytes to f32 using PT2 dtype numbering.
fn bytes_to_f32(bytes: &[u8], dtype: u32) -> Vec<f32> {
    match dtype {
        7 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        6 => bytes
            .chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        13 => bytes
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        8 => bytes
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32)
            .collect(),
        5 => bytes
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32)
            .collect(),
        4 => bytes
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32)
            .collect(),
        3 => bytes
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32)
            .collect(),
        2 => bytes.iter().map(|&b| (b as i8) as f32).collect(),
        1 => bytes.iter().map(|&b| b as f32).collect(),
        12 => bytes
            .iter()
            .map(|&b| if b != 0 { 1.0 } else { 0.0 })
            .collect(),
        _ => {
            eprintln!("[luminal] Warning: unrecognized dtype {dtype}, interpreting as f32");
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        }
    }
}

fn load_constants_impl(
    cx: &LuminalGraph,
    parsed: &pt2_parser::ParsedPT2,
    mut set_data: impl FnMut(NodeIndex, Vec<f32>),
) -> anyhow::Result<()> {
    let constants_config = match &parsed.constants_config {
        Some(c) => c,
        None => return Ok(()),
    };

    for (name, entry) in &constants_config.config {
        let raw_bytes = match pt2_parser::read_constant_bytes(
            &parsed.pt2_path,
            &parsed.archive_prefix,
            entry,
        ) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "[luminal] Warning: failed to load constant '{}': {:#}",
                    name, e
                );
                continue;
            }
        };
        let f32_data = bytes_to_f32(&raw_bytes, entry.tensor_meta.dtype);

        for node_id in cx.graph.node_indices() {
            if let Some(input) = (*cx.graph[node_id])
                .as_any()
                .downcast_ref::<luminal::hlir::Input>()
                && input.label == *name
            {
                set_data(node_id, f32_data.clone());
            }
        }
    }
    Ok(())
}

fn load_constants_native(
    rt: &mut NativeRuntime,
    cx: &LuminalGraph,
    parsed: &pt2_parser::ParsedPT2,
) -> anyhow::Result<()> {
    load_constants_impl(cx, parsed, |node, data| rt.set_data(node, data))
}

#[cfg(feature = "cuda")]
fn load_constants_cuda(
    rt: &mut CudaRuntime,
    cx: &LuminalGraph,
    parsed: &pt2_parser::ParsedPT2,
) -> anyhow::Result<()> {
    load_constants_impl(cx, parsed, |node, data| rt.set_data(node, data))
}
