use std::{collections::HashMap, fs::File, io::Read, path::Path};

use luminal::prelude::*;
use prost::Message;
use thiserror::Error;

use crate::onnx::proto as onnx;

#[derive(Debug, Error)]
pub enum OnnxImportError {
    #[error("failed to read file: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to decode ONNX: {0}")]
    Decode(String),
    #[error("unsupported or missing op input: {0}")]
    MissingInput(String),
    #[error("unsupported data type {0}")]
    UnsupportedDtype(i32),
    #[error("shape not found or unsupported for op {0}")]
    BadShape(String),
    #[error("unsupported operator: {0}")]
    UnsupportedOp(String),
}

pub struct OnnxImportResult {
    pub graph: Box<Graph>,
    pub inputs: HashMap<String, GraphTensor>,
    pub outputs: HashMap<String, GraphTensor>,
}

struct Ctx {
    g: Box<Graph>,
    // mapping from value name to tensor
    env: HashMap<String, GraphTensor>,
    // input/output maps
    inputs: HashMap<String, GraphTensor>,
    outputs: HashMap<String, GraphTensor>,
    // map of dim_param -> char symbol
    symmap: HashMap<String, char>,
    next_sym: u8,
}

impl Ctx {
    fn new() -> Self {
        Self {
            g: Box::new(Graph::new()),
            env: HashMap::default(),
            inputs: HashMap::default(),
            outputs: HashMap::default(),
            symmap: HashMap::default(),
            next_sym: b'a',
        }
    }

    fn sym_for(&mut self, s: &str) -> char {
        if let Some(&c) = self.symmap.get(s) {
            return c;
        }
        // allocate a new ascii letter, wrap to 'a'..'z' then 'A'..'Z'
        let mut c = self.next_sym as char;
        if !(c.is_ascii_alphabetic()) {
            // fallback to 'a'
            c = 'a';
        }
        self.next_sym = self.next_sym.wrapping_add(1);
        self.symmap.insert(s.to_string(), c);
        c
    }
}

pub fn import_onnx(path: impl AsRef<Path>) -> Result<OnnxImportResult, OnnxImportError> {
    let mut f = File::open(path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    let model =
        onnx::ModelProto::decode(&*buf).map_err(|e| OnnxImportError::Decode(e.to_string()))?;
    let mut ctx = Ctx::new();
    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| OnnxImportError::BadShape("model.graph".into()))?;
    import_graph(&mut ctx, graph)?;
    Ok(OnnxImportResult {
        graph: ctx.g,
        inputs: ctx.inputs,
        outputs: ctx.outputs,
    })
}

fn import_graph(ctx: &mut Ctx, gp: &onnx::GraphProto) -> Result<(), OnnxImportError> {
    // Initializers
    for t in &gp.initializer {
        let name = t.name.clone();
        let shape = to_dims(ctx, &t.dims.iter().map(|&d| Some(d)).collect::<Vec<_>>());
        let data = tensor_to_f32(t)?;
        let gt = ctx.g.tensor(shape).set(data);
        ctx.env.insert(name, gt);
    }

    // Inputs (exclude initializers by name)
    for vi in &gp.input {
        if ctx.env.contains_key(&vi.name) {
            continue;
        }
        if let Some(shape) = value_info_shape(ctx, vi) {
            let gt = ctx.g.tensor(shape);
            ctx.inputs.insert(vi.name.clone(), gt);
            ctx.env.insert(vi.name.clone(), gt);
        } else {
            return Err(OnnxImportError::BadShape(format!("input {}", vi.name)));
        }
    }

    // Nodes
    for n in &gp.node {
        let op = n.op_type.as_str();
        match op {
            "Constant" => {
                // attribute 'value' -> TensorProto
                let out_name = n.output.first().cloned().unwrap_or_default();
                let attr = n
                    .attribute
                    .iter()
                    .find(|a| a.name == "value")
                    .and_then(|a| a.t.as_ref())
                    .ok_or_else(|| {
                        OnnxImportError::UnsupportedOp("Constant without value".into())
                    })?;
                let shape = to_dims(ctx, &attr.dims.iter().map(|&d| Some(d)).collect::<Vec<_>>());
                let data = tensor_to_f32(attr)?;
                let gt = ctx.g.tensor(shape).set(data);
                ctx.env.insert(out_name, gt);
            }
            "Add" | "Div" | "Mul" | "Sub" | "Max" | "Min" => {
                let mut a = get_input(ctx, n, 0)?;
                let mut b = get_input(ctx, n, 1)?;
                // naive broadcasting to the larger-rank tensor
                if a.shape.len() > b.shape.len() {
                    b = b.expand(a.shape);
                } else if b.shape.len() > a.shape.len() {
                    a = a.expand(b.shape);
                } else {
                    // same rank: try to expand dims where needed
                    let target = a.shape;
                    b = b.expand(target);
                }
                let out = match op {
                    "Add" => a + b,
                    "Sub" => a - b,
                    "Mul" => a * b,
                    "Div" => a / b,
                    "Max" => a.maximum(b),
                    "Min" => a.minimum(b),
                    _ => unreachable!(),
                };
                set_output(ctx, n, out);
            }
            "Relu" => {
                let a = get_input(ctx, n, 0)?;
                set_output(ctx, n, a.relu());
            }
            "Sigmoid" => {
                let a = get_input(ctx, n, 0)?;
                set_output(ctx, n, a.sigmoid());
            }
            "Tanh" => {
                let a = get_input(ctx, n, 0)?;
                set_output(ctx, n, a.tanh());
            }
            "Sqrt" => {
                let a = get_input(ctx, n, 0)?;
                set_output(ctx, n, a.sqrt());
            }
            "MatMul" => {
                let a = get_input(ctx, n, 0)?;
                let b = get_input(ctx, n, 1)?;
                set_output(ctx, n, a.matmul(b));
            }
            "Gemm" => {
                // Y = alpha * A * B + beta * C
                let a = get_input(ctx, n, 0)?;
                let b = get_input(ctx, n, 1)?;
                let c = get_input(ctx, n, 2).ok();
                let mut alpha = 1.0f32;
                let mut beta = 1.0f32;
                let mut trans_a = false;
                let mut trans_b = false;
                for a in &n.attribute {
                    match a.name.as_str() {
                        "alpha" => alpha = a.f,
                        "beta" => beta = a.f,
                        "transA" => trans_a = a.i != 0,
                        "transB" => trans_b = a.i != 0,
                        _ => {}
                    }
                }
                let mut aa = a;
                let mut bb = b;
                if trans_a {
                    let (m, n) = aa.dims2();
                    aa = aa.permute((1, 0)).reshape((n, m));
                }
                if trans_b {
                    let (m, n) = bb.dims2();
                    bb = bb.permute((1, 0)).reshape((n, m));
                }
                let mut y = aa.matmul(bb) * alpha;
                if let Some(cc) = c { y += cc * beta; }
                set_output(ctx, n, y);
            }
            "Softmax" => {
                // default axis is 1 in ONNX (older), sometimes -1 newer - we'll honor attr if present
                let a = get_input(ctx, n, 0)?;
                let mut axis: i64 = 1;
                for at in &n.attribute {
                    if at.name == "axis" {
                        axis = at.i;
                    }
                }
                // map negative axis
                let axis = normalize_axis(axis, a.shape.len());
                set_output(ctx, n, a.softmax(axis));
            }
            "Reshape" => {
                // input[1] is shape; must be constant/initializer
                let a = get_input(ctx, n, 0)?;
                let shape_name = n.input.get(1).cloned().unwrap_or_default();
                let shape_const = ctx
                    .env
                    .get(&shape_name)
                    .copied()
                    .ok_or_else(|| OnnxImportError::MissingInput(shape_name.clone()))?;
                // Pull shape values from Tensor if constant
                let new_shape_ints: Vec<i64> =
                    shape_const.data().into_iter().map(|e| e as i64).collect();
                let mut shape_spec: Vec<usize> = Vec::with_capacity(new_shape_ints.len());
                for v in new_shape_ints.iter() {
                    if *v == -1 {
                        shape_spec.push(usize::MAX - 1);
                    } else if *v == 0 {
                        shape_spec.push(usize::MAX - 2);
                    } else {
                        shape_spec.push(*v as usize);
                    }
                }
                let new_dims = infer_reshape_dims(a.shape.dims(), &shape_spec);
                set_output(ctx, n, a.reshape(new_dims));
            }
            "Transpose" => {
                let a = get_input(ctx, n, 0)?;
                // perm attr
                let mut perm: Vec<usize> = (0..a.shape.len()).collect();
                for at in &n.attribute {
                    if at.name == "perm" {
                        perm = at.ints.iter().map(|&i| i as usize).collect();
                    }
                }
                set_output(ctx, n, a.permute(perm));
            }
            "Unsqueeze" => {
                let a = get_input(ctx, n, 0)?;
                let mut axes: Vec<usize> = vec![];
                // Prefer second input as tensor of axes if present
                if n.input.len() > 1 {
                    if let Ok(ax_t) = get_input(ctx, n, 1) {
                        let vals: Vec<i64> = ax_t.data().into_iter().map(|e| e as i64).collect();
                        axes = vals
                            .iter()
                            .map(|&i| normalize_axis(i, a.shape.len() + 1))
                            .collect();
                    }
                }
                if axes.is_empty() {
                    for at in &n.attribute {
                        if at.name == "axes" {
                            axes = at
                                .ints
                                .iter()
                                .map(|&i| normalize_axis(i, a.shape.len() + 1))
                                .collect();
                        }
                    }
                }
                let mut out = a;
                // Insert in sorted order to maintain correct indexes
                axes.sort();
                for ax in axes { out = out.unsqueeze(ax); }
                set_output(ctx, n, out);
            }
            "Squeeze" => {
                let a = get_input(ctx, n, 0)?;
                let mut axes: Vec<usize> = vec![];
                if n.input.len() > 1 {
                    if let Ok(ax_t) = get_input(ctx, n, 1) {
                        let vals: Vec<i64> = ax_t.data().into_iter().map(|e| e as i64).collect();
                        axes = vals
                            .iter()
                            .map(|&i| normalize_axis(i, a.shape.len()))
                            .collect();
                    }
                }
                if axes.is_empty() {
                    for at in &n.attribute {
                        if at.name == "axes" {
                            axes = at
                                .ints
                                .iter()
                                .map(|&i| normalize_axis(i, a.shape.len()))
                                .collect();
                        }
                    }
                }
                let mut dims = a.dims();
                if axes.is_empty() {
                    dims.retain(|d| d.to_usize().unwrap_or(1) != 1);
                } else {
                    // remove given axes
                    axes.sort();
                    for ax in axes.into_iter().rev() {
                        dims.remove(ax);
                    }
                }
                set_output(ctx, n, a.reshape(dims));
            }
            "Concat" => {
                let axis = n
                    .attribute
                    .iter()
                    .find(|a| a.name == "axis")
                    .map(|a| a.i)
                    .unwrap_or(0);
                let mut it = n.input.iter();
                let first = get_input_by_name(ctx, it.next().unwrap())?;
                let mut out = first;
                for name in it {
                    let t = get_input_by_name(ctx, name)?;
                    out = out.concat_along(t, normalize_axis(axis, out.shape.len()));
                }
                set_output(ctx, n, out);
            }
            other => return Err(OnnxImportError::UnsupportedOp(other.to_string())),
        }
    }

    // Outputs
    for vo in &gp.output {
        let name = &vo.name;
        if let Some(&t) = ctx.env.get(name) {
            ctx.outputs.insert(name.clone(), t.retrieve());
        }
    }
    Ok(())
}

fn get_input(ctx: &Ctx, n: &onnx::NodeProto, idx: usize) -> Result<GraphTensor, OnnxImportError> {
    let name = n.input.get(idx).cloned().unwrap_or_default();
    get_input_by_name(ctx, &name)
}

fn get_input_by_name(ctx: &Ctx, name: &str) -> Result<GraphTensor, OnnxImportError> {
    ctx.env
        .get(name)
        .copied()
        .ok_or_else(|| OnnxImportError::MissingInput(name.to_string()))
}

fn set_output(ctx: &mut Ctx, n: &onnx::NodeProto, t: GraphTensor) {
    for out in &n.output {
        if !out.is_empty() {
            ctx.env.insert(out.clone(), t);
        }
    }
}

fn normalize_axis(axis: i64, rank: usize) -> usize {
    if axis >= 0 {
        axis as usize
    } else {
        (rank as i64 + axis) as usize
    }
}

fn infer_reshape_dims(old: Vec<Expression>, target: &[usize]) -> Vec<Expression> {
    // implements ONNX reshape rules for -1 and 0
    let mut new_dims: Vec<Expression> = Vec::with_capacity(target.len());
    let mut known: usize = 1;
    let mut infer_at: Option<usize> = None;
    for (i, &d) in target.iter().enumerate() {
        if d == usize::MAX - 1 {
            // sentinel for -1
            infer_at = Some(i);
            new_dims.push(1.into());
        } else if d == usize::MAX - 2 {
            // sentinel for 0 -> copy from input
            new_dims.push(old[i]);
            known *= new_dims.last().unwrap().to_usize().unwrap_or(1);
        } else {
            new_dims.push(d.into());
            known *= d;
        }
    }
    if let Some(ix) = infer_at {
        let total: usize = old.iter().map(|e| e.to_usize().unwrap_or(1)).product();
        let inferred = total / known.max(1);
        new_dims[ix] = inferred.into();
    }
    new_dims
}

fn value_info_shape(ctx: &mut Ctx, vi: &onnx::ValueInfoProto) -> Option<Vec<Expression>> {
    let t = vi.r#type.as_ref()?;
    let ten = match &t.value {
        Some(onnx::type_proto::Value::TensorType(tt)) => tt,
        _ => return None,
    };
    let dims = &ten.shape.as_ref()?.dim;
    // Build an intermediate vector first to avoid borrow issues
    let mut dim_vals: Vec<Option<i64>> = Vec::with_capacity(dims.len());
    for d in dims {
        match &d.value {
            Some(onnx::tensor_shape_proto_dimension::Value::DimValue(v)) => dim_vals.push(Some(*v)),
            Some(onnx::tensor_shape_proto_dimension::Value::DimParam(p)) => {
                let c = ctx.sym_for(p);
                dim_vals.push(Some(-(c as i64)));
            }
            None => dim_vals.push(None),
        }
    }
    Some(to_dims(ctx, &dim_vals))
}

fn to_dims(ctx: &mut Ctx, dims: &[Option<i64>]) -> Vec<Expression> {
    let mut out = vec![];
    for (i, d) in dims.iter().enumerate() {
        match d {
            Some(v) if *v >= 0 => out.push(Expression::from(*v as usize)),
            Some(v) if *v < 0 => {
                // negative marker used for dim_param -> char
                let c = (-*v) as u8 as char;
                out.push(Expression::from(c));
            }
            None => {
                // unknown -> allocate symbol based on position
                let c = ctx.sym_for(&format!("dim_{i}"));
                out.push(Expression::from(c));
            }
            _ => unreachable!(),
        }
    }
    out
}

fn tensor_to_f32(t: &onnx::TensorProto) -> Result<Vec<f32>, OnnxImportError> {
    use onnx::tensor_proto::DataType as Dt;
    let dt = t.data_type;
    let elem_count = t.dims.iter().map(|&d| d as usize).product::<usize>().max(1);
    if !t.raw_data.is_empty() {
        let raw = &t.raw_data;
        match dt {
            x if x == Dt::Float as i32 => {
                let mut out = vec![0f32; raw.len() / 4];
                for (i, chunk) in raw.chunks_exact(4).take(elem_count).enumerate() {
                    out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
                Ok(out)
            }
            x if x == Dt::Double as i32 => {
                let mut out = vec![0f32; raw.len() / 8];
                for (i, chunk) in raw.chunks_exact(8).take(elem_count).enumerate() {
                    out[i] = f64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ]) as f32;
                }
                Ok(out)
            }
            x if x == Dt::Int64 as i32 => {
                let mut out = vec![0f32; raw.len() / 8];
                for (i, chunk) in raw.chunks_exact(8).take(elem_count).enumerate() {
                    out[i] = i64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ]) as f32;
                }
                Ok(out)
            }
            x if x == Dt::Int32 as i32 => {
                let mut out = vec![0f32; raw.len() / 4];
                for (i, chunk) in raw.chunks_exact(4).take(elem_count).enumerate() {
                    out[i] = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32;
                }
                Ok(out)
            }
            x => Err(OnnxImportError::UnsupportedDtype(x)),
        }
    } else if !t.float_data.is_empty() {
        let mut out = t.float_data.clone();
        out.truncate(elem_count);
        Ok(out)
    } else if !t.int64_data.is_empty() {
        Ok(t.int64_data
            .iter()
            .take(elem_count)
            .map(|&v| v as f32)
            .collect())
    } else if !t.int32_data.is_empty() {
        Ok(t.int32_data
            .iter()
            .take(elem_count)
            .map(|&v| v as f32)
            .collect())
    } else {
        Ok(vec![0.0; elem_count])
    }
}
