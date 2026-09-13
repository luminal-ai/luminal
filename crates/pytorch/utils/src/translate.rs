//! ATen (PT2 `model.json`) -> recorder-frontend translator.
//!
//! SSA values stay SSA; boundary storage is stated once, at the graph
//! inputs/outputs. A functionalized in-place mutation (PT2
//! `user_input_mutation`) becomes `GraphTensor::output_into` on the mutated
//! input, which the binding layer pins to one buffer id.
//!
//! Coverage is honest: an unknown ATen target bails with its name. This is
//! the M4 translator re-attachment, rebuilt against the native recorder.

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow, bail};
use luminal::prelude::*;

mod conv;
mod expr;
mod index;
mod movement_more;
mod ops;
mod pooling;
mod special;
mod unary;
mod util;

use ops::ReductionOp;

use crate::dtype::TorchDType;
use crate::pt2_parser::{InputKind, ParsedPT2};
use crate::pt2_schema::{DimSize, ExprValue, Node, NodeInput, TensorMeta};

/// One bound graph input, in export order.
pub struct TranslatedInput {
    pub graph_name: String,
    /// The original checkpoint name for parameters/buffers (e.g. "0.weight").
    pub parameter_name: Option<String>,
    pub kind: InputKind,
    pub tensor: NodeIndex,
    pub dtype: DType,
    pub shape: Vec<usize>,
}

/// One graph output, in export order.
pub struct TranslatedOutput {
    pub graph_name: String,
    pub tensor: NodeIndex,
    pub dtype: DType,
    pub shape: Vec<usize>,
    /// For a `user_input_mutation` output: the graph name of the input it
    /// writes into. These are writebacks, not returned tensors.
    pub mutation_target: Option<String>,
    /// Whether this output is part of the caller's returned pytree. A
    /// mutation-only sink is `false`; a mutated input that the model also
    /// returns is one output with `returned = true` and a target.
    pub returned: bool,
}

/// A translated program: the recorder graph plus its boundary tables.
pub struct Translation {
    pub graph: Graph,
    pub inputs: Vec<TranslatedInput>,
    pub outputs: Vec<TranslatedOutput>,
    /// Symbolic dim -> the concrete hint torch exported with it.
    pub dims: HashMap<String, usize>,
}

struct Translator<'a> {
    cx: Graph,
    values: HashMap<String, GraphTensor>,
    /// Input values, never shadowed by a later in-place rebinding.
    input_values: HashMap<String, GraphTensor>,
    /// Writeback sinks registered by in-place ATen nodes, in dispatch order.
    sinks: Vec<TranslatedOutput>,
    /// Value id -> index in `sinks`, so a graph output that returns the
    /// mutated value marks ITS sink as returned instead of adding a second.
    sink_by_value: HashMap<NodeIndex, usize>,
    /// PT2 symbol name -> recorder dim symbol (dynamic dims).
    symbols: HashMap<String, Symbol>,
    parsed: &'a ParsedPT2,
    dims: HashMap<String, usize>,
}

/// Translate a parsed PT2 program into the recorder frontend.
pub fn translate(parsed: &ParsedPT2) -> Result<Translation> {
    let mut t = Translator {
        cx: Graph::new(),
        values: HashMap::new(),
        input_values: HashMap::new(),
        sinks: Vec::new(),
        sink_by_value: HashMap::new(),
        symbols: expr::build_sym_map(parsed),
        parsed,
        dims: HashMap::new(),
    };

    let kinds: HashMap<String, InputKind> = parsed
        .classify_inputs()
        .into_iter()
        .map(|kind| {
            let name = match &kind {
                InputKind::Parameter { graph_name, .. }
                | InputKind::Buffer { graph_name, .. }
                | InputKind::UserInput { graph_name } => graph_name.clone(),
            };
            (name, kind)
        })
        .collect();

    // 1. Inputs, in export order.
    let mut inputs = Vec::new();
    for tref in &parsed.program.graph_module.graph.inputs {
        let name = tref
            .value_name()
            .ok_or_else(|| anyhow!("graph input has no tensor/scalar name: {tref:?}"))?
            .to_string();
        let meta = t
            .tensor_meta(&name)
            .with_context(|| format!("input {name} has no tensor metadata"))?
            .clone();
        let dtype = dtype_of(meta.dtype)?;
        let shape = t.static_shape(&meta, &name)?;
        let tensor = t.cx.named_tensor(&name, shape.as_slice(), dtype);
        t.values.insert(name.clone(), tensor);
        t.input_values.insert(name.clone(), tensor);
        let kind = kinds.get(&name).cloned().unwrap_or(InputKind::UserInput {
            graph_name: name.clone(),
        });
        let parameter_name = match &kind {
            InputKind::Parameter { original_name, .. } => Some(original_name.clone()),
            InputKind::Buffer { original_name, .. } => Some(original_name.clone()),
            InputKind::UserInput { .. } => None,
        };
        inputs.push(TranslatedInput {
            graph_name: name,
            parameter_name,
            kind,
            tensor: tensor.id,
            dtype,
            shape,
        });
    }

    // 2. Nodes, in topological order.
    for node in &parsed.program.graph_module.graph.nodes {
        t.dispatch(node)
            .with_context(|| format!("translating `{}`", node.target))?;
    }

    // 3. Outputs, in export order. Mutations write back into their target
    //    input's storage instead of materializing a new boundary. An
    //    in-place ATen node already registered its sink while dispatching;
    //    a graph output that returns that value marks the sink as returned
    //    rather than adding a second boundary.
    let output_specs = &parsed.program.graph_module.signature.output_specs;
    let mut regular: Vec<TranslatedOutput> = Vec::new();
    for (position, tref) in parsed.program.graph_module.graph.outputs.iter().enumerate() {
        let name = tref
            .value_name()
            .ok_or_else(|| anyhow!("graph output has no tensor name: {tref:?}"))?
            .to_string();
        let meta = t
            .tensor_meta(&name)
            .with_context(|| format!("output {name} has no tensor metadata"))?
            .clone();
        let dtype = dtype_of(meta.dtype)?;
        let shape = t.static_shape(&meta, &name)?;
        let value = *t
            .values
            .get(&name)
            .ok_or_else(|| anyhow!("output {name} was never produced"))?;

        if let Some(&sink) = t.sink_by_value.get(&value.id) {
            if !matches!(
                output_specs.get(position),
                Some(crate::pt2_schema::OutputSpec::UserInputMutation { .. })
            ) {
                t.sinks[sink].returned = true;
            }
            continue;
        }

        let mutation_target = match output_specs.get(position) {
            Some(crate::pt2_schema::OutputSpec::UserInputMutation {
                user_input_mutation,
            }) => {
                let target_name = user_input_mutation.user_input_name.clone();
                let target = *t.input_values.get(&target_name).ok_or_else(|| {
                    anyhow!("mutation output {name} targets unknown input {target_name:?}")
                })?;
                value.output_into(&target);
                Some(target_name)
            }
            _ => {
                value.output();
                None
            }
        };
        regular.push(TranslatedOutput {
            graph_name: name,
            tensor: value.id,
            dtype,
            shape,
            mutation_target,
            returned: true,
        });
    }

    let mut outputs = std::mem::take(&mut t.sinks);
    outputs.extend(regular);

    Ok(Translation {
        graph: t.cx,
        inputs,
        outputs,
        dims: t.dims,
    })
}

fn dtype_of(code: u32) -> Result<DType> {
    let torch =
        TorchDType::from_code(code).map_err(|code| anyhow!("unknown PT2 dtype code {code}"))?;
    DType::try_from(torch).map_err(|t| anyhow!("unsupported dtype {t:?}"))
}

impl Translator<'_> {
    fn tensor_meta(&self, name: &str) -> Result<&TensorMeta> {
        self.parsed
            .program
            .graph_module
            .graph
            .tensor_values
            .get(name)
            .ok_or_else(|| anyhow!("no tensor_values entry for {name}"))
    }

    /// Resolve a tensor's dims to concrete sizes. Symbols use the hint torch
    /// exported with them; a symbol without a hint refuses (dynamic shapes
    /// are a later milestone).
    fn static_shape(&mut self, meta: &TensorMeta, name: &str) -> Result<Vec<usize>> {
        let mut shape = Vec::with_capacity(meta.sizes.len());
        for size in &meta.sizes {
            match size {
                DimSize::Int(i) => {
                    shape.push(usize::try_from(i.as_int).context("negative dim")?);
                }
                DimSize::Expr(expr) => {
                    let symbol = symbol_of(&expr.as_expr)
                        .ok_or_else(|| anyhow!("{name}: unreadable dim expression"))?;
                    let hint = expr
                        .as_expr
                        .hint
                        .as_ref()
                        .and_then(|h| h.as_int())
                        .ok_or_else(|| {
                            anyhow!("{name}: dynamic dim {symbol:?} has no concrete hint")
                        })?;
                    self.dims.insert(symbol.clone(), usize::try_from(hint)?);
                    shape.push(usize::try_from(hint)?);
                }
            }
        }
        Ok(shape)
    }

    /// Resolve a node operand: a tensor reference, a scalar literal, or a
    /// symbolic reference (unsupported today).
    fn operand(&mut self, input: &NodeInput) -> Result<GraphTensor> {
        if let Some(name) = input.arg.as_tensor_name() {
            return self
                .values
                .get(name)
                .copied()
                .ok_or_else(|| anyhow!("operand {name:?} was never produced"));
        }
        bail!(
            "operand {:?} is not a tensor (arg: {:?})",
            input.name,
            input.arg
        )
    }

    fn optional_tensor_operand(&mut self, input: &NodeInput) -> Result<Option<GraphTensor>> {
        if input.arg.as_tensor_name().is_some() {
            return Ok(Some(self.operand(input)?));
        }
        Ok(None)
    }

    /// A scalar literal as a rank-0 tensor of `dtype`.
    fn scalar(&mut self, input: &NodeInput, dtype: DType) -> Result<GraphTensor> {
        if let Some(v) = input.arg.as_float() {
            return Ok(self.cx.constant_f32(v as f32).cast(dtype));
        }
        if let Some(v) = input.arg.as_int() {
            return Ok(self.cx.constant_i32(v).cast(dtype));
        }
        bail!(
            "operand {:?} is not a scalar (arg: {:?})",
            input.name,
            input.arg
        )
    }

    /// Bind a node's outputs to fresh SSA values. Single-output only today.
    fn bind_outputs(&mut self, node: &Node, mut values: Vec<GraphTensor>) -> Result<()> {
        if node.outputs.len() != values.len() {
            bail!(
                "`{}` produced {} values but {} outputs were bound",
                node.target,
                values.len(),
                node.outputs.len()
            );
        }
        for (tref, value) in node.outputs.iter().zip(values.drain(..)) {
            let name = tref
                .value_name()
                .ok_or_else(|| anyhow!("`{}` wrote an unnameable output", node.target))?
                .to_string();
            self.values.insert(name, value);
        }
        Ok(())
    }

    fn dispatch(&mut self, node: &Node) -> Result<()> {
        let target = node
            .target
            .strip_prefix("torch.ops.aten.")
            .or_else(|| node.target.strip_prefix("torch.ops."))
            .unwrap_or(&node.target);

        // Assertion nodes carry no dataflow; they never bind outputs.
        if matches!(
            target,
            "_assert_tensor_metadata.default" | "_assert_scalar.default"
        ) {
            return Ok(());
        }

        let n = &node.inputs;

        let value = match target {
            // ---- linear ----
            "linear.default" => {
                let x = self.operand(&n[0])?;
                let w = self.operand(&n[1])?;
                let mut out = x.matmul(w.t());
                if let Some(bias) = n
                    .get(2)
                    .map(|b| self.optional_tensor_operand(b))
                    .transpose()?
                    .flatten()
                {
                    let dims = out.dims();
                    out += broadcast_to(bias, &dims);
                }
                out
            }
            // ---- matmul family ----
            "mm.default" | "bmm.default" | "matmul.default" => {
                let a = self.operand(&n[0])?;
                let b = self.operand(&n[1])?;
                a.matmul(b)
            }
            // ---- elementwise unary ----
            "relu.default" => self.operand(&n[0])?.relu(),
            "sigmoid.default" => self.operand(&n[0])?.sigmoid(),
            "tanh.default" => self.operand(&n[0])?.tanh(),
            "log.default" => self.operand(&n[0])?.log(),
            "sqrt.default" => self.operand(&n[0])?.sqrt(),
            "abs.default" => self.operand(&n[0])?.abs(),
            "neg.default" => -self.operand(&n[0])?,
            "silu.default" => self.operand(&n[0])?.silu(),
            "gelu.default" => self.operand(&n[0])?.gelu(),
            "reciprocal.default" => self.operand(&n[0])?.reciprocal(),
            "sin.default" => self.operand(&n[0])?.sin(),
            "square.default" => self.operand(&n[0])?.square(),
            // ---- rounding (dtype-preserving; no casts) ----
            "floor.default" => self.operand(&n[0])?.floor(),
            "ceil.default" => self.operand(&n[0])?.ceil(),
            "trunc.default" => self.operand(&n[0])?.trunc(),
            "round.default" => self.operand(&n[0])?.round(),
            "round.decimals" => {
                let decimals = n.get(1).and_then(|i| i.arg.as_int()).unwrap_or(0);
                if decimals != 0 {
                    bail!("round(decimals={decimals}) is not ported (only decimals=0)");
                }
                self.operand(&n[0])?.round()
            }
            // ---- elementwise binary ----
            "add.Tensor" => self.binary(n, |a, b| a + b)?,
            "sub.Tensor" => self.binary(n, |a, b| a - b)?,
            "mul.Tensor" => self.binary(n, |a, b| a * b)?,
            "div.Tensor" => self.binary(n, |a, b| a / b)?,
            "maximum.default" => self.binary(n, |a, b| a.maximum(b))?,
            "minimum.default" => self.binary(n, |a, b| a.maximum(b * -1.0) * -1.0)?,
            // ---- in-place (functional SSA + caller-storage writeback) ----
            "add_.Tensor" => self.inplace_binary(node, |a, b| a + b)?,
            "sub_.Tensor" => self.inplace_binary(node, |a, b| a - b)?,
            "mul_.Tensor" => self.inplace_binary(node, |a, b| a * b)?,
            "div_.Tensor" => self.inplace_binary(node, |a, b| a / b)?,
            "relu_.default" => self.inplace_unary(node, |x| x.relu())?,
            "sigmoid_.default" => self.inplace_unary(node, |x| x.sigmoid())?,
            "tanh_.default" => self.inplace_unary(node, |x| x.tanh())?,
            "copy_.default" => {
                let x = self.operand(&n[0])?;
                let y = self.operand(&n[1])?;
                let value = y.cast(x.dtype);
                self.register_inplace(node, value)?;
                value
            }
            // ---- movement ----
            "t.default" => self.operand(&n[0])?.t(),
            "transpose.int" => {
                let x = self.operand(&n[0])?;
                let d0 = self.int_arg(&n[1])? as usize;
                let d1 = self.int_arg(&n[2])? as usize;
                x.transpose(d0, d1)
            }
            "permute.default" => {
                let x = self.operand(&n[0])?;
                let axes = self.ints_arg(&n[1])?;
                x.permute(normalize_axes(&axes, x.rank())?)
            }
            "unsqueeze.default" => {
                let x = self.operand(&n[0])?;
                let dim = self.int_arg(&n[1])?;
                let dim = if dim < 0 {
                    dim + x.rank() as i64 + 1
                } else {
                    dim
                } as usize;
                x.unsqueeze(dim)
            }
            "squeeze.dim" => {
                let x = self.operand(&n[0])?;
                let dim = self.int_arg(&n[1])?;
                let dim = if dim < 0 { dim + x.rank() as i64 } else { dim } as usize;
                x.squeeze(dim)
            }
            // ---- reductions ----
            "sum.default" | "sum.dim_IntList" => {
                self.translate_reduction(node, ReductionOp::Sum)?
            }
            "mean.default" | "mean.dim" => self.translate_reduction(node, ReductionOp::Mean)?,
            "max.default" | "amax.default" => self.translate_reduction(node, ReductionOp::Max)?,
            "min.default" | "amin.default" => self.translate_reduction(node, ReductionOp::Min)?,
            "prod.default" | "prod.dim_int" => self.translate_reduction(node, ReductionOp::Prod)?,
            "argmax.default" => self.translate_argextremum(node, true)?,
            "argmin.default" => self.translate_argextremum(node, false)?,
            "var.default" | "var.dim" | "var.correction" => self.translate_var(node, false)?,
            "std.default" | "std.dim" | "std.correction" => self.translate_var(node, true)?,
            "cumsum.default" => self.translate_cumulative(node, false)?,
            "cumprod.default" => self.translate_cumulative(node, true)?,
            "max.dim" => {
                self.translate_dim_extremum(node, true)?;
                return Ok(());
            }
            "min.dim" => {
                self.translate_dim_extremum(node, false)?;
                return Ok(());
            }
            // ---- comparisons and logic ----
            "eq.Tensor" => self.comparison(node, |a, b| a.eq(b))?,
            "ne.Tensor" => self.comparison(node, |a, b| a.ne(b))?,
            "lt.Tensor" => self.comparison(node, |a, b| a.lt(b))?,
            "le.Tensor" => self.comparison(node, |a, b| a.le(b))?,
            "gt.Tensor" => self.comparison(node, |a, b| a.gt(b))?,
            "ge.Tensor" => self.comparison(node, |a, b| a.ge(b))?,
            "eq.Scalar" => self.comparison(node, |a, b| a.eq(b))?,
            "ne.Scalar" => self.comparison(node, |a, b| a.ne(b))?,
            "lt.Scalar" => self.comparison(node, |a, b| a.lt(b))?,
            "le.Scalar" => self.comparison(node, |a, b| a.le(b))?,
            "gt.Scalar" => self.comparison(node, |a, b| a.gt(b))?,
            "ge.Scalar" => self.comparison(node, |a, b| a.ge(b))?,
            "logical_and.default" => self.logical_binary(node, false, false)?,
            "logical_or.default" => self.logical_binary(node, true, false)?,
            "logical_xor.default" => self.logical_binary(node, false, true)?,
            "logical_not.default" => {
                let x = self.operand(&n[0])?;
                let one = self.cx.constant_f32(1.0).expand_rhs(x.dims());
                (one - x.cast(DType::F32)).cast(DType::Bool)
            }
            // ---- power / modulo ----
            "pow.Tensor_Scalar" => self.pow_tensor_scalar(node)?,
            "pow.Tensor_Tensor" => self.pow_tensor_tensor(node)?,
            "pow.Scalar" => self.pow_scalar_base(node)?,
            "fmod.Tensor" | "fmod.Scalar" => self.fmod_remainder(node, true)?,
            "remainder.Tensor" | "remainder.Scalar" => self.fmod_remainder(node, false)?,
            // ---- movement batch ----
            "view.default" | "reshape.default" | "_unsafe_view.default" => {
                self.translate_view(node)?
            }
            "flatten.using_ints" | "flatten.default" => self.translate_flatten(node)?,
            "slice.Tensor" => self.translate_slice(node)?,
            "select.int" => self.translate_select(node)?,
            "expand.default" => self.translate_expand(node)?,
            "repeat.default" => self.translate_repeat(node)?,
            "clone.default" | "alias.default" => self.operand(&n[0])?,
            "stack.default" => self.translate_stack(node)?,
            // ---- creation / selection ----
            "full.default" => self.translate_full(node, false)?,
            "full_like.default" => self.translate_full(node, true)?,
            "zeros_like.default" => self.translate_like_fill(node, 0.0)?,
            "ones_like.default" => self.translate_like_fill(node, 1.0)?,
            "arange.start_step" => self.translate_arange(node, 2)?,
            "arange.start" => self.translate_arange(node, 1)?,
            "arange.default" => self.translate_arange(node, 0)?,
            "scalar_tensor.default" => self.translate_scalar_tensor(node)?,
            "where.self" => self.translate_where(node, false)?,
            "where.ScalarOther" => self.translate_where(node, true)?,
            "masked_fill.Scalar" => self.translate_masked_fill_scalar(node)?,
            "clamp.default" => self.translate_clamp(node)?,
            "clamp.Tensor" => self.translate_clamp_tensor(node)?,
            "softmax.int" | "_softmax.default" => self.translate_softmax(node, false)?,
            "log_softmax.int" | "_log_softmax.default" => self.translate_softmax(node, true)?,
            "embedding.default" => self.translate_embedding(node)?,
            "item.default" | "_local_scalar_dense.default" => {
                self.translate_item(node)?;
                return Ok(());
            }
            "native_layer_norm.default" | "layer_norm.default" => {
                self.translate_layer_norm(node)?;
                return Ok(());
            }
            // ---- unary / special functions (batch 3) ----
            "exp.default" => self.translate_exp(node)?,
            "expm1.default" => self.translate_expm1(node)?,
            "log1p.default" => self.translate_log1p(node)?,
            "log10.default" => self.translate_log10(node)?,
            "rsqrt.default" => self.translate_rsqrt(node)?,
            "sinh.default" => self.translate_sinh(node)?,
            "cosh.default" => self.translate_cosh(node)?,
            "tan.default" => self.translate_tan(node)?,
            "cos.default" => self.translate_cos(node)?,
            "asin.default" => self.translate_asin(node)?,
            "acos.default" => self.translate_acos(node)?,
            "atan.default" => self.translate_atan(node)?,
            "asinh.default" => self.translate_asinh(node)?,
            "acosh.default" => self.translate_acosh(node)?,
            "atanh.default" => self.translate_atanh(node)?,
            "hardtanh.default" => self.translate_hardtanh(node)?,
            "elu.default" => self.translate_elu(node)?,
            "erf.default" => self.translate_erf(node)?,
            "erfc.default" => self.translate_erfc(node)?,
            "sign.default" => self.translate_sign(node)?,
            "signbit.default" => self.translate_signbit(node)?,
            "isinf.default" => self.translate_isinf(node)?,
            "bitwise_not.default" => self.translate_bitwise_not(node)?,
            "ldexp.Tensor" => self.translate_ldexp(node)?,
            "floor_divide.default" => self.translate_floor_divide(node)?,
            "div.Tensor_mode" => self.translate_div_tensor_mode(node)?,
            // ---- pooling / conv / norms (batch 4) ----
            "avg_pool2d.default" => self.translate_avg_pool(node, 2)?,
            "avg_pool3d.default" => self.translate_avg_pool(node, 3)?,
            "_adaptive_avg_pool2d.default" => self.translate_adaptive_avg_pool(node, 2)?,
            "_adaptive_avg_pool3d.default" => self.translate_adaptive_avg_pool(node, 3)?,
            "max_pool2d_with_indices.default" => {
                self.translate_max_pool(node, 2)?;
                return Ok(());
            }
            "max_pool3d_with_indices.default" => {
                self.translate_max_pool(node, 3)?;
                return Ok(());
            }
            "adaptive_max_pool2d.default" => {
                self.translate_adaptive_max_pool(node, 2)?;
                return Ok(());
            }
            "adaptive_max_pool3d.default" => {
                self.translate_adaptive_max_pool(node, 3)?;
                return Ok(());
            }
            "convolution.default" => self.translate_conv(node)?,
            "conv2d.default" => self.translate_conv(node)?,
            "max_pool2d.default" => {
                self.translate_max_pool(node, 2)?;
                return Ok(());
            }
            "pad.default" => self.translate_constant_pad_nd(node)?,
            "_native_batch_norm_legit.no_stats"
            | "_native_batch_norm_legit_no_training.default"
            | "_native_batch_norm_legit_functional.default"
            | "_batch_norm_with_update_functional.default"
            | "batch_norm.default" => {
                self.translate_batch_norm_functional(node)?;
                return Ok(());
            }
            "_fused_rms_norm.default" => self.translate_fused_rms_norm(node)?,
            "native_group_norm.default" | "group_norm.default" => {
                self.translate_group_norm(node)?;
                return Ok(());
            }
            // ---- cast ----
            "to.dtype" | "_to_copy.default" => {
                let x = self.operand(&n[0])?;
                let dtype = self.scalar_type_arg(&n[1])?;
                // Lossless casts go through `cast`; float -> int is the
                // explicit truncating conversion (`torch.int()`).
                if is_float(x.dtype) && is_int(dtype) {
                    x.trunc_cast(dtype)
                } else {
                    x.cast(dtype)
                }
            }
            // ---- cat ----
            "cat.default" => {
                let tensors = n[0]
                    .arg
                    .as_tensors()
                    .ok_or_else(|| anyhow!("cat: first operand is not a tensor list"))?;
                let axis = self.int_arg(&n[1])? as usize;
                let mut values = Vec::with_capacity(tensors.len());
                for t in tensors {
                    values.push(
                        *self
                            .values
                            .get(&t.name)
                            .ok_or_else(|| anyhow!("cat: unknown tensor {}", t.name))?,
                    );
                }
                let mut iter = values.into_iter();
                let mut acc = iter.next().ok_or_else(|| anyhow!("cat: empty list"))?;
                for next in iter {
                    acc = acc.concat_along(next, axis);
                }
                acc
            }
            // ---- index / scatter (batch 5) ----
            "index.Tensor" => self.translate_index_tensor(node)?,
            "index_select.default" => self.translate_index_select(node)?,
            "gather.default" => self.translate_gather(node)?,
            "scatter.src" => self.translate_scatter(node, 0)?,
            "scatter.value" => self.translate_scatter(node, 1)?,
            "scatter.reduce" => self.translate_scatter(node, 2)?,
            "scatter.value_reduce" => self.translate_scatter(node, 3)?,
            "scatter_add.default" => self.translate_scatter(node, 4)?,
            "scatter_reduce.two" => self.translate_scatter(node, 5)?,
            "index_put_.default" | "index_put.default" => self.translate_index_put(node)?,
            "index_reduce.default" => self.translate_index_reduce(node)?,
            "masked_scatter.default" => self.translate_masked_scatter(node)?,
            "put.default" => self.translate_put(node)?,
            "nonzero_static.default" => self.translate_nonzero_static(node)?,
            "_embedding_bag_forward_only.default" => {
                self.translate_embedding_bag(node)?;
                return Ok(());
            }
            // ---- movement / selection odds and ends (batch 5) ----
            "flip.default" => self.translate_flip(node)?,
            "diagonal.default" => self.translate_diagonal(node)?,
            "diagonal_scatter.default" => self.translate_diagonal_scatter(node)?,
            // `F.unfold` exports as `im2col.default`; both share one lowering.
            "unfold.default" | "im2col.default" => self.translate_unfold(node)?,
            "narrow_copy.default" => self.translate_narrow_copy(node)?,
            "unbind_copy.int" => {
                self.translate_unbind_copy(node)?;
                return Ok(());
            }
            "split_with_sizes.default" => {
                self.translate_split_with_sizes(node)?;
                return Ok(());
            }
            "repeat_interleave.Tensor"
            | "repeat_interleave.self_int"
            | "repeat_interleave.self_Tensor" => self.translate_repeat_interleave(node)?,
            "constant_pad_nd.default" => self.translate_constant_pad_nd(node)?,
            "topk.default" => {
                self.translate_topk(node)?;
                return Ok(());
            }
            "sort.default" => {
                self.translate_sort(node, false)?;
                return Ok(());
            }
            "sort.stable" => {
                self.translate_sort(node, true)?;
                return Ok(());
            }
            "argsort.default" => self.translate_argsort(node)?,
            "cummax.default" => {
                self.translate_cumextremum(node, true)?;
                return Ok(());
            }
            "cummin.default" => {
                self.translate_cumextremum(node, false)?;
                return Ok(());
            }
            "median.default" | "median.dim" => {
                self.translate_median(node, false)?;
                return Ok(());
            }
            "nanmedian.default" | "nanmedian.dim" => {
                self.translate_median(node, true)?;
                return Ok(());
            }
            // ---- special functions (batch 5) ----
            bessel @ ("i0.default"
            | "special_i0e.default"
            | "special_i1.default"
            | "special_i1e.default"
            | "special_modified_bessel_i0.default"
            | "special_modified_bessel_i1.default") => {
                let order = usize::from(bessel.contains("i1"));
                self.translate_modified_bessel(
                    node,
                    order,
                    bessel.contains("i0e") || bessel.contains("i1e"),
                )?
            }
            "special_spherical_bessel_j0.default" => self.translate_spherical_bessel_j0(node)?,
            bessel @ ("special_bessel_j0.default"
            | "special_bessel_j1.default"
            | "special_bessel_y0.default"
            | "special_bessel_y1.default") => {
                let order = usize::from(bessel.contains("j1") || bessel.contains("y1"));
                self.translate_cylindrical_bessel(node, order, bessel.contains("_y"))?
            }
            bessel @ ("special_modified_bessel_k0.default"
            | "special_modified_bessel_k1.default"
            | "special_scaled_modified_bessel_k0.default"
            | "special_scaled_modified_bessel_k1.default") => {
                let order = usize::from(bessel.contains("k1"));
                self.translate_modified_bessel_k(node, order, bessel.contains("scaled"))?
            }
            "special_airy_ai.default" => self.translate_airy_ai(node)?,
            "special_ndtri.default" => self.translate_ndtri(node)?,
            "erfinv.default" => self.translate_erfinv(node)?,
            // Public aliases of the private ATen spellings.
            "adaptive_avg_pool2d.default" => self.translate_adaptive_avg_pool(node, 2)?,
            "adaptive_avg_pool3d.default" => self.translate_adaptive_avg_pool(node, 3)?,
            "narrow.default" => self.translate_narrow_copy(node)?,
            "special_i0.default" => self.translate_modified_bessel(node, 0, false)?,
            cheb @ ("special_chebyshev_polynomial_t.default"
            | "special_chebyshev_polynomial_u.default"
            | "special_chebyshev_polynomial_v.default"
            | "special_chebyshev_polynomial_w.default"
            | "special_shifted_chebyshev_polynomial_t.default"
            | "special_shifted_chebyshev_polynomial_u.default"
            | "special_shifted_chebyshev_polynomial_v.default"
            | "special_shifted_chebyshev_polynomial_w.default"
            | "special_chebyshev_polynomial_t.n_scalar"
            | "special_chebyshev_polynomial_u.n_scalar"
            | "special_chebyshev_polynomial_v.n_scalar"
            | "special_chebyshev_polynomial_w.n_scalar"
            | "special_shifted_chebyshev_polynomial_t.n_scalar"
            | "special_shifted_chebyshev_polynomial_u.n_scalar"
            | "special_shifted_chebyshev_polynomial_v.n_scalar"
            | "special_shifted_chebyshev_polynomial_w.n_scalar") => {
                let kind = if cheb.contains("_t.") {
                    0
                } else if cheb.contains("_u.") {
                    1
                } else if cheb.contains("_v.") {
                    2
                } else {
                    3
                };
                self.translate_chebyshev_polynomial(node, kind, cheb.contains("shifted"))?
            }
            "lgamma.default" => self.translate_lgamma(node)?,
            "digamma.default" => self.translate_digamma(node)?,
            "polygamma.default" => self.translate_polygamma(node)?,
            "special_erfcx.default" => self.translate_erfcx(node)?,
            "logcumsumexp.default" => self.translate_logcumsumexp(node)?,
            "angle.default" => self.translate_angle(node)?,
            _ => bail!("unsupported ATen op `{}`", node.target),
        };

        self.bind_outputs(node, vec![value])
    }

    fn binary(
        &mut self,
        n: &[NodeInput],
        op: impl FnOnce(GraphTensor, GraphTensor) -> GraphTensor,
    ) -> Result<GraphTensor> {
        let a = self.operand(&n[0])?;
        let b = if let Some(t) = self.optional_tensor_operand(&n[1])? {
            t
        } else {
            self.scalar(&n[1], a.dtype)?
        };
        let (a, b) = broadcast_pair(a, b);
        Ok(op(a, b))
    }

    fn inplace_binary(
        &mut self,
        node: &Node,
        op: impl FnOnce(GraphTensor, GraphTensor) -> GraphTensor,
    ) -> Result<GraphTensor> {
        let value = self.binary(&node.inputs, op)?;
        self.register_inplace(node, value)?;
        Ok(value)
    }

    fn inplace_unary(
        &mut self,
        node: &Node,
        op: impl FnOnce(GraphTensor) -> GraphTensor,
    ) -> Result<GraphTensor> {
        let value = op(self.operand(&node.inputs[0])?);
        self.register_inplace(node, value)?;
        Ok(value)
    }

    /// An in-place result targeting a graph input registers a writeback
    /// sink: the input and the result share one boundary buffer, so the
    /// caller's tensor is updated. Mutations of intermediates need no
    /// writeback (SSA already carries the new value).
    fn register_inplace(&mut self, node: &Node, value: GraphTensor) -> Result<()> {
        let Some(target_name) = node.inputs[0].arg.as_tensor_name().map(str::to_string) else {
            return Ok(());
        };
        let Some(&target) = self.input_values.get(&target_name) else {
            return Ok(());
        };
        if target.dtype != value.dtype {
            bail!(
                "in-place `{}` changes dtype from {:?} to {:?}",
                node.target,
                target.dtype,
                value.dtype
            );
        }
        let output_name = node.outputs[0]
            .value_name()
            .ok_or_else(|| anyhow!("in-place `{}` has no output name", node.target))?
            .to_string();
        let meta = self.tensor_meta(&output_name)?.clone();
        let shape = self.static_shape(&meta, &output_name)?;
        value.output_into(&target);
        self.sink_by_value.insert(value.id, self.sinks.len());
        self.sinks.push(TranslatedOutput {
            graph_name: output_name,
            tensor: value.id,
            dtype: value.dtype,
            shape,
            mutation_target: Some(target_name),
            returned: false,
        });
        Ok(())
    }

    fn int_arg(&mut self, input: &NodeInput) -> Result<i64> {
        input.arg.as_int().ok_or_else(|| {
            anyhow!(
                "operand {:?} is not an int (arg: {:?})",
                input.name,
                input.arg
            )
        })
    }

    fn ints_arg(&mut self, input: &NodeInput) -> Result<Vec<i64>> {
        input
            .arg
            .as_ints()
            .map(|v| v.to_vec())
            .ok_or_else(|| anyhow!("operand {:?} is not an int list", input.name))
    }

    #[allow(dead_code)] // retained for the remaining category ports
    fn optional_ints_arg(&mut self, input: Option<&NodeInput>) -> Result<Vec<usize>> {
        let Some(input) = input else {
            return Ok(vec![]);
        };
        if let Some(v) = input.arg.as_ints() {
            return Ok(v.iter().map(|&i| i.max(0) as usize).collect());
        }
        Ok(vec![])
    }

    fn scalar_type_arg(&mut self, input: &NodeInput) -> Result<DType> {
        let code = input
            .arg
            .as_scalar_type()
            .ok_or_else(|| anyhow!("operand {:?} is not a scalar type", input.name))?;
        dtype_of(code)
    }
}

/// Right-aligned PyTorch broadcasting: prepend size-1 dims, then expand.
fn broadcast_to(mut t: GraphTensor, target: &[IntExpr]) -> GraphTensor {
    while t.rank() < target.len() {
        t = t.expand_dim(0, 1usize);
    }
    t.expand(target.to_vec())
}

/// Broadcast a binary pair to a common shape.
fn broadcast_pair(a: GraphTensor, b: GraphTensor) -> (GraphTensor, GraphTensor) {
    let ad = a.dims();
    let bd = b.dims();
    let rank = ad.len().max(bd.len());
    let mut out = Vec::with_capacity(rank);
    for i in 0..rank {
        let ax = ad.len().checked_sub(rank - i).and_then(|j| ad.get(j));
        let bx = bd.len().checked_sub(rank - i).and_then(|j| bd.get(j));
        let dim = match (ax, bx) {
            (Some(x), Some(y)) => {
                if x == y {
                    *x
                } else if x.to_usize() == Some(1) {
                    *y
                } else if y.to_usize() == Some(1) {
                    *x
                } else {
                    panic!("broadcast: incompatible dims {x:?} and {y:?}");
                }
            }
            (Some(x), None) => *x,
            (None, Some(y)) => *y,
            (None, None) => unreachable!(),
        };
        out.push(dim);
    }
    (broadcast_to(a, &out), broadcast_to(b, &out))
}

/// Float storage dtypes (the sources of a truncating cast).
fn is_float(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::F32 | DType::F64 | DType::F16 | DType::Bf16 | DType::TF32
    )
}

/// Integer storage dtypes the truncating cast may target.
fn is_int(dtype: DType) -> bool {
    matches!(dtype, DType::Int | DType::I64)
}

fn normalize_axes(axes: &[i64], rank: usize) -> Result<Vec<usize>> {
    axes.iter()
        .map(|&a| {
            let a = if a < 0 { a + rank as i64 } else { a };
            usize::try_from(a)
                .ok()
                .filter(|a| *a < rank)
                .ok_or_else(|| anyhow!("axis {a} out of range for rank {rank}"))
        })
        .collect()
}

/// Pull the symbol name out of an `expr_str` like
/// `Symbol('s77', positive=True, integer=True)`.
fn symbol_of(expr: &ExprValue) -> Option<String> {
    let s = &expr.expr_str;
    let start = s.find("Symbol(")? + 7;
    let rest = s.get(start..)?;
    let quote = rest.chars().next()?;
    if quote != '\'' && quote != '"' {
        return None;
    }
    let rest = &rest[1..];
    Some(rest[..rest.find(quote)?].to_string())
}
