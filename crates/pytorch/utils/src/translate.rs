//! ATen (PT2 `model.json`) -> recorder-frontend translator.
//!
//! SSA values stay SSA; the model says nothing about boundary storage.
//! Which values leave, and through whose storage, is a table here, read
//! from the export's output specs: a functionalized mutation (PT2
//! `user_input_mutation` or `buffer_mutation`) records the mutated graph
//! input's name as the output's `mutation_target`, and the backend binds
//! that output on that input's buffer. One row per output spec, keyed by
//! the export's node: two nodes with equal values (a clone and what it
//! copies) are two rows, and the same node as writeback and user output
//! is one row, returned. The program must be functionalised before it
//! reaches here — an in-place op is refused, since which storage it
//! writes is not something a name can tell.
//!
//! Coverage is honest: an unknown ATen target bails with its name. This is
//! the M4 translator re-attachment, rebuilt against the native recorder.

use std::collections::HashMap;
use std::mem::{Discriminant, discriminant};

use anyhow::{Context, Result, anyhow, bail};
use luminal::prelude::*;

mod array_extra;
mod attention;
mod complex;
mod conv;
mod dim_arith;
mod elementwise_more;
mod expr;
mod grouped_mm;
mod index;
mod movement_more;
#[cfg(test)]
mod opmath_tests;
mod ops;
mod pooling;
mod reductions_more;
mod special;
mod stats_more;
mod sympy;
mod unary;
mod upsample;
mod util;

use ops::ReductionOp;
use reductions_more::AddMmVariant;
use stats_more::DistVariant;

use crate::dtype::TorchDType;
use crate::pt2_parser::{InputKind, ParsedPT2};
use crate::pt2_schema::{DimSize, Node, NodeInput, RangeConstraint, TensorMeta};

/// One bound graph input, in export order.
pub struct TranslatedInput {
    pub graph_name: String,
    /// The original checkpoint name for parameters/buffers (e.g. "0.weight").
    pub parameter_name: Option<String>,
    pub kind: InputKind,
    pub tensor: NodeIndex,
    pub dtype: DType,
    /// Symbolic dims: literal extents and recorder dim symbols. Resolve with
    /// [`Translation::dims`] to recover the concrete shape.
    pub shape: Vec<IntExpr>,
}

/// One graph output, in export order.
pub struct TranslatedOutput {
    pub graph_name: String,
    pub tensor: NodeIndex,
    pub dtype: DType,
    /// Symbolic dims (see [`TranslatedInput::shape`]).
    pub shape: Vec<IntExpr>,
    /// For a mutation output: the graph name of the graph input (user input
    /// or module buffer) it writes into. These are writebacks, not returned
    /// tensors.
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
    /// Recorder dim symbol -> the concrete hint torch exported with it.
    pub dims: HashMap<Symbol, usize>,
    /// PT2 symbol name -> recorder dim symbol, for runtime `set_dim` by name.
    pub symbols: HashMap<String, Symbol>,
}

/// The dtypes a torch op performs its math in: tensor operands meet at
/// `common` (torch's TensorIterator common dtype, which for arithmetic is
/// the declared result dtype) and the math runs at `compute`.
#[derive(Clone, Copy)]
struct OpMath {
    common: DType,
    compute: DType,
}

/// Torch's opmath: half-precision math is performed in F32.
fn opmath_compute(common: DType) -> DType {
    match common {
        DType::F16 | DType::Bf16 => DType::F32,
        dtype => dtype,
    }
}

struct Translator<'a> {
    cx: Graph,
    values: HashMap<String, GraphTensor>,
    /// Input values by graph name.
    input_values: HashMap<String, GraphTensor>,
    /// PT2 symbol name -> recorder dim symbol (dynamic dims).
    symbols: HashMap<String, Symbol>,
    /// PT2 symbol name -> torch's exported range constraint.
    ranges: HashMap<String, RangeConstraint>,
    /// Complex-is-a-virtual-type table: value name -> the two real
    /// components that carry it. HLIR never sees a complex dtype.
    complex_tensors: HashMap<String, complex::ComplexTensor>,
    parsed: &'a ParsedPT2,
    /// Recorder dim symbol -> concrete hint, seeded into the runtime.
    dims: HashMap<Symbol, usize>,
    /// The dtype pattern of the op being translated, when torch performs
    /// that op in opmath; `None` for the ops that round nothing.
    opmath: Option<OpMath>,
    /// One widening per (value, common dtype), so several nodes widening
    /// the same value share one cast.
    widened: HashMap<(NodeIndex, Discriminant<DType>), GraphTensor>,
}

/// Read one dimension expression a caller stated in sympy's `srepr`
/// form — `Integer(4)`, `Symbol('s77')`, `Mul(Integer(4), Symbol('s77'))`
/// — against this translation's PT2 symbols.
///
/// A boundary states its element strides in the same vocabulary the
/// exported program states its shapes in, so a caller whose storage is
/// shaped by a dynamic dimension names that dimension rather than the
/// number one example call happened to have.
pub fn parse_dim_expr(translation: &Translation, expr: &str) -> Result<IntExpr> {
    sympy::parse_sympy_expr(expr, &translation.symbols).ok_or_else(|| {
        anyhow!(
            "{expr:?} is not a dimension expression this program states: it names a \
             symbol the exported program does not declare, or a sympy form the \
             parser does not read"
        )
    })
}

/// Translate a parsed PT2 program into the recorder frontend.
pub fn translate(parsed: &ParsedPT2) -> Result<Translation> {
    let sym_dim_map = parsed.build_sym_dim_map();
    let mut t = Translator {
        cx: Graph::new(),
        values: HashMap::new(),
        input_values: HashMap::new(),
        symbols: sym_dim_map.sym_to_symbol.clone(),
        ranges: sym_dim_map.ranges,
        complex_tensors: HashMap::new(),
        parsed,
        dims: HashMap::new(),
        opmath: None,
        widened: HashMap::new(),
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
    // A `buffer_mutation` output names its target by module FQN; the
    // boundary binds by graph input name.
    let buffer_graph_names: HashMap<String, String> = kinds
        .values()
        .filter_map(|kind| match kind {
            InputKind::Buffer {
                graph_name,
                original_name,
            } => Some((original_name.clone(), graph_name.clone())),
            _ => None,
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
        let torch_dtype = TorchDType::from_code(meta.dtype)
            .map_err(|code| anyhow!("unknown PT2 dtype code {code} for input {name}"))?;
        let mut shape = t.boundary_shape(&meta, &name)?;
        let kind = kinds.get(&name).cloned().unwrap_or(InputKind::UserInput {
            graph_name: name.clone(),
        });
        let parameter_name = match &kind {
            InputKind::Parameter { original_name, .. } => Some(original_name.clone()),
            InputKind::Buffer { original_name, .. } => Some(original_name.clone()),
            InputKind::UserInput { .. } => None,
        };
        // Complex storage is interleaved real/imaginary pairs: carry one
        // real-valued input with a trailing extent-2 axis and split it into
        // the two frontend-only components.
        let (dtype, tensor) = if let Some(component_dtype) = torch_dtype.complex_component_dtype() {
            shape.push(2usize.into());
            let backing = t.cx.named_tensor(&name, shape.as_slice(), component_dtype);
            let value = complex::ComplexTensor::from_interleaved(&mut t.cx, backing, torch_dtype)?;
            t.complex_tensors.insert(name.clone(), value);
            t.values.insert(name.clone(), backing);
            t.input_values.insert(name.clone(), backing);
            (component_dtype, backing)
        } else {
            let dtype = dtype_of(meta.dtype)?;
            let tensor = t.cx.named_tensor(&name, shape.as_slice(), dtype);
            t.values.insert(name.clone(), tensor);
            t.input_values.insert(name.clone(), tensor);
            (dtype, tensor)
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

    // 3. Outputs, in export order: one row per output spec, keyed by the
    //    export's own node name. A mutation spec writes back into its target
    //    input's storage and is not returned; torch emits it before the user
    //    output that returns the same NODE, and that user output IS the
    //    writeback (the caller's tensor), not a second boundary. Two specs
    //    naming different nodes are two rows even when they translate to one
    //    value (a returned clone of a written-back value): the boundary gives
    //    each its own buffer, and the plan writes one and copies.
    let output_specs = &parsed.program.graph_module.signature.output_specs;
    let mut regular: Vec<TranslatedOutput> = Vec::new();
    // Export node name of each mutation row -> index in `regular`.
    let mut writeback_by_name: HashMap<String, usize> = HashMap::new();
    for (position, tref) in parsed.program.graph_module.graph.outputs.iter().enumerate() {
        // Literal user outputs have no runtime storage. The Python boundary
        // restores them from the export signature, retaining their positions.
        if let Some(crate::pt2_schema::OutputSpec::Other(spec)) = output_specs.get(position)
            && let Some(arg) = spec.get("user_output").and_then(|output| output.get("arg"))
            && ["as_none", "as_int", "as_float", "as_bool", "as_string"]
                .iter()
                .any(|key| arg.get(key).is_some())
        {
            continue;
        }
        let name = tref
            .value_name()
            .ok_or_else(|| anyhow!("graph output has no tensor name: {tref:?}"))?
            .to_string();
        let meta = t
            .tensor_meta(&name)
            .with_context(|| format!("output {name} has no tensor metadata"))?
            .clone();
        let torch_dtype = TorchDType::from_code(meta.dtype)
            .map_err(|code| anyhow!("unknown PT2 dtype code {code} for output {name}"))?;
        let dtype = match torch_dtype.complex_component_dtype() {
            Some(component_dtype) => component_dtype,
            None => dtype_of(meta.dtype)?,
        };
        let shape = t.boundary_shape(&meta, &name)?;
        // A complex output is repacked into interleaved `[..., 2]` storage
        // for the boundary; its logical shape stays the unpacked one.
        let value = if let Some(value) = t.complex_tensors.get(&name).copied() {
            value.pack(&mut t.cx)
        } else {
            *t.values
                .get(&name)
                .ok_or_else(|| anyhow!("output {name} was never produced"))?
        };

        let mutation_target = match output_specs.get(position) {
            Some(crate::pt2_schema::OutputSpec::UserInputMutation {
                user_input_mutation,
            }) => Some(user_input_mutation.user_input_name.clone()),
            Some(crate::pt2_schema::OutputSpec::BufferMutation { buffer_mutation }) => {
                let buffer_name = &buffer_mutation.buffer_name;
                let Some(target_name) = buffer_graph_names.get(buffer_name) else {
                    bail!("mutation output {name} targets unknown buffer {buffer_name:?}");
                };
                Some(target_name.clone())
            }
            _ => None,
        };
        // A mutation writes an INPUT's storage: the target must be a graph
        // input for the backend to have a buffer to bind this output on.
        if let Some(target_name) = &mutation_target
            && !t.input_values.contains_key(target_name)
        {
            bail!("mutation output {name} targets unknown input {target_name:?}");
        }
        if let Some(target_name) = &mutation_target {
            let target = t.input_values[target_name];
            if target.dtype != value.dtype {
                bail!(
                    "mutation output {name} changes {target_name:?} from {:?} to {:?}",
                    target.dtype,
                    value.dtype
                );
            }
        }
        // A writeback is not part of the caller's returned pytree.
        let returned = mutation_target.is_none();
        if returned && let Some(&index) = writeback_by_name.get(&name) {
            regular[index].returned = true;
            continue;
        }
        if !returned {
            writeback_by_name.insert(name.clone(), regular.len());
        }
        regular.push(TranslatedOutput {
            graph_name: name,
            tensor: value.id,
            dtype,
            shape,
            mutation_target,
            returned,
        });
    }

    let outputs = regular;

    Ok(Translation {
        graph: t.cx,
        inputs,
        outputs,
        dims: t.dims,
        symbols: t.symbols,
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

    /// Resolve a tensor's dims to recorder expressions: literals stay literal
    /// and symbols (including compound sympy extents) stay symbolic. Every
    /// symbol's exported hint is recorded in `self.dims`, so a caller can pin
    /// the graph (static execution) or drive it dynamically from the bound
    /// values.
    fn boundary_shape(&mut self, meta: &TensorMeta, name: &str) -> Result<Vec<IntExpr>> {
        let mut shape = Vec::with_capacity(meta.sizes.len());
        for size in &meta.sizes {
            let expr = self
                .dim_size_to_expr(size)
                .with_context(|| format!("{name}: cannot resolve a dimension expression"))?;
            if let DimSize::Expr(value) = size
                && let Some(hint) = value.as_expr.hint.as_ref().and_then(|h| h.as_int())
                // A compound extent's hint is the WHOLE expression's hint, so
                // only a bare symbol may seed that symbol's value.
                && value.as_expr.expr_str.trim_start().starts_with("Symbol(")
            {
                let hint = usize::try_from(hint).context("negative dim hint")?;
                for symbol in expr.to_symbols() {
                    self.dims.insert(symbol, hint);
                    self.cx.set_dim(symbol, hint);
                }
            }
            shape.push(expr);
        }
        Ok(shape)
    }

    /// Resolve a node operand: a tensor reference, a scalar literal, or a
    /// symbolic reference (unsupported today).
    fn operand(&mut self, input: &NodeInput) -> Result<GraphTensor> {
        if let Some(name) = input.arg.as_tensor_name() {
            let value = self
                .values
                .get(name)
                .copied()
                .ok_or_else(|| anyhow!("operand {name:?} was never produced"))?;
            return Ok(self.widen(value));
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

    /// A tensor operand as recorded, outside the op's dtype pattern: a
    /// predicate or index the op reads for its meaning, not its arithmetic.
    fn raw_operand(&mut self, input: &NodeInput) -> Result<GraphTensor> {
        let name = input.arg.as_tensor_name().ok_or_else(|| {
            anyhow!(
                "operand {:?} is not a tensor (arg: {:?})",
                input.name,
                input.arg
            )
        })?;
        self.values
            .get(name)
            .copied()
            .ok_or_else(|| anyhow!("operand {name:?} was never produced"))
    }

    fn optional_raw_operand(&mut self, input: &NodeInput) -> Result<Option<GraphTensor>> {
        if input.arg.as_tensor_name().is_some() {
            return Ok(Some(self.raw_operand(input)?));
        }
        Ok(None)
    }

    /// A parameter torch reads at the compute dtype without rounding it to
    /// the operands' common dtype first (a norm's affine weights and running
    /// statistics are not operands of its element iterator).
    fn optional_operand_at_compute(&mut self, input: &NodeInput) -> Result<Option<GraphTensor>> {
        let Some(value) = self.optional_raw_operand(input)? else {
            return Ok(None);
        };
        Ok(Some(match self.opmath {
            Some(opmath) => convert(value, opmath.compute),
            None => value,
        }))
    }

    /// A scalar literal as a rank-0 tensor of `dtype`. A Python float is
    /// read at the dtype the op computes in (F64 keeps the double).
    fn scalar(&mut self, input: &NodeInput, dtype: DType) -> Result<GraphTensor> {
        if let Some(v) = input.arg.as_float() {
            return Ok(match dtype {
                DType::F64 => self.cx.constant_f64(v),
                dtype => self.cx.constant_f32(v as f32).cast(dtype),
            });
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

    /// A tensor operand of an opmath op: rounded to the op's common dtype
    /// first (torch's TensorIterator casts the operands to it), then
    /// widened to the dtype the math is performed in.
    fn widen(&mut self, value: GraphTensor) -> GraphTensor {
        let Some(opmath) = self.opmath else {
            return value;
        };
        if value.dtype == opmath.common && value.dtype == opmath.compute {
            return value;
        }
        let key = (value.id, discriminant(&opmath.common));
        if let Some(widened) = self.widened.get(&key) {
            return *widened;
        }
        let widened = convert(convert(value, opmath.common), opmath.compute);
        self.widened.insert(key, widened);
        widened
    }

    /// The dtype pattern this node's op performs: torch's common dtype is
    /// the declared result dtype, except for a comparison, whose Bool
    /// result says nothing about where the operands met.
    fn opmath_for(&self, node: &Node) -> Result<OpMath> {
        let declared = self.first_output_dtype(node)?;
        let common = match declared {
            DType::Bool => {
                let promoted = self.promoted_operand_dtype(node).unwrap_or(declared);
                let float_literal = node
                    .inputs
                    .iter()
                    .any(|input| input.arg.as_float().is_some());
                // A Python float against integer operands compares at
                // torch's default float dtype.
                if float_literal && !is_float(promoted) {
                    DType::F32
                } else {
                    promoted
                }
            }
            dtype => dtype,
        };
        Ok(OpMath {
            common,
            compute: opmath_compute(common),
        })
    }

    /// The dtype the export declares for the node's first tensor output
    /// (tuple outputs included).
    fn first_output_dtype(&self, node: &Node) -> Result<DType> {
        let name = Self::tensor_output_names(node)
            .into_iter()
            .find(|name| !name.is_empty())
            .ok_or_else(|| anyhow!("`{}` has no named tensor output", node.target))?;
        let meta = self
            .tensor_meta(&name)
            .with_context(|| format!("missing tensor meta for output {name}"))?;
        dtype_of(meta.dtype)
    }

    /// The torch lattice over the node's tensor operands. A rank-0 operand
    /// takes part only when its category (bool < int < float) is higher
    /// than every dimensioned operand's.
    fn promoted_operand_dtype(&self, node: &Node) -> Option<DType> {
        let mut dimensioned: Option<DType> = None;
        let mut zero_dim: Option<DType> = None;
        for input in &node.inputs {
            let Some(name) = input.arg.as_tensor_name() else {
                continue;
            };
            let Some(value) = self.values.get(name) else {
                continue;
            };
            let slot = if value.rank() == 0 {
                &mut zero_dim
            } else {
                &mut dimensioned
            };
            *slot = Some(match *slot {
                Some(dtype) => util::promote(dtype, value.dtype),
                None => value.dtype,
            });
        }
        match (dimensioned, zero_dim) {
            (Some(wide), Some(scalar)) if dtype_category(scalar) > dtype_category(wide) => {
                Some(util::promote(wide, scalar))
            }
            (Some(wide), _) => Some(wide),
            (None, scalar) => scalar,
        }
    }

    /// The dtype an arm computes in: the op's opmath dtype when it has
    /// one, else the dtype the export declares for the result.
    fn compute_dtype(&self, node: &Node) -> Result<DType> {
        match self.opmath {
            Some(opmath) => Ok(opmath.compute),
            None => self.output_meta_dtype(node),
        }
    }

    /// Bind one output value under its export name. An opmath result is
    /// rounded here — once, at the store, to the dtype the export declares
    /// for THIS output (`native_layer_norm`'s mean/rstd stay F32).
    fn bind_value(&mut self, name: String, value: GraphTensor) {
        let value = match self.opmath {
            Some(_) => {
                let declared = self
                    .tensor_meta(&name)
                    .ok()
                    .map(|meta| meta.dtype)
                    .and_then(|code| dtype_of(code).ok());
                match declared {
                    Some(dtype) => convert(value, dtype),
                    None => value,
                }
            }
            None => value,
        };
        self.values.insert(name, value);
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
            self.bind_value(name, value);
        }
        Ok(())
    }

    fn dispatch(&mut self, node: &Node) -> Result<()> {
        let target = node
            .target
            .strip_prefix("torch.ops.aten.")
            .or_else(|| node.target.strip_prefix("torch.ops."))
            .unwrap_or(&node.target);

        // A functionalised program has no in-place ops. One here means the
        // export was not functionalised, and which storage it writes is not
        // something a name can tell (a view of an input is not an input):
        // refused, never guessed. Dunder ops (`__and__`) end in `__`, not
        // in the mutation marker.
        let base = target.split('.').next().unwrap_or(target);
        if base.ends_with('_') && !base.ends_with("__") {
            bail!(
                "in-place ATen op `{}` in a non-functionalised program: export it with \
                 run_decompositions so the mutation is an output spec",
                node.target
            );
        }

        // Assertion nodes carry no dataflow; they never bind outputs.
        // `sym_size` produces a scalar SymInt, not a tensor: its value is
        // carried in `sym_int_values` and resolved where a shape argument
        // needs it (`get_int_exprs_arg`), so it binds nothing here.
        if matches!(
            target,
            "_assert_tensor_metadata.default" | "_assert_scalar.default" | "sym_size.int"
        ) {
            return Ok(());
        }

        // Symbolic integer arithmetic has no tensor dataflow. Its exported
        // expression is consumed by shape arguments and scalar_tensor below.
        if !node.outputs.is_empty()
            && node.outputs.iter().all(|out| out.as_sym_int.is_some())
            && node.outputs.iter().all(|out| {
                self.resolve_sym_int(&out.as_sym_int.as_ref().unwrap().as_name)
                    .is_some()
            })
        {
            return Ok(());
        }

        // Shape comparisons feed export's scalar assertions. They contain no
        // tensor data; Dynamo/export owns their guards at the call boundary.
        if !node.outputs.is_empty() && node.outputs.iter().all(|out| out.as_sym_bool.is_some()) {
            return Ok(());
        }

        // Complex is a frontend virtual type. Route every node that consumes
        // or produces one through algebraic real-component lowerings before
        // the ordinary GraphTensor-only dispatch below.
        let first_output = Self::tensor_output_names(node)
            .into_iter()
            .next()
            .unwrap_or_default();
        if self.node_uses_complex(node, &first_output) {
            return self.translate_complex_node(node, &first_output);
        }

        // The op's dtype pattern is set for the whole arm and cleared on
        // every exit, so its operands widen and its outputs round once.
        if opmath_target(target) {
            self.opmath = Some(self.opmath_for(node)?);
        }
        let result = self.dispatch_target(node, target);
        self.opmath = None;
        result
    }

    fn dispatch_target(&mut self, node: &Node, target: &str) -> Result<()> {
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
            // ---- grouped GEMM (batch 6) ----
            "_grouped_mm.default" | "transformers.grouped_mm_fallback.default" => {
                self.translate_grouped_mm(node)?
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
            "add.Tensor" | "add.Scalar" => self.binary(n, |a, b| a + b)?,
            "sub.Tensor" | "sub.Scalar" => self.binary(n, |a, b| a - b)?,
            "mul.Tensor" | "mul.Scalar" => self.binary(n, |a, b| a * b)?,
            "div.Tensor" | "div.Scalar" => self.binary(n, |a, b| a / b)?,
            "maximum.default" => self.binary(n, |a, b| a.maximum(b))?,
            "minimum.default" => self.binary(n, |a, b| a.minimum(b))?,
            // ---- movement ----
            "t.default" => self.operand(&n[0])?.t(),
            "transpose.int" => {
                let x = self.operand(&n[0])?;
                let rank = x.rank();
                let d0 = util::normalize_dim(self.int_arg(&n[1])?, rank);
                let d1 = util::normalize_dim(self.int_arg(&n[2])?, rank);
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
            // ---- batch 7: elementwise specials, squeeze, triangle ----
            "atan2.default" => self.translate_atan2(node)?,
            "copysign.Tensor" => self.translate_copysign(node, false)?,
            "copysign.Scalar" => self.translate_copysign(node, true)?,
            "fmax.default" => self.translate_fmax_fmin(node, true)?,
            "fmin.default" => self.translate_fmax_fmin(node, false)?,
            "hypot.default" => self.translate_hypot(node)?,
            "gcd.default" => self.translate_gcd(node)?,
            "exp2.default" => self.translate_exp2(node)?,
            "log2.default" => self.translate_log2(node)?,
            "isnan.default" => self.translate_isnan(node)?,
            "leaky_relu.default" => self.translate_leaky_relu(node)?,
            "bitwise_and.Tensor" | "__and__.Tensor" => self.translate_bitwise(node, false)?,
            "bitwise_or.Tensor" | "__or__.Tensor" => self.translate_bitwise(node, true)?,
            "squeeze.default" => self.translate_squeeze(node, true)?,
            "squeeze.dims" => self.translate_squeeze(node, false)?,
            "tril.default" => self.translate_triangular(node, false)?,
            "triu.default" => self.translate_triangular(node, true)?,
            // ---- batch 8: reductions, linalg, copies, constructors ----
            "any.default" | "any.dim" | "any.dims" => self.translate_any(node)?,
            "var_mean.default" | "var_mean.dim" | "var_mean.correction" => {
                self.translate_var_mean(node)?;
                return Ok(());
            }
            "addmm.default" => self.translate_addmm(node, AddMmVariant::AddMm)?,
            "addbmm.default" => self.translate_addmm(node, AddMmVariant::AddBmm)?,
            "addmv.default" => self.translate_addmm(node, AddMmVariant::AddMv)?,
            "copy.default" => self.translate_copy(node)?,
            "view_copy.default" => self.translate_view_copy(node)?,
            "permute_copy.default" => self.translate_permute_copy(node)?,
            "empty.memory_format"
            | "empty_permuted.default"
            | "empty_strided.default"
            | "new_empty_strided.default" => self.translate_empty(node)?,
            // ---- batch 9: sampling, search, indices, scatter, renorm ----
            "grid_sampler_2d.default" => self.translate_grid_sampler(node, 2)?,
            "grid_sampler_3d.default" => self.translate_grid_sampler(node, 3)?,
            "searchsorted.Tensor" => self.translate_searchsorted(node, false)?,
            "searchsorted.Scalar" => self.translate_searchsorted(node, true)?,
            "bucketize.Tensor" => self.translate_bucketize(node)?,
            "tril_indices.default" => self.translate_triangular_indices(node, false)?,
            "triu_indices.default" => self.translate_triangular_indices(node, true)?,
            "slice_scatter.default" => self.translate_slice_scatter(node)?,
            "select_scatter.default" => self.translate_select_scatter(node)?,
            "embedding_renorm.default" => self.translate_embedding_renorm(node)?,
            "higher_order.wrap_with_set_grad_enabled" => {
                self.translate_wrap_set_grad(node)?;
                return Ok(());
            }
            // ---- batch 10: distances, norms, histograms, segment reduce ----
            "dist.default" => self.translate_dist(node, DistVariant::Dist)?,
            "_cdist_forward.default" => self.translate_dist(node, DistVariant::Cdist)?,
            "_pdist_forward.default" => self.translate_dist(node, DistVariant::Pdist)?,
            "_trilinear.default" => self.translate_trilinear(node)?,
            "linalg_vector_norm.default" => self.translate_linalg_vector_norm(node)?,
            "histc.default" => self.translate_histc(node)?,
            "histogram.bin_ct" => {
                self.translate_histogram(node, false)?;
                return Ok(());
            }
            "histogram.bins_tensor" => {
                self.translate_histogram(node, true)?;
                return Ok(());
            }
            "_histogramdd_bin_edges.default"
            | "_histogramdd_from_bin_cts.default"
            | "_histogramdd_from_bin_tensors.default" => {
                self.translate_histogramdd(node)?;
                return Ok(());
            }
            "segment_reduce.default" => self.translate_segment_reduce(node)?,
            "fractional_max_pool2d.default" => {
                self.translate_fractional_max_pool(node, 2)?;
                return Ok(());
            }
            "fractional_max_pool3d.default" => {
                self.translate_fractional_max_pool(node, 3)?;
                return Ok(());
            }
            "max_pool2d_with_indices_backward.default" => self.translate_max_pool_backward(node)?,
            // ---- movement batch ----
            "view.default" | "reshape.default" | "_unsafe_view.default" => {
                self.translate_view(node)?
            }
            "flatten.using_ints" | "flatten.default" => self.translate_flatten(node)?,
            "slice.Tensor" => self.translate_slice(node)?,
            "select.int" => self.translate_select(node)?,
            "expand.default" => self.translate_expand(node)?,
            "repeat.default" => self.translate_repeat(node)?,
            // Value identity: a clone, alias or detach has its operand's
            // contents. Which storage each returned one gets is the
            // boundary's statement, read from the export (a clone's own
            // buffer; an alias's or detach's the storage it shares).
            "clone.default" | "alias.default" | "detach.default" => {
                let x = self.operand(&n[0])?;
                util::materialize_tensor(x)
            }
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
            // ---- attention (batch 6) ----
            "scaled_dot_product_attention.default"
            | "_scaled_dot_product_efficient_attention.default"
            | "_scaled_dot_product_flash_attention.default"
            | "_scaled_dot_product_flash_attention_for_cpu.default"
            | "_scaled_dot_product_cudnn_attention.default" => {
                self.translate_sdpa(node)?;
                return Ok(());
            }
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
            // ---- upsample / resize (batch 6) ----
            "upsample_nearest2d.vec" => self.translate_upsample_nearest2d(node)?,
            "upsample_bilinear2d.vec" => self.translate_upsample_bilinear2d(node)?,
            "_upsample_bilinear2d_aa.default" | "_upsample_bilinear2d_aa.vec" => {
                self.translate_upsample_bilinear2d_aa(node)?
            }
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
                // Export omits the schema default dim=0.
                let raw_axis = n
                    .get(1)
                    .map(|arg| self.int_arg(arg))
                    .transpose()?
                    .unwrap_or(0);
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
                let rank = acc.rank();
                let axis = if raw_axis < 0 {
                    raw_axis + rank as i64
                } else {
                    raw_axis
                };
                let axis = usize::try_from(axis)
                    .ok()
                    .filter(|a| *a < rank)
                    .ok_or_else(|| anyhow!("cat: axis {raw_axis} out of range for rank {rank}"))?;
                for next in iter {
                    // An empty concatenation axis contributes no elements. Do
                    // not construct indexing expressions into its empty range.
                    if acc.dims()[axis].to_usize() == Some(0) {
                        acc = next;
                    } else if next.dims()[axis].to_usize() != Some(0) {
                        acc = acc.concat_along(next, axis);
                    }
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
            "index_put.default" => self.translate_index_put(node)?,
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

/// Convert a value to `dtype`: `cast` is lossless-only, so a float -> int
/// conversion is the explicit truncating read, and a float -> narrow-int
/// conversion, which the frontend has no op for, leaves the value alone.
fn convert(value: GraphTensor, dtype: DType) -> GraphTensor {
    if value.dtype == dtype {
        return value;
    }
    if is_float(value.dtype) && is_narrowing_int(dtype) {
        return if is_int(dtype) {
            value.trunc_cast(dtype)
        } else {
            value
        };
    }
    value.cast(dtype)
}

/// Whether torch performs this op in opmath — half-precision math in F32,
/// rounded once at the store. Everything the dispatch table names and this
/// does not is exact: it rounds nothing, so it keeps its operands' dtype
/// (movement, indexing, selection, casts, max/min-family reductions,
/// nearest-neighbour resampling, the operand-dtype scans, and
/// `_grouped_mm`, whose `offs` operand is an int index).
fn opmath_target(target: &str) -> bool {
    // Every `special_*` lowering is a float math function.
    if target.starts_with("special_") {
        return true;
    }
    matches!(
        target,
        // arithmetic
        "add.Tensor"
            | "add.Scalar"
            | "sub.Tensor"
            | "sub.Scalar"
            | "mul.Tensor"
            | "mul.Scalar"
            | "div.Tensor"
            | "div.Scalar"
            | "div.Tensor_mode"
            | "floor_divide.default"
            | "maximum.default"
            | "minimum.default"
            | "fmax.default"
            | "fmin.default"
            | "pow.Tensor_Scalar"
            | "pow.Tensor_Tensor"
            | "pow.Scalar"
            | "fmod.Tensor"
            | "fmod.Scalar"
            | "remainder.Tensor"
            | "remainder.Scalar"
            | "atan2.default"
            | "hypot.default"
            | "copysign.Tensor"
            | "copysign.Scalar"
            | "ldexp.Tensor"
            | "square.default"
            | "reciprocal.default"
            // unary math
            | "sigmoid.default"
            | "tanh.default"
            | "log.default"
            | "log2.default"
            | "log10.default"
            | "log1p.default"
            | "sqrt.default"
            | "rsqrt.default"
            | "exp.default"
            | "exp2.default"
            | "expm1.default"
            | "sin.default"
            | "cos.default"
            | "tan.default"
            | "asin.default"
            | "acos.default"
            | "atan.default"
            | "sinh.default"
            | "cosh.default"
            | "asinh.default"
            | "acosh.default"
            | "atanh.default"
            | "erf.default"
            | "erfc.default"
            | "erfinv.default"
            | "lgamma.default"
            | "digamma.default"
            | "polygamma.default"
            | "i0.default"
            | "silu.default"
            | "gelu.default"
            | "elu.default"
            | "leaky_relu.default"
            | "angle.default"
            // reductions
            | "sum.default"
            | "sum.dim_IntList"
            | "mean.default"
            | "mean.dim"
            | "prod.default"
            | "prod.dim_int"
            | "var.default"
            | "var.dim"
            | "var.correction"
            | "std.default"
            | "std.dim"
            | "std.correction"
            | "var_mean.default"
            | "var_mean.dim"
            | "var_mean.correction"
            | "linalg_vector_norm.default"
            | "logcumsumexp.default"
            // (cumsum/cumprod are exact-class: torch's scans accumulate
            // in the operand dtype.)
            | "dist.default"
            | "_cdist_forward.default"
            | "_pdist_forward.default"
            // softmax and the normalizations
            | "softmax.int"
            | "_softmax.default"
            | "log_softmax.int"
            | "_log_softmax.default"
            | "layer_norm.default"
            | "native_layer_norm.default"
            | "_fused_rms_norm.default"
            | "group_norm.default"
            | "native_group_norm.default"
            | "batch_norm.default"
            | "_native_batch_norm_legit.no_stats"
            | "_native_batch_norm_legit_no_training.default"
            | "_native_batch_norm_legit_functional.default"
            | "_batch_norm_with_update_functional.default"
            // matmul and convolution
            | "mm.default"
            | "bmm.default"
            | "matmul.default"
            | "linear.default"
            | "addmm.default"
            | "addbmm.default"
            | "addmv.default"
            | "_trilinear.default"
            | "conv2d.default"
            | "convolution.default"
            // averaging resamplers
            | "avg_pool2d.default"
            | "avg_pool3d.default"
            | "adaptive_avg_pool2d.default"
            | "adaptive_avg_pool3d.default"
            | "_adaptive_avg_pool2d.default"
            | "_adaptive_avg_pool3d.default"
            | "upsample_bilinear2d.vec"
            | "_upsample_bilinear2d_aa.default"
            | "_upsample_bilinear2d_aa.vec"
            | "grid_sampler_2d.default"
            | "grid_sampler_3d.default"
            // attention
            | "scaled_dot_product_attention.default"
            | "_scaled_dot_product_efficient_attention.default"
            | "_scaled_dot_product_flash_attention.default"
            | "_scaled_dot_product_flash_attention_for_cpu.default"
            | "_scaled_dot_product_cudnn_attention.default"
            // comparisons: torch compares at the promoted operand dtype
            | "eq.Tensor"
            | "ne.Tensor"
            | "lt.Tensor"
            | "le.Tensor"
            | "gt.Tensor"
            | "ge.Tensor"
            | "eq.Scalar"
            | "ne.Scalar"
            | "lt.Scalar"
            | "le.Scalar"
            | "gt.Scalar"
            | "ge.Scalar"
    )
}

/// torch's promotion categories: bool < integer < floating.
fn dtype_category(dtype: DType) -> u8 {
    match dtype {
        DType::Bool => 0,
        dtype if is_float(dtype) => 2,
        _ => 1,
    }
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

/// Integer dtypes a float may not be `cast` into (the frontend refuses a
/// lossy read as a cast).
fn is_narrowing_int(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::Int | DType::I64 | DType::I8 | DType::U8 | DType::I16
    )
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

#[cfg(test)]
mod dim_expr_tests {
    use super::*;

    /// A translation that declares only dimension symbols: a dim
    /// expression is read against those and nothing else.
    fn translation(symbols: &[&str]) -> Translation {
        Translation {
            graph: Graph::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            dims: HashMap::new(),
            symbols: symbols
                .iter()
                .map(|name| ((*name).to_string(), Symbol::new(*name)))
                .collect(),
        }
    }

    #[test]
    fn a_literal_extent_reads_as_its_number() {
        let t = translation(&[]);
        assert_eq!(
            parse_dim_expr(&t, "Integer(4)").unwrap(),
            IntExpr::from(4i64)
        );
    }

    /// A boundary stride stated as the program's own dimension: asked of
    /// what it COMPUTES at a dim value, never of how it is spelled.
    #[test]
    fn a_symbolic_extent_reads_as_the_programs_dim() {
        let t = translation(&["s77"]);
        let symbol = Symbol::new("s77");
        let stride = parse_dim_expr(
            &t,
            "Mul(Integer(2), Symbol('s77', positive=True, integer=True))",
        )
        .unwrap();
        assert!(stride.to_symbols().contains(&symbol));
        let dims: DynMap = [(symbol, 5usize)].into_iter().collect();
        assert_eq!(stride.exec(&dims), Some(10));
    }

    #[test]
    fn an_undeclared_symbol_is_refused_by_name() {
        let t = translation(&[]);
        let err = parse_dim_expr(&t, "Symbol('s77')").expect_err("s77 is not declared");
        assert!(format!("{err:#}").contains("Symbol('s77')"), "{err:#}");
    }

    #[test]
    fn an_unreadable_form_is_refused_by_name() {
        let t = translation(&[]);
        let err = parse_dim_expr(&t, "Piecewise(Integer(1))").expect_err("not a dim form");
        assert!(
            format!("{err:#}").contains("Piecewise(Integer(1))"),
            "{err:#}"
        );
    }
}
