//! Running a code object: one call, from torch's argument list to the
//! values torch gets back.
//!
//! Everything a call does happens here, in order: read each argument slot,
//! state the call's dimensions, check them, stage the tensors the program
//! reads, run the plan, and build the outputs. Nothing in this module parses
//! a program or looks at a graph.

use std::ptr;

use anyhow::{Context, Result, anyhow, bail, ensure};
use luminal::prelude::{DType, DynMap, IntExpr, NodeIndex, Symbol};
use luminal_reference::{ReferenceRuntime, TypedBuffer};
use luminal_torch_code::to_py;
use luminal_torch_slots::{Device, DeviceType, TensorInfo, TorchDtype, empty_strided, read_tensor};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::buffers;
use crate::code::Code;

/// One code object, ready to be called. It is handed between threads, and
/// one call uses it at a time.
pub struct Executable {
    code: Code,
    /// `None` for a code object with no plan: nothing is computed.
    runtime: Option<ReferenceRuntime>,
    /// The storage dtype of each input and of each row, resolved once.
    input_dtypes: Vec<DType>,
    row_dtypes: Vec<DType>,
    /// The value every dimension has now: the saved one until a call states
    /// it, then the last call's.
    dims: DynMap,
    /// Whether each call evaluates the conditions torch asserted on the
    /// call's dimensions. Binding does not depend on them.
    pub runtime_assertions: bool,
}

impl Executable {
    pub fn new(code: Code) -> Result<Self> {
        let input_dtypes = code
            .inputs
            .iter()
            .map(|input| {
                buffers::storage_dtype(input.dtype)
                    .with_context(|| format!("input {:?}", input.name))
            })
            .collect::<Result<Vec<_>>>()?;
        let row_dtypes = code
            .rows
            .iter()
            .map(|row| {
                buffers::storage_dtype(row.dtype).with_context(|| format!("output {:?}", row.name))
            })
            .collect::<Result<Vec<_>>>()?;
        let dims = code.boundary.state();
        let runtime = match &code.plan {
            None => {
                ensure!(
                    code.rows.is_empty(),
                    "a code object with no plan states {} output rows",
                    code.rows.len()
                );
                None
            }
            Some(plan) => {
                let saved = serde_json::to_vec(plan).context("reading the saved plan")?;
                let mut runtime = ReferenceRuntime::default();
                // The runtime keys its boundary by an opaque id per tensor.
                // A code object has no graph, so positions stand in for them.
                let inputs: Vec<NodeIndex> = (0..code.inputs.len()).map(NodeIndex::new).collect();
                let outputs: Vec<NodeIndex> = (0..code.rows.len())
                    .map(|row| NodeIndex::new(code.inputs.len() + row))
                    .collect();
                let buffers = runtime
                    .deserialize_compiled(&saved, &inputs, &outputs, code.memory_budget_bytes)
                    .context("installing the saved plan")?;
                // The plan's own boundary must be the one the tables state.
                let stated: Vec<i64> = code.rows.iter().map(|row| row.buffer).collect();
                ensure!(
                    buffers == stated,
                    "the saved plan writes its outputs on buffers {buffers:?}, and the code \
                     object's rows state {stated:?}"
                );
                for (input, dtype) in code.inputs.iter().zip(&input_dtypes) {
                    if let Some(elements) = &input.constant {
                        let buffer = buffers::typed_buffer(*dtype, elements)
                            .with_context(|| format!("constant {:?}", input.name))?;
                        runtime.set_data_buffer(input.buffer, buffer);
                    }
                }
                for (symbol, value) in &dims {
                    runtime.set_dim(*symbol, *value);
                }
                Some(runtime)
            }
        };
        Ok(Self {
            code,
            runtime,
            input_dtypes,
            row_dtypes,
            dims,
            runtime_assertions: true,
        })
    }

    pub fn code(&self) -> &Code {
        &self.code
    }

    /// Run one call. `args` is torch's argument list, one value per slot;
    /// the result has one value per output leaf, in torch's order.
    pub fn call<'py>(&mut self, args: &Bound<'py, PyList>) -> PyResult<Bound<'py, PyList>> {
        let py = args.py();
        let Self {
            code,
            runtime,
            input_dtypes,
            row_dtypes,
            dims,
            runtime_assertions,
        } = self;
        // Read every slot, then stage each tensor the program reads: its
        // shape states dimensions, and its elements are copied to the host.
        let arguments = code
            .boundary
            .read_arguments(args, "luminal_torch_reference_runtime")?;
        let mut staged: Vec<(usize, TypedBuffer)> = Vec::new();
        let mut from_tensors: Vec<(Symbol, usize)> = Vec::new();
        for argument in &arguments.tensors {
            let name = &code.boundary.slot_names[argument.slot];
            let declared = &code.inputs[argument.input];
            from_tensors.extend(
                stated_dimensions(&declared.shape, &argument.info.shape)
                    .with_context(|| format!("argument '{name}'"))
                    .map_err(to_py)?,
            );
            let buffer =
                buffers::typed_buffer(input_dtypes[argument.input], &elements(&argument.info))
                    .with_context(|| format!("argument '{name}'"))
                    .map_err(to_py)?;
            staged.push((argument.input, buffer));
        }
        // An integer argument that is a dimension defines it, as it does for
        // torch, so it is applied after the tensor shapes.
        dims.extend(from_tensors);
        dims.extend(arguments.integers);

        if *runtime_assertions && let Some(check) = code.boundary.failed_check(dims) {
            return Err(PyRuntimeError::new_err(check.message.clone()));
        }

        // Run the plan and read each row it wrote.
        let mut rows: Vec<Bound<'py, PyAny>> = Vec::with_capacity(code.rows.len());
        if let Some(runtime) = runtime {
            for (input, buffer) in staged {
                runtime.set_data_buffer(code.inputs[input].buffer, buffer);
            }
            for (symbol, value) in dims.iter() {
                runtime.set_dim(*symbol, *value);
            }
            // The interpreter is released while the plan computes, as torch
            // releases it around its own kernels and Inductor around its
            // compiled code: other Python threads run meanwhile. This call
            // still holds the executable, so a second call to it waits.
            py.detach(|| {
                luminal_reference::search::with_interrupt_check(
                    || Python::attach(|py| py.check_signals().map_err(anyhow::Error::from)),
                    || runtime.execute(),
                )
            })
            .map_err(to_py)?;
            for (row, dtype) in code.rows.iter().zip(row_dtypes.iter()) {
                let shape = resolve(&row.shape, dims)
                    .with_context(|| format!("output {:?}", row.name))
                    .map_err(to_py)?;
                let buffer = runtime.get_buffer(row.buffer).map_err(to_py)?;
                let elements = buffers::buffer_bytes(*dtype, buffer).map_err(to_py)?;
                let tensor = new_tensor(py, &row.name, row.dtype, &shape, elements)?;
                rows.push(match row.writes_back {
                    // A mutation writes the caller's own tensor, and that
                    // tensor is the row's value: the same storage eager
                    // would have written.
                    Some(slot) => {
                        let destination = args.get_item(slot)?;
                        destination.call_method1("copy_", (tensor,))?;
                        destination
                    }
                    None => tensor,
                });
            }
        }

        code.boundary.outputs(args, &rows)
    }
}

/// Solve a declared extent `a * symbol + b = value` exactly. Never bind
/// every symbol to the whole extent: the size of `3*s` is not `s`.
pub fn bind_extent(expr: &IntExpr, value: usize) -> Result<Option<(Symbol, usize)>> {
    use luminal::shape::Term;
    let mut symbols = expr.to_symbols();
    symbols.sort();
    symbols.dedup();
    if symbols.is_empty() {
        ensure!(
            expr.to_usize() == Some(value),
            "input extent {value} does not match {expr:?}"
        );
        return Ok(None);
    }
    ensure!(
        symbols.len() == 1,
        "cannot bind multivariate input extent {expr:?}"
    );
    let mut stack: Vec<(i128, i128)> = Vec::new();
    for term in expr.terms.read().iter() {
        match term {
            Term::Num(n) => stack.push((0, i128::from(*n))),
            Term::Var(_) => stack.push((1, 0)),
            op => {
                let (a, b) = stack.pop().ok_or_else(|| anyhow!("malformed extent"))?;
                let (c, d) = stack.pop().ok_or_else(|| anyhow!("malformed extent"))?;
                let pair = match op {
                    Term::Add => a.checked_add(c).zip(b.checked_add(d)),
                    Term::Sub => a.checked_sub(c).zip(b.checked_sub(d)),
                    Term::Mul if c == 0 => a.checked_mul(d).zip(b.checked_mul(d)),
                    Term::Mul if a == 0 => c.checked_mul(b).zip(d.checked_mul(b)),
                    Term::Div if c == 0 && d != 0 && a % d == 0 && b % d == 0 => {
                        Some((a / d, b / d))
                    }
                    _ => None,
                }
                .ok_or_else(|| {
                    anyhow!("input extent {expr:?} is not a supported affine dimension")
                })?;
                stack.push(pair);
            }
        }
    }
    let (scale, offset) = stack.pop().ok_or_else(|| anyhow!("empty extent"))?;
    let numerator = value as i128 - offset;
    ensure!(
        scale > 0 && numerator >= 0 && numerator % scale == 0,
        "input extent {value} violates {expr:?}"
    );
    let root = usize::try_from(numerator / scale)?;
    Ok(Some((symbols[0], root)))
}

/// The dimensions one tensor's shape states, read against its declared
/// extents. Two axes that state one dimension must agree.
fn stated_dimensions(declared: &[IntExpr], shape: &[i64]) -> Result<Vec<(Symbol, usize)>> {
    ensure!(
        declared.len() == shape.len(),
        "was compiled at rank {} and called at rank {} (shape {shape:?})",
        declared.len(),
        shape.len()
    );
    let mut stated: Vec<(Symbol, usize)> = Vec::new();
    for (extent, size) in declared.iter().zip(shape) {
        if let Some((symbol, root)) = bind_extent(extent, *size as usize)? {
            if let Some((_, first)) = stated.iter().find(|(known, _)| *known == symbol)
                && *first != root
            {
                bail!(
                    "axes sharing dimension {symbol} must have equal sizes, got {first} and \
                     {root}"
                );
            }
            stated.push((symbol, root));
        }
    }
    Ok(stated)
}

/// Each extent at the call's dimensions.
fn resolve(shape: &[IntExpr], dims: &DynMap) -> Result<Vec<i64>> {
    shape
        .iter()
        .map(|extent| {
            extent
                .exec(dims)
                .or_else(|| extent.to_usize())
                .map(|size| size as i64)
                .ok_or_else(|| anyhow!("extent {extent} names a dimension no call has stated"))
        })
        .collect()
}

/// torch's contiguous strides for a shape: the running product multiplies by
/// `max(size, 1)`.
fn contiguous_strides(shape: &[i64]) -> Vec<i64> {
    let mut strides = vec![1i64; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1].max(1);
    }
    strides
}

/// A tensor's elements, row-major, whatever its strides: what
/// `tensor.contiguous()` holds.
fn elements(info: &TensorInfo) -> Vec<u8> {
    let width = info.dtype.element_size();
    let count: usize = info.shape.iter().map(|size| *size as usize).product();
    let mut bytes = Vec::with_capacity(count * width);
    if count == 0 {
        return bytes;
    }
    let base = info.data as *const u8;
    if info.strides == contiguous_strides(&info.shape) {
        // SAFETY: a contiguous tensor of `count` elements owns `count *
        // width` bytes from its first element, and the caller's argument
        // keeps it alive for the call.
        bytes.extend_from_slice(unsafe { std::slice::from_raw_parts(base, count * width) });
        return bytes;
    }
    let rank = info.shape.len();
    let mut index = vec![0i64; rank];
    loop {
        let offset: i64 = index
            .iter()
            .zip(&info.strides)
            .map(|(position, stride)| position * stride)
            .sum();
        // SAFETY: every index is inside the tensor's shape, so the element
        // the strides reach is one the tensor addresses.
        bytes.extend_from_slice(unsafe {
            std::slice::from_raw_parts(base.offset(offset as isize * width as isize), width)
        });
        // Advance the index, last axis fastest.
        let mut axis = rank;
        loop {
            if axis == 0 {
                return bytes;
            }
            axis -= 1;
            index[axis] += 1;
            if index[axis] < info.shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
}

/// A fresh CPU tensor torch owns, holding a row's elements.
fn new_tensor<'py>(
    py: Python<'py>,
    name: &str,
    dtype: TorchDtype,
    shape: &[i64],
    elements: &[u8],
) -> PyResult<Bound<'py, PyAny>> {
    let count: usize = shape.iter().map(|size| *size as usize).product();
    if elements.len() != count * dtype.element_size() {
        return Err(PyRuntimeError::new_err(format!(
            "output {name:?}: the runtime wrote {} bytes for shape {shape:?} of torch.{}",
            elements.len(),
            dtype.name()
        )));
    }
    let tensor = empty_strided(
        py,
        shape,
        &contiguous_strides(shape),
        dtype,
        Device {
            kind: DeviceType::Cpu,
            index: -1,
        },
    )?;
    if !elements.is_empty() {
        let info = read_tensor(&tensor)?;
        // SAFETY: the tensor was just allocated contiguous with exactly
        // `elements.len()` bytes, and nothing else refers to it yet.
        unsafe {
            ptr::copy_nonoverlapping(elements.as_ptr(), info.data as *mut u8, elements.len())
        };
    }
    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(data: &[f32], shape: &[i64], strides: &[i64]) -> TensorInfo {
        TensorInfo {
            data: data.as_ptr() as usize,
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            dtype: TorchDtype::Float32,
            device: Device {
                kind: DeviceType::Cpu,
                index: -1,
            },
            storage_offset: 0,
        }
    }

    fn floats(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    /// Elements are read row-major whatever the strides: a transposed view,
    /// an every-other-element view and an expanded one all read as their
    /// contiguous copy would.
    #[test]
    fn elements_are_read_row_major_whatever_the_strides() {
        let storage: Vec<f32> = (0..6).map(|value| value as f32).collect();
        assert_eq!(
            floats(&elements(&info(&storage, &[2, 3], &[3, 1]))),
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        );
        // The transpose of a 2x3: shape (3, 2), strides (1, 3).
        assert_eq!(
            floats(&elements(&info(&storage, &[3, 2], &[1, 3]))),
            [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
        );
        assert_eq!(
            floats(&elements(&info(&storage, &[3], &[2]))),
            [0.0, 2.0, 4.0]
        );
        // One row expanded to three: stride zero on the expanded axis.
        assert_eq!(
            floats(&elements(&info(&storage, &[3, 2], &[0, 1]))),
            [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        );
        assert_eq!(floats(&elements(&info(&storage, &[], &[]))), [0.0]);
        assert!(elements(&info(&storage, &[0, 3], &[3, 1])).is_empty());
    }

    /// A declared extent is solved for its dimension, never assumed to be
    /// it, and two axes that state one dimension must agree.
    #[test]
    fn a_shape_states_dimensions_through_its_declared_extents() {
        let n = Symbol::new("s0");
        let declared = vec![
            IntExpr::from(n) * IntExpr::from(2i64),
            IntExpr::from(4i64),
            IntExpr::from(n),
        ];
        assert_eq!(
            stated_dimensions(&declared, &[6, 4, 3]).unwrap(),
            [(n, 3), (n, 3)]
        );
        let error = stated_dimensions(&declared, &[6, 4, 5]).unwrap_err();
        assert!(error.to_string().contains("equal sizes"), "{error}");
        let error = stated_dimensions(&declared, &[7, 4, 3]).unwrap_err();
        assert!(error.to_string().contains("violates"), "{error}");
        let error = stated_dimensions(&declared, &[6, 5, 3]).unwrap_err();
        assert!(error.to_string().contains("does not match"), "{error}");
        let error = stated_dimensions(&declared, &[6, 4]).unwrap_err();
        assert!(error.to_string().contains("rank 3"), "{error}");
    }

    #[test]
    fn contiguous_strides_are_torchs() {
        assert_eq!(contiguous_strides(&[2, 3, 4]), [12, 4, 1]);
        assert_eq!(contiguous_strides(&[2, 0, 3]), [3, 3, 1]);
        assert!(contiguous_strides(&[]).is_empty());
    }
}
