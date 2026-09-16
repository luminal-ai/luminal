//! Python bindings for the CUDA-lite PyTorch backend.
//!
//! This is the GPU twin of the `luminal_reference_py` crate: the same
//! translation seam (`luminal_pytorch_utils`), but the runtime behind the
//! pyo3 class is [`luminal_cuda_lite::CudaRuntime`].
//!
//! THE BOUNDARY IS DECLARED AT LOAD. [`bind`] states every boundary tensor
//! once, in the vocabulary of [`luminal_cuda_lite::CudaBindings`]: one buffer
//! id per boundary tensor, the layout the caller recognized for it, and
//! `Placement::External` — the storage is the caller's own live device
//! allocation, never host-staged and never given an arena range. Aliasing has
//! one spelling: a writeback binds on the buffer of the input it mutates.
//! Python addresses buffers, not tensors: `set_device_ptr(buffer, ptr, bytes)`
//! before each execution.
//!
//! The runtime also runs on a borrowed `CUstream` and takes its
//! intermediate-scratch arena from the caller per execution
//! (`use_borrowed_stream`, `arena_bytes`/`set_arena`).

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow, bail, ensure};
use luminal::layout_ir::{Access, FreedBy};
use luminal::prelude::{DType, DimBucket, DynMap, IntExpr, NodeIndex, Symbol};

/// Largest value a dynamic dimension's bucket covers (the searched plan stays
/// symbolic inside it, so one compile serves every covered context length).
const MAX_DYNAMIC_DIM: usize = 4096;
use luminal_cuda_lite::bindings::{BoundaryLayout, CudaBindings};
use luminal_cuda_lite::{CompileOptions, CudaRuntime, HostBuffer, harness_search_options};
use luminal_pytorch_utils::translate::parse_dim_expr;
use luminal_pytorch_utils::{InputKind, TorchDType, Translation, translate};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

fn to_py(err: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{err:#}"))
}

fn kind_name(kind: &InputKind) -> &'static str {
    match kind {
        InputKind::Parameter { .. } => "parameter",
        InputKind::Buffer { .. } => "buffer",
        InputKind::UserInput { .. } => "user_input",
    }
}

fn torch_code(dtype: DType) -> Result<u32> {
    Ok(TorchDType::try_from(dtype)
        .map_err(|d| anyhow!("no torch dtype for {d:?}"))?
        .code())
}

/// A compiled CUDA-lite graph with its boundary tables.
#[pyclass(unsendable)]
pub struct CompiledGraph {
    translation: Translation,
    runtime: CudaRuntime,
    /// The buffer each translation input was bound on, by input index.
    input_buffers: Vec<i64>,
    /// The buffer each translation output was bound on, by output index. A
    /// writeback repeats the buffer of the input it mutates.
    output_buffers: Vec<i64>,
    searched: bool,
    /// Current concrete value of every symbolic dim, seeded from the exported
    /// hints and updated from real input shapes as they are bound.
    dims: DynMap,
}

/// Resolve a symbolic recorder shape to concrete extents. Literals and
/// hint-seeded symbols resolve immediately; a symbol with no value yet is a
/// programming error (it should have been seeded at translate time).
fn resolve_shape(shape: &[IntExpr], dims: &DynMap) -> Vec<usize> {
    shape
        .iter()
        .map(|dim| {
            dim.exec(dims)
                .or_else(|| dim.to_usize())
                .unwrap_or_else(|| panic!("shape dim {dim:?} has no bound value"))
        })
        .collect()
}

#[pymethods]
impl CompiledGraph {
    #[getter]
    fn input_names(&self) -> Vec<String> {
        self.translation
            .inputs
            .iter()
            .map(|input| input.graph_name.clone())
            .collect()
    }

    #[getter]
    fn input_kinds(&self) -> Vec<String> {
        self.translation
            .inputs
            .iter()
            .map(|input| kind_name(&input.kind).to_string())
            .collect()
    }

    #[getter]
    fn parameter_names(&self) -> Vec<Option<String>> {
        self.translation
            .inputs
            .iter()
            .map(|input| input.parameter_name.clone())
            .collect()
    }

    #[getter]
    fn input_dtypes(&self) -> PyResult<Vec<u32>> {
        self.translation
            .inputs
            .iter()
            .map(|input| torch_code(input.dtype).map_err(to_py))
            .collect()
    }

    #[getter]
    fn input_shapes(&self) -> Vec<Vec<usize>> {
        self.translation
            .inputs
            .iter()
            .map(|input| resolve_shape(&input.shape, &self.dims))
            .collect()
    }

    /// The buffer id each input was bound on, aligned with `input_names`.
    #[getter]
    fn input_buffers(&self) -> Vec<i64> {
        self.input_buffers.clone()
    }

    #[getter]
    fn output_names(&self) -> Vec<String> {
        self.translation
            .outputs
            .iter()
            .map(|output| output.graph_name.clone())
            .collect()
    }

    #[getter]
    fn output_dtypes(&self) -> PyResult<Vec<u32>> {
        self.translation
            .outputs
            .iter()
            .map(|output| torch_code(output.dtype).map_err(to_py))
            .collect()
    }

    #[getter]
    fn output_shapes(&self) -> Vec<Vec<usize>> {
        self.translation
            .outputs
            .iter()
            .map(|output| resolve_shape(&output.shape, &self.dims))
            .collect()
    }

    #[getter]
    fn output_mutations(&self) -> Vec<Option<String>> {
        self.translation
            .outputs
            .iter()
            .map(|output| output.mutation_target.clone())
            .collect()
    }

    #[getter]
    fn output_returns(&self) -> Vec<bool> {
        self.translation
            .outputs
            .iter()
            .map(|output| output.returned)
            .collect()
    }

    /// The buffer id each output was bound on, aligned with `output_names`.
    /// A writeback repeats the buffer of the input it mutates.
    #[getter]
    fn output_buffers(&self) -> Vec<i64> {
        self.output_buffers.clone()
    }

    /// Record one input's concrete shape by graph name. Its axes bind the
    /// graph's symbolic dims, so a symbolic input runs at a new extent
    /// without re-exporting. Dims only: no payload crosses here.
    fn bind_input_shape(&mut self, name: &str, shape: Vec<usize>) -> PyResult<()> {
        self.bind_dims(name, &shape)
    }

    /// Address one EXTERNAL buffer for the next execution: the caller's
    /// allocation at `ptr` IS the storage every binding on that buffer names.
    /// One pointer per buffer — a writeback and the input it mutates share the
    /// buffer and the pointer.
    ///
    /// The caller owns `ptr` and must keep the allocation live, at the
    /// buffer's bound layout, until `execute` returns.
    fn set_device_ptr(&mut self, buffer: i64, ptr: u64, bytes: usize) -> PyResult<()> {
        // SAFETY: upheld by the Python layer, which binds the pointer of a
        // tensor it holds for the duration of the call.
        unsafe { self.runtime.set_device_ptr(buffer, ptr, bytes) }.map_err(to_py)
    }

    /// Forget a buffer's address. The next `execute` refuses by name until one
    /// is supplied again.
    fn clear_device_ptr(&mut self, buffer: i64) {
        self.runtime.clear_device_ptr(buffer);
    }

    /// Bytes of intermediate-scratch arena the selected plan set needs for one
    /// execution. The Python layer allocates exactly this from PyTorch's
    /// caching allocator, passes it to [`Self::set_arena`], and frees it after.
    #[cfg(feature = "device")]
    fn arena_bytes(&self) -> PyResult<usize> {
        self.runtime.arena_bytes().map_err(to_py)
    }

    /// Bind the per-execution arena (a device address from PyTorch's caching
    /// allocator). Never freed by the runtime.
    #[cfg(feature = "device")]
    fn set_arena(&mut self, ptr: u64, bytes: usize) {
        self.runtime.set_arena(ptr, bytes);
    }

    #[cfg(feature = "device")]
    fn clear_arena(&mut self) {
        self.runtime.clear_arena();
    }

    /// Run on PyTorch's current stream (`torch.cuda.current_stream().cuda_stream`).
    #[cfg(feature = "device")]
    fn use_borrowed_stream(&mut self, raw_stream: u64) {
        self.runtime.use_borrowed_stream(raw_stream);
    }

    #[cfg(feature = "device")]
    fn use_owned_stream(&mut self) {
        self.runtime.use_owned_stream();
    }

    /// Override a dynamic dimension's value before `search`, by PT2 symbol
    /// name (e.g. `"s77"`). The value becomes the dim's bucket
    /// representative, so it steers the searched plan without narrowing the
    /// bucket. Hints are seeded at compile time, so static graphs need no
    /// call.
    fn set_dim(&mut self, name: &str, value: usize) -> PyResult<()> {
        if self.searched {
            return Err(PyRuntimeError::new_err(
                "set_dim must be called before search()",
            ));
        }
        let symbol = self
            .translation
            .symbols
            .get(name)
            .copied()
            .ok_or_else(|| PyRuntimeError::new_err(format!("unknown dim symbol {name:?}")))?;
        // The bucket binding owns the runtime's dims until `search` runs;
        // recording the value here is what reaches it.
        self.dims.insert(symbol, value);
        Ok(())
    }

    /// The PT2 symbol name of every dynamic dimension.
    #[getter]
    fn dim_symbols(&self) -> Vec<String> {
        self.translation.symbols.keys().cloned().collect()
    }

    /// The exported hint for each dynamic dimension, aligned with
    /// `dim_symbols`.
    #[getter]
    fn dim_hints(&self) -> Vec<usize> {
        self.translation
            .symbols
            .values()
            .map(|symbol| self.translation.dims.get(symbol).copied().unwrap_or(0))
            .collect()
    }

    /// Saturate and search.
    #[pyo3(signature = (generations = None))]
    fn search(&mut self, generations: Option<usize>) -> PyResult<()> {
        // NO PAYLOAD CROSSES HERE. The default evaluator ranks candidates by
        // the device-free heuristic, which runs nothing; only a
        // device-profiling search consumes boundary bytes, and this backend's
        // boundary is the caller's device memory, never host bytes to copy.
        let data: FxHashMap<NodeIndex, HostBuffer> = FxHashMap::default();
        let mut options: CompileOptions = harness_search_options();
        if let Some(generations) = generations {
            options.generations = generations;
        }
        options.search_log = false;
        if !self.dims.is_empty() {
            // Dynamic program: bind one bucket per symbolic dim and search it
            // ONCE. The winning plan keeps symbolic spans, so every later call
            // whose dims fall in the bucket re-renders without re-searching.
            let hints: Vec<(Symbol, usize)> = self.dims.iter().map(|(s, v)| (*s, *v)).collect();
            for (symbol, hint) in hints {
                let representative = hint.clamp(1, MAX_DYNAMIC_DIM);
                let bucket = DimBucket::new(1, MAX_DYNAMIC_DIM).representative(representative);
                self.runtime
                    .bind_dim_buckets(symbol, vec![bucket])
                    .map_err(to_py)?;
            }
        }
        self.runtime.search(&data, &options).map_err(to_py)?;
        self.searched = true;
        Ok(())
    }

    fn execute(&mut self) -> PyResult<()> {
        if !self.searched {
            return Err(PyRuntimeError::new_err(
                "search() must run before execute()",
            ));
        }
        self.runtime.execute().map_err(to_py)
    }
}

impl CompiledGraph {
    /// Record an input's concrete shape into the symbolic-dim map (and the
    /// runtime once searched).
    fn bind_dims(&mut self, name: &str, shape: &[usize]) -> PyResult<()> {
        let bindings: Vec<(usize, Symbol)> = {
            let input = self
                .translation
                .inputs
                .iter()
                .find(|input| input.graph_name == name)
                .ok_or_else(|| PyRuntimeError::new_err(format!("unknown input {name:?}")))?;
            let mut bindings = Vec::new();
            for (axis, dim) in input.shape.iter().enumerate() {
                if let Some(value) = shape.get(axis) {
                    for symbol in dim.to_symbols() {
                        bindings.push((*value, symbol));
                    }
                }
            }
            bindings
        };
        for (value, symbol) in bindings {
            self.dims.insert(symbol, value);
            // Before search the bucket binding owns the dims; setting them now
            // would make `bind_dim_buckets` refuse as "already set".
            if self.searched {
                self.runtime.set_dim(symbol, value);
            }
        }
        Ok(())
    }
}

/// One boundary tensor's layout as the caller spelled it: a tag, plus the
/// element strides a strided layout carries. A stride is a sympy `srepr`
/// expression read against the translated program's own symbols, so a
/// caller whose storage is shaped by a dynamic dimension states that
/// dimension (`Symbol('s77')`) where a static one states a number
/// (`Integer(4)`).
fn layout_of(
    translation: &Translation,
    name: &str,
    tag: &str,
    strides: &[String],
) -> Result<BoundaryLayout> {
    Ok(match tag {
        "row_major" => BoundaryLayout::RowMajor,
        "column_major" => BoundaryLayout::ColumnMajor,
        "strided" => {
            let strides = strides
                .iter()
                .enumerate()
                .map(|(axis, stride)| {
                    parse_dim_expr(translation, stride)
                        .with_context(|| format!("input {name:?}, stride on axis {axis}"))
                })
                .collect::<Result<Vec<_>>>()?;
            BoundaryLayout::Strided { strides }
        }
        other => bail!("input {name:?}: unknown boundary layout {other:?}"),
    })
}

/// The caller's layout table, keyed by graph input name.
fn layout_table(
    translation: &Translation,
    rows: &[(String, String, Vec<String>)],
) -> Result<HashMap<String, BoundaryLayout>> {
    let mut table = HashMap::new();
    for (name, tag, strides) in rows {
        let layout = layout_of(translation, name, tag, strides)?;
        if table.insert(name.clone(), layout).is_some() {
            bail!("input {name:?} was given two boundary layouts");
        }
    }
    Ok(table)
}

/// The CUDA-lite boundary for a translated program: every input is the
/// caller's live device memory, at the layout the caller recognized for it, on
/// its own buffer; every output is a fresh caller-owned device buffer, except
/// a writeback, which binds on the buffer of the input it mutates — two
/// bindings naming one buffer id being the single spelling of aliasing.
/// Returns the bindings and the buffer each input and each output took.
fn bind(
    translation: &Translation,
    layouts: &HashMap<String, BoundaryLayout>,
) -> Result<(CudaBindings, Vec<i64>, Vec<i64>)> {
    for name in layouts.keys() {
        ensure!(
            translation
                .inputs
                .iter()
                .any(|input| &input.graph_name == name),
            "a boundary layout was declared for {name:?}, which is not a graph input"
        );
    }
    let mut bindings = CudaBindings::new();
    let mut input_buffers = Vec::with_capacity(translation.inputs.len());
    for input in &translation.inputs {
        let layout = layouts.get(&input.graph_name).ok_or_else(|| {
            anyhow!(
                "input {:?} has no declared boundary layout",
                input.graph_name
            )
        })?;
        input_buffers.push(bindings.input_external_with(input.tensor, layout.clone()));
    }
    let mut output_buffers = Vec::with_capacity(translation.outputs.len());
    for output in &translation.outputs {
        let buffer = match &output.mutation_target {
            Some(target) => {
                let index = translation
                    .inputs
                    .iter()
                    .position(|input| &input.graph_name == target)
                    .ok_or_else(|| {
                        anyhow!(
                            "output {} mutates {target:?}, which is not a graph input",
                            output.graph_name
                        )
                    })?;
                let layout = layouts[target].clone();
                // A writeback writes the TARGET's storage, so it is bound at
                // the target's layout and never reinterprets it. A kernel
                // destination must be row-major (the only destination layout
                // the kernels write), so any other layout is refused here by
                // name rather than written as though it were row-major.
                ensure!(
                    layout == BoundaryLayout::RowMajor,
                    "output {} writes back into input {target:?}, whose boundary layout is \
                     {layout:?}; a writeback destination must be row-major",
                    output.graph_name
                );
                let buffer = input_buffers[index];
                // The caller's storage is written through: the shared buffer
                // must say so.
                bindings.declare(buffer, Access::ReadWrite, FreedBy::Caller);
                bindings.output_on_with(output.tensor, buffer, layout);
                buffer
            }
            None => bindings.output_external(output.tensor),
        };
        output_buffers.push(buffer);
    }
    Ok((bindings, input_buffers, output_buffers))
}

/// Parse, translate, and load a `.pt2` on the CUDA-lite runtime under the
/// caller's boundary layouts: one `(graph input name, layout tag, element
/// strides)` row per graph input, the tag being `row_major`, `column_major`
/// or `strided`, and each stride a sympy `srepr` expression over the
/// exported program's symbols.
#[pyfunction]
fn compile(
    pt2_path: &str,
    input_layouts: Vec<(String, String, Vec<String>)>,
) -> PyResult<CompiledGraph> {
    let parsed = luminal_pytorch_utils::parse_pt2(pt2_path)
        .with_context(|| format!("parsing {pt2_path}"))
        .map_err(to_py)?;
    let translation = translate(&parsed).map_err(to_py)?;
    let dims: DynMap = translation.dims.iter().map(|(k, v)| (*k, *v)).collect();
    let layouts = layout_table(&translation, &input_layouts).map_err(to_py)?;
    let (bindings, input_buffers, output_buffers) = bind(&translation, &layouts)
        .context("declaring the translated program's boundary")
        .map_err(to_py)?;
    let runtime = CudaRuntime::load_with(
        &translation.graph,
        bindings,
        luminal_cuda_lite::ops::cuda_registry(),
    )
    .context("loading the translated graph on the cuda-lite runtime")
    .map_err(to_py)?;
    Ok(CompiledGraph {
        translation,
        runtime,
        input_buffers,
        output_buffers,
        searched: false,
        dims,
    })
}

#[pymodule]
fn _luminal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<CompiledGraph>()?;
    m.add_function(wrap_pyfunction!(compile, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use luminal::prelude::Graph;
    use luminal_pytorch_utils::{TranslatedInput, TranslatedOutput};

    /// A translation with one 2x3 F32 input per name and one output
    /// value. `mutates` is the graph input that output writes back into.
    /// Only the boundary tables matter here: `bind` reads the input and
    /// output records, never the graph's structure.
    fn translation(names: &[&str], mutates: Option<&str>) -> Translation {
        let mut cx = Graph::new();
        let shape = vec![IntExpr::from(2i64), IntExpr::from(3i64)];
        let inputs = names
            .iter()
            .map(|name| {
                let tensor = cx.named_tensor(*name, (2usize, 3usize), DType::F32);
                TranslatedInput {
                    graph_name: (*name).to_string(),
                    parameter_name: None,
                    kind: InputKind::UserInput {
                        graph_name: (*name).to_string(),
                    },
                    tensor: tensor.id,
                    dtype: DType::F32,
                    shape: shape.clone(),
                }
            })
            .collect();
        let out = cx.named_tensor("out_source", (2usize, 3usize), DType::F32);
        let outputs = vec![TranslatedOutput {
            graph_name: "out".to_string(),
            tensor: out.id,
            dtype: DType::F32,
            shape,
            mutation_target: mutates.map(str::to_string),
            returned: true,
        }];
        Translation {
            graph: cx,
            inputs,
            outputs,
            dims: std::collections::HashMap::new(),
            symbols: std::collections::HashMap::new(),
        }
    }

    fn row_major(names: &[&str]) -> HashMap<String, BoundaryLayout> {
        names
            .iter()
            .map(|name| ((*name).to_string(), BoundaryLayout::RowMajor))
            .collect()
    }

    /// ALIASING HAS ONE SPELLING: the writeback and the input it mutates
    /// name one buffer id, so the caller addresses one pointer.
    #[test]
    fn a_writeback_and_its_target_are_one_buffer() {
        let translation = translation(&["x"], Some("x"));
        let (bindings, inputs, outputs) =
            bind(&translation, &row_major(&["x"])).expect("writeback binds");
        assert_eq!(outputs[0], inputs[0]);
        assert_eq!(bindings.buffers()[&inputs[0]].access, Access::ReadWrite);
    }

    #[test]
    fn a_mutation_target_that_is_not_a_graph_input_is_refused() {
        let translation = translation(&["x"], Some("elsewhere"));
        let err = bind(&translation, &row_major(&["x"])).expect_err("no such input");
        assert!(
            format!("{err:#}").contains("mutates \"elsewhere\""),
            "{err:#}"
        );
    }

    #[test]
    fn an_input_with_no_layout_row_is_refused() {
        let translation = translation(&["x", "y"], None);
        let err = bind(&translation, &row_major(&["x"])).expect_err("y has no layout");
        assert!(
            format!("{err:#}").contains("input \"y\" has no declared boundary layout"),
            "{err:#}"
        );
    }

    /// A layout row naming a tensor that is not a graph input is a
    /// statement about nothing: refused rather than dropped.
    #[test]
    fn a_layout_row_for_a_non_input_is_refused() {
        let translation = translation(&["x"], None);
        let err = bind(&translation, &row_major(&["x", "ghost"])).expect_err("no such input");
        assert!(format!("{err:#}").contains("\"ghost\""), "{err:#}");
    }

    #[test]
    fn an_unknown_layout_tag_is_refused() {
        let translation = translation(&["x"], None);
        let rows = vec![("x".to_string(), "diagonal".to_string(), Vec::new())];
        let err = layout_table(&translation, &rows).expect_err("no such layout");
        assert!(
            format!("{err:#}").contains("unknown boundary layout \"diagonal\""),
            "{err:#}"
        );
    }

    /// A kernel writes row-major destinations only, so a writeback into a
    /// target the caller handed over at another layout is refused BY NAME
    /// rather than written as though it were row-major.
    #[test]
    fn a_writeback_into_a_non_row_major_target_is_refused_by_name() {
        let translation = translation(&["x"], Some("x"));
        let layouts: HashMap<String, BoundaryLayout> =
            [("x".to_string(), BoundaryLayout::ColumnMajor)].into();
        let err = bind(&translation, &layouts).expect_err("column-major writeback target");
        let text = format!("{err:#}");
        assert!(text.contains("\"x\""), "{text}");
        assert!(text.contains("must be row-major"), "{text}");
    }

    /// A SYMBOLIC STRIDE reaches the binding as the program's own dim: a
    /// strided dynamic view states the dimension it is strided by, not a
    /// number one example call happened to have. Asked of what the stride
    /// COMPUTES at a dim value, never of how it is spelled.
    #[test]
    fn a_symbolic_stride_is_read_against_the_programs_symbols() {
        let symbol = Symbol::new("s77");
        let mut translation = translation(&["x"], None);
        translation.symbols.insert("s77".to_string(), symbol);
        let rows = vec![(
            "x".to_string(),
            "strided".to_string(),
            vec![
                "Integer(1)".to_string(),
                "Mul(Integer(2), Symbol('s77', positive=True, integer=True))".to_string(),
            ],
        )];
        let table = layout_table(&translation, &rows).expect("symbolic strides");
        let BoundaryLayout::Strided { strides } = &table["x"] else {
            panic!("expected a strided layout, got {:?}", table["x"]);
        };
        assert_eq!(strides[0], IntExpr::from(1i64));
        assert!(strides[1].to_symbols().contains(&symbol));
        let dims: DynMap = [(symbol, 5usize)].into_iter().collect();
        assert_eq!(strides[1].exec(&dims), Some(10));
    }

    #[test]
    fn a_stride_naming_an_undeclared_symbol_is_refused() {
        let translation = translation(&["x"], None);
        let rows = vec![(
            "x".to_string(),
            "strided".to_string(),
            vec!["Integer(1)".to_string(), "Symbol('s77')".to_string()],
        )];
        let err = layout_table(&translation, &rows).expect_err("s77 is not declared");
        let text = format!("{err:#}");
        assert!(text.contains("Symbol('s77')"), "{text}");
        assert!(text.contains("axis 1"), "{text}");
    }
}
