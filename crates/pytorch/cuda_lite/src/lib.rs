//! Python bindings for the CUDA-lite PyTorch backend.
//!
//! This is the GPU twin of the `luminal_reference_py` crate: the same
//! translation seam (`luminal_pytorch_utils`), but the runtime behind the
//! pyo3 class is [`luminal_cuda_lite::CudaRuntime`].
//!
//! THE BOUNDARY IS DECLARED AT LOAD. [`bind`] states every boundary tensor
//! once, in the vocabulary of [`luminal_cuda_lite::CudaBindings`]: one buffer
//! id per boundary tensor, the layout the caller declared for it, and
//! `Placement::External` — the storage is the caller's own live device
//! allocation, never host-staged and never given an arena range. Aliasing has
//! one spelling, the same buffer id: a writeback binds on the buffer of the
//! input it mutates, and an output that shares storage with an earlier
//! boundary tensor (a view of an input, of a writeback, or of another
//! output — read off the export's traced storage) binds on that tensor's
//! buffer at its own layout. A user-visible output is bound at eager's exact
//! strides, so the tensor the caller receives is laid out the way the
//! uncompiled program lays it out.
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
    /// The layout each translation output was bound at, by output index,
    /// spelled the way the caller spelled it. A writeback repeats the row of
    /// the input it mutates.
    output_layouts: Vec<(String, Vec<String>)>,
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

    /// The layout each output was bound at, aligned with `output_names`: the
    /// tag and, for a strided layout, its element strides, in the spelling
    /// the caller declared them in. A writeback repeats the row of the input
    /// it mutates, so the caller reads one table for every output rather than
    /// re-deriving which is which.
    #[getter]
    fn output_layouts(&self) -> Vec<(String, Vec<String>)> {
        self.output_layouts.clone()
    }

    /// The buffer id the installed plan writes this output's bytes into.
    fn output_backing_buffer(&self, name: &str) -> PyResult<i64> {
        let slot = self.output_position(name)?;
        self.runtime.output_slot_backing_buffer(slot).map_err(to_py)
    }

    /// The bytes that backing buffer spans at the dims bound now.
    fn output_span_bytes(&self, name: &str) -> PyResult<usize> {
        let slot = self.output_position(name)?;
        self.runtime.output_slot_span_bytes(slot).map_err(to_py)
    }

    /// The element strides the installed plan elected for this output.
    fn output_elected_strides(&self, name: &str) -> PyResult<Vec<i64>> {
        let slot = self.output_position(name)?;
        self.runtime
            .output_slot_elected_strides(slot)
            .map_err(to_py)
    }

    /// Record the concrete shapes of one call's inputs by graph name.
    /// Their axes bind the graph's symbolic dims, so a symbolic input runs
    /// at a new extent without re-exporting. Dims only: no payload crosses
    /// here.
    ///
    /// ONE CALL IS READ AT ONCE, because a compound extent is checked
    /// against the dimensions THIS call states: read input by input, an
    /// input declared `2*s0` would be checked against whatever s0 was left
    /// at by the call before it.
    fn bind_input_shapes(&mut self, shapes: Vec<(String, Vec<usize>)>) -> PyResult<()> {
        self.bind_dims(&shapes).map_err(to_py)
    }

    /// The same for a single input — the whole call, when it has one
    /// tensor in it.
    fn bind_input_shape(&mut self, name: &str, shape: Vec<usize>) -> PyResult<()> {
        self.bind_dims(&[(name.to_string(), shape)]).map_err(to_py)
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
        self.run_search(generations).map_err(to_py)
    }

    fn execute(&mut self) -> PyResult<()> {
        if !self.searched {
            return Err(PyRuntimeError::new_err(
                "search() must run before execute()",
            ));
        }
        self.runtime.execute().map_err(to_py)
    }

    /// The runtime's cumulative counters, or None before it touched a device.
    fn graph_stats(&self) -> Option<HashMap<String, u64>> {
        #[cfg(feature = "device")]
        {
            self.runtime.graph_stats().map(|stats| {
                HashMap::from([
                    ("launches".to_string(), stats.launches),
                    ("instantiations".to_string(), stats.instantiations),
                    ("graph_cache_hits".to_string(), stats.graph_cache_hits),
                    ("host_captures".to_string(), stats.host_captures),
                    ("host_cache_hits".to_string(), stats.host_cache_hits),
                    ("node_updates".to_string(), stats.node_updates),
                    ("address_rebinds".to_string(), stats.address_rebinds),
                    ("kernel_compilations".to_string(), stats.kernel_compilations),
                    ("arena_generation".to_string(), stats.arena_generation),
                    ("arena_base".to_string(), stats.arena_base),
                    ("arena_bytes".to_string(), stats.arena_bytes as u64),
                    ("staging_bytes".to_string(), stats.staging_bytes as u64),
                    (
                        "resident_upload_bytes".to_string(),
                        stats.resident_upload_bytes,
                    ),
                ])
            })
        }
        #[cfg(not(feature = "device"))]
        {
            None
        }
    }
}

impl CompiledGraph {
    /// The graph value one output name names.
    /// A translation output's position, which is its slot in the bound
    /// program: outputs are bound in translation order, one slot each, so a
    /// value returned under two names has two slots and each name finds its own.
    fn output_position(&self, name: &str) -> PyResult<usize> {
        self.translation
            .outputs
            .iter()
            .position(|output| output.graph_name == name)
            .ok_or_else(|| PyRuntimeError::new_err(format!("{name:?} is not a graph output")))
    }

    /// Saturate and search. What this states is what Python reads: the
    /// `search` method maps it through [`to_py`] unchanged.
    fn run_search(&mut self, generations: Option<usize>) -> Result<()> {
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
                self.runtime.bind_dim_buckets(symbol, vec![bucket])?;
            }
        }
        self.runtime.search(&data, &options)?;
        self.searched = true;
        Ok(())
    }

    /// Record this call's concrete input shapes into the symbolic-dim map
    /// (and the runtime once searched).
    ///
    /// A DIMENSION IS BOUND ONLY FROM AN AXIS THAT IS THAT DIMENSION. A
    /// compound extent (`2*s0`) says what its dimensions multiply to, not
    /// what any one of them is, so it is checked against them rather than
    /// read backwards; one dimension on two axes must be given one extent,
    /// wherever in the call those axes are; and nothing is recorded until
    /// every input has passed, so a refusal leaves the map as it was.
    fn bind_dims(&mut self, shapes: &[(String, Vec<usize>)]) -> Result<()> {
        let declared = |name: &str| -> Result<Vec<IntExpr>> {
            Ok(self
                .translation
                .inputs
                .iter()
                .find(|input| input.graph_name == name)
                .ok_or_else(|| anyhow!("unknown input {name:?}"))?
                .shape
                .clone())
        };
        // Which axis IS a dimension: the bare ones, over the whole call.
        let mut bound: HashMap<Symbol, (String, usize, usize)> = HashMap::new();
        for (name, shape) in shapes {
            for (axis, dim) in declared(name)?.iter().enumerate() {
                let Some(value) = shape.get(axis).copied() else {
                    continue;
                };
                if !is_bare(dim) {
                    continue;
                }
                let symbol = dim.to_symbols()[0];
                if let Some((first_input, first_axis, first)) =
                    bound.insert(symbol, (name.clone(), axis, value))
                    && first != value
                {
                    bail!(
                        "dimension {symbol} is axis {first_axis} of input {first_input:?} and \
                         axis {axis} of input {name:?}, called with {first} and {value}"
                    );
                }
            }
        }
        let call: DynMap = bound
            .iter()
            .map(|(symbol, (_, _, value))| (*symbol, *value))
            .collect();

        // A compound extent is a statement ABOUT the dimensions, checked
        // against the ones THIS call states and never inverted into one of
        // them.
        for (name, shape) in shapes {
            for (axis, dim) in declared(name)?.iter().enumerate() {
                let Some(value) = shape.get(axis).copied() else {
                    continue;
                };
                if is_bare(dim) || dim.to_symbols().is_empty() {
                    continue;
                }
                match dim.exec(&call) {
                    Some(computed) if computed == value => {}
                    Some(computed) => bail!(
                        "input {name:?}: axis {axis} is declared {dim}, which this call's \
                         dimensions make {computed}, but the caller's extent is {value}"
                    ),
                    None => bail!(
                        "input {name:?}: axis {axis} is declared {dim}, whose dimensions this \
                         call does not state, so the caller's extent {value} cannot be read \
                         as one of them"
                    ),
                }
            }
        }

        for (symbol, value) in call {
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

/// Is this declared extent a dimension itself, rather than an expression
/// over dimensions?
fn is_bare(dim: &IntExpr) -> bool {
    let symbols = dim.to_symbols();
    symbols.len() == 1 && *dim == IntExpr::from(symbols[0])
}

/// One boundary tensor's layout as the caller spelled it: the parsed form
/// the bindings carry, beside the spelling it arrived in. The spelling is
/// kept because it is the caller's own vocabulary — `output_layouts` hands
/// it back so Python rebuilds its binding in the terms it declared, rather
/// than re-reading the runtime's rendering of the same expression.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DeclaredLayout {
    layout: BoundaryLayout,
    tag: String,
    strides: Vec<String>,
}

impl DeclaredLayout {
    /// The wire form: the tag and, for a strided layout, its element
    /// strides.
    fn spelling(&self) -> (String, Vec<String>) {
        (self.tag.clone(), self.strides.clone())
    }
}

/// Read one declared layout: a tag, plus the element strides a strided
/// layout carries. A stride is a sympy `srepr` expression read against the
/// translated program's own symbols, so a caller whose storage is shaped
/// by a dynamic dimension states that dimension (`Symbol('s77')`) where a
/// static one states a number (`Integer(4)`). `role` is the side of the
/// boundary the row names, so a refusal says which.
///
/// The torch backend declares `strided` for every boundary and lets the
/// e-graph discover what map the chain is; the contiguous tags are for a
/// caller that states a contiguous layout itself.
fn layout_of(
    translation: &Translation,
    role: &str,
    name: &str,
    tag: &str,
    strides: &[String],
) -> Result<DeclaredLayout> {
    let layout = match tag {
        "row_major" => BoundaryLayout::RowMajor,
        "column_major" => BoundaryLayout::ColumnMajor,
        "strided" => {
            let parsed = strides
                .iter()
                .enumerate()
                .map(|(axis, stride)| {
                    parse_dim_expr(translation, stride)
                        .with_context(|| format!("{role} {name:?}, stride on axis {axis}"))
                })
                .collect::<Result<Vec<_>>>()?;
            BoundaryLayout::Strided { strides: parsed }
        }
        other => bail!("{role} {name:?}: unknown boundary layout {other:?}"),
    };
    Ok(DeclaredLayout {
        layout,
        tag: tag.to_string(),
        strides: strides.to_vec(),
    })
}

/// The caller's layout table for one side of the boundary, keyed by graph
/// name.
fn layout_table(
    translation: &Translation,
    role: &str,
    rows: &[(String, String, Vec<String>)],
) -> Result<HashMap<String, DeclaredLayout>> {
    let mut table = HashMap::new();
    for (name, tag, strides) in rows {
        let layout = layout_of(translation, role, name, tag, strides)?;
        if table.insert(name.clone(), layout).is_some() {
            bail!("{role} {name:?} was given two boundary layouts");
        }
    }
    Ok(table)
}

/// A bound translated program: the bindings, the buffer each input and
/// each output took, and the layout each output was bound at, spelled the
/// way the caller spelled it.
#[derive(Debug)]
struct Boundary {
    bindings: CudaBindings,
    input_buffers: Vec<i64>,
    output_buffers: Vec<i64>,
    output_layouts: Vec<(String, Vec<String>)>,
}

/// The CUDA-lite boundary for a translated program: every input is the
/// caller's live device memory, at the layout the caller declared for it, on
/// its own buffer; every user-visible output is a fresh caller-owned device
/// buffer at the layout the caller declared for it — eager's exact strides —
/// except a writeback, which binds on the buffer of the input it mutates, at
/// that input's layout — two bindings naming one buffer id being the single
/// spelling of aliasing.
fn bind(
    translation: &Translation,
    input_layouts: &HashMap<String, DeclaredLayout>,
    output_layouts: &HashMap<String, DeclaredLayout>,
    aliases: &HashMap<String, String>,
) -> Result<Boundary> {
    for name in input_layouts.keys() {
        ensure!(
            translation
                .inputs
                .iter()
                .any(|input| &input.graph_name == name),
            "a boundary layout was declared for {name:?}, which is not a graph input"
        );
    }
    for name in output_layouts.keys() {
        let output = translation
            .outputs
            .iter()
            .find(|output| &output.graph_name == name)
            .ok_or_else(|| {
                anyhow!("a boundary layout was declared for {name:?}, which is not a graph output")
            })?;
        // A writeback's layout is its target's, stated once on the input
        // side; a second statement here could disagree with it.
        if let Some(target) = &output.mutation_target {
            bail!(
                "a boundary layout was declared for output {name:?}, which writes back into \
                 {target:?}: a writeback is bound at the layout of the input it mutates"
            );
        }
    }
    let mut bindings = CudaBindings::new();
    let mut input_buffers = Vec::with_capacity(translation.inputs.len());
    for input in &translation.inputs {
        let layout = input_layouts.get(&input.graph_name).ok_or_else(|| {
            anyhow!(
                "input {:?} has no declared boundary layout",
                input.graph_name
            )
        })?;
        input_buffers.push(bindings.input_external_with(input.tensor, layout.layout.clone()));
    }
    let mut output_buffers = Vec::with_capacity(translation.outputs.len());
    let mut spellings = Vec::with_capacity(translation.outputs.len());
    for output in &translation.outputs {
        let (buffer, layout) = match &output.mutation_target {
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
                // A writeback writes the TARGET's storage, so it is bound at
                // the target's layout and never reinterprets it. Whether any
                // kernel writes that layout is the search's question.
                let layout = input_layouts[target].clone();
                let buffer = input_buffers[index];
                // The caller's storage is written through: the shared buffer
                // must say so.
                bindings.declare(buffer, Access::ReadWrite, FreedBy::Caller);
                bindings.output_on_with(output.tensor, buffer, layout.layout.clone());
                (buffer, layout)
            }
            None => {
                let layout = output_layouts.get(&output.graph_name).ok_or_else(|| {
                    anyhow!(
                        "output {:?} has no declared boundary layout",
                        output.graph_name
                    )
                })?;
                match aliases.get(&output.graph_name) {
                    // A user-visible output that shares its storage with an
                    // earlier boundary tensor — a view of an input, of a
                    // writeback, or of another output — is bound on THAT
                    // tensor's buffer at its own layout. Aliasing has one
                    // spelling, the same buffer id; the caller receives a
                    // view of the tensor it already holds.
                    Some(owner) => {
                        let buffer = if let Some(index) = translation
                            .inputs
                            .iter()
                            .position(|input| &input.graph_name == owner)
                        {
                            input_buffers[index]
                        } else if let Some(index) = translation
                            .outputs
                            .iter()
                            .position(|earlier| &earlier.graph_name == owner)
                            && index < output_buffers.len()
                        {
                            output_buffers[index]
                        } else {
                            bail!(
                                "output {:?} is declared a view of {owner:?}, which is neither \
                                 a graph input nor an earlier graph output",
                                output.graph_name
                            )
                        };
                        bindings.output_on_with(output.tensor, buffer, layout.layout.clone());
                        (buffer, layout.clone())
                    }
                    // Otherwise the caller's own fresh device allocation,
                    // laid out the way eager lays it out.
                    None => {
                        let buffer =
                            bindings.output_external_with(output.tensor, layout.layout.clone());
                        (buffer, layout.clone())
                    }
                }
            }
        };
        output_buffers.push(buffer);
        spellings.push(layout.spelling());
    }
    Ok(Boundary {
        bindings,
        input_buffers,
        output_buffers,
        output_layouts: spellings,
    })
}

/// Parse, translate, and load a `.pt2` on the CUDA-lite runtime under the
/// caller's boundary layouts: one `(graph name, layout tag, element strides)`
/// row per graph input and per user-visible graph output, the tag being
/// `row_major`, `column_major` or `strided`, and each stride a sympy `srepr`
/// expression over the exported program's symbols. A writeback takes no row —
/// it writes the storage of the input it mutates, at that input's layout.
/// `output_aliases` names each output that shares its storage with an earlier
/// boundary tensor (`(output, owner)`); it is bound on the owner's buffer.
#[pyfunction]
fn compile(
    pt2_path: &str,
    input_layouts: Vec<(String, String, Vec<String>)>,
    output_layouts: Vec<(String, String, Vec<String>)>,
    output_aliases: Vec<(String, String)>,
) -> PyResult<CompiledGraph> {
    let parsed = luminal_pytorch_utils::parse_pt2(pt2_path)
        .with_context(|| format!("parsing {pt2_path}"))
        .map_err(to_py)?;
    let translation = translate(&parsed).map_err(to_py)?;
    let dims: DynMap = translation.dims.iter().map(|(k, v)| (*k, *v)).collect();
    let inputs = layout_table(&translation, "input", &input_layouts).map_err(to_py)?;
    let outputs = layout_table(&translation, "output", &output_layouts).map_err(to_py)?;
    let aliases: HashMap<String, String> = output_aliases.into_iter().collect();
    let boundary = bind(&translation, &inputs, &outputs, &aliases)
        .context("declaring the translated program's boundary")
        .map_err(to_py)?;
    let runtime = CudaRuntime::load_with(
        &translation.graph,
        boundary.bindings,
        luminal_cuda_lite::ops::cuda_registry(),
    )
    .context("loading the translated graph on the cuda-lite runtime")
    .map_err(to_py)?;
    Ok(CompiledGraph {
        translation,
        runtime,
        input_buffers: boundary.input_buffers,
        output_buffers: boundary.output_buffers,
        output_layouts: boundary.output_layouts,
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

    /// One declared row: the parsed layout beside the spelling it came in.
    fn declared(layout: BoundaryLayout, tag: &str, strides: &[&str]) -> DeclaredLayout {
        DeclaredLayout {
            layout,
            tag: tag.to_string(),
            strides: strides.iter().map(|s| (*s).to_string()).collect(),
        }
    }

    fn row_major(names: &[&str]) -> HashMap<String, DeclaredLayout> {
        names
            .iter()
            .map(|name| {
                (
                    (*name).to_string(),
                    declared(BoundaryLayout::RowMajor, "row_major", &[]),
                )
            })
            .collect()
    }

    /// A row-major row for every output the caller allocates — what a
    /// contiguous eager output declares.
    fn row_major_outputs(translation: &Translation) -> HashMap<String, DeclaredLayout> {
        let names: Vec<&str> = translation
            .outputs
            .iter()
            .filter(|output| output.mutation_target.is_none())
            .map(|output| output.graph_name.as_str())
            .collect();
        row_major(&names)
    }

    /// A translation declaring one shape per named input — the spelling
    /// `bind_dims` reads — over a program that uses them all, so it loads.
    /// The graph's own extents are immaterial: `bind_dims` reads the
    /// DECLARED shapes and nothing else.
    fn declaring_inputs(rows: &[(&str, Vec<IntExpr>)]) -> Translation {
        let mut cx = Graph::new();
        let tensors: Vec<_> = rows
            .iter()
            .map(|(name, _)| cx.named_tensor(*name, (2usize, 3usize), DType::F32))
            .collect();
        let out = tensors[1..]
            .iter()
            .fold(tensors[0] + 1., |acc, tensor| acc * *tensor);
        Translation {
            graph: cx,
            inputs: rows
                .iter()
                .zip(&tensors)
                .map(|((name, shape), tensor)| TranslatedInput {
                    graph_name: (*name).to_string(),
                    parameter_name: None,
                    kind: InputKind::UserInput {
                        graph_name: (*name).to_string(),
                    },
                    tensor: tensor.id,
                    dtype: DType::F32,
                    shape: shape.clone(),
                })
                .collect(),
            outputs: vec![TranslatedOutput {
                graph_name: "out".to_string(),
                tensor: out.id,
                dtype: DType::F32,
                shape: vec![IntExpr::from(2i64), IntExpr::from(3i64)],
                mutation_target: None,
                returned: true,
            }],
            dims: HashMap::new(),
            symbols: HashMap::new(),
        }
    }

    /// The one-input case.
    fn declaring(shape: Vec<IntExpr>) -> Translation {
        declaring_inputs(&[("x", shape)])
    }

    /// One call's input shapes, the way `bind_input_shapes` takes them.
    fn call(rows: &[(&str, &[usize])]) -> Vec<(String, Vec<usize>)> {
        rows.iter()
            .map(|(name, shape)| ((*name).to_string(), shape.to_vec()))
            .collect()
    }

    /// The object `compile` hands back, built from a synthetic
    /// translation. `bind_dims` reads the declared input shapes and
    /// nothing else, so the loaded program's own structure is immaterial.
    fn compiled(translation: Translation) -> CompiledGraph {
        let names: Vec<&str> = translation
            .inputs
            .iter()
            .map(|input| input.graph_name.as_str())
            .collect();
        let layouts = row_major(&names);
        compiled_with(translation, &layouts)
    }

    /// The same, under a stated input layout table; every output the
    /// caller allocates is row-major.
    fn compiled_with(
        translation: Translation,
        layouts: &HashMap<String, DeclaredLayout>,
    ) -> CompiledGraph {
        let outputs = row_major_outputs(&translation);
        let boundary = bind(&translation, layouts, &outputs, &HashMap::new())
            .expect("the synthetic boundary binds");
        let dims: DynMap = translation.dims.iter().map(|(k, v)| (*k, *v)).collect();
        let runtime = CudaRuntime::load_with(
            &translation.graph,
            boundary.bindings,
            luminal_cuda_lite::ops::cuda_registry_without_cublaslt(),
        )
        .expect("the synthetic program loads");
        CompiledGraph {
            translation,
            runtime,
            input_buffers: boundary.input_buffers,
            output_buffers: boundary.output_buffers,
            output_layouts: boundary.output_layouts,
            searched: false,
            dims,
        }
    }

    /// ONE DIMENSION, ONE EXTENT: a dimension standing on two axes of one
    /// input is refused when the call gives the two axes different
    /// extents, rather than the second silently overwriting the first.
    #[test]
    fn a_dimension_on_two_axes_of_one_input_must_get_one_extent() {
        let symbol = Symbol::new("s0");
        let mut graph = compiled(declaring(vec![
            IntExpr::from(symbol),
            IntExpr::from(symbol),
        ]));
        let err = graph
            .bind_dims(&call(&[("x", &[3, 5])]))
            .expect_err("s0 cannot be both 3 and 5");
        let text = format!("{err:#}");
        assert!(text.contains("s0"), "{text}");
        assert!(text.contains('3') && text.contains('5'), "{text}");
        assert_eq!(
            graph.dims.get(&symbol),
            None,
            "a refused call bound nothing"
        );
    }

    /// A COMPOUND EXTENT IS NOT INVERTED: `2*s0` at extent 6 never says
    /// s0 is 6, and with nothing else on this input stating s0 it says
    /// nothing at all — so the call is refused by name rather than
    /// solved backwards.
    #[test]
    fn a_compound_extent_is_checked_never_read_backwards() {
        let symbol = Symbol::new("s0");
        let mut graph = compiled(declaring(vec![
            IntExpr::from(symbol) * IntExpr::from(2i64),
            IntExpr::from(4i64),
        ]));
        let err = graph
            .bind_dims(&call(&[("x", &[6, 4])]))
            .expect_err("nothing on this call states s0");
        let text = format!("{err:#}");
        assert!(text.contains("s0"), "{text}");
        assert!(text.contains("axis 0"), "{text}");
        assert!(text.contains('6'), "{text}");
        assert_eq!(
            graph.dims.get(&symbol),
            None,
            "the axis extent was read backwards into the dimension"
        );
    }

    /// A COMPOUND EXTENT IS CHECKED AGAINST THE DIMENSIONS the call does
    /// state: `2*s0` over an input whose other axis IS s0 agrees or is
    /// refused naming the axis, what the dimensions make it, and what the
    /// caller brought.
    #[test]
    fn a_compound_extent_is_checked_against_the_calls_dimensions() {
        let symbol = Symbol::new("s0");
        let shape = vec![
            IntExpr::from(symbol),
            IntExpr::from(symbol) * IntExpr::from(2i64),
        ];
        let mut graph = compiled(declaring(shape.clone()));
        let err = graph
            .bind_dims(&call(&[("x", &[3, 7])]))
            .expect_err("s0 is 3, so axis 1 is 6, not 7");
        let text = format!("{err:#}");
        assert!(text.contains("axis 1"), "{text}");
        assert!(text.contains('6') && text.contains('7'), "{text}");

        let mut graph = compiled(declaring(shape));
        graph
            .bind_dims(&call(&[("x", &[3, 6])]))
            .expect("3 and 2*3 agree");
        assert_eq!(graph.dims.get(&symbol), Some(&3));
    }

    /// ONE CALL, ONE DIMENSION MAP: a compound extent is checked against
    /// the dimensions the SAME call states, wherever in the call they are
    /// spelled — never against what the call before it left behind.
    #[test]
    fn a_compound_extent_is_checked_against_the_whole_calls_dimensions() {
        let symbol = Symbol::new("s0");
        let rows = [
            (
                "a",
                vec![
                    IntExpr::from(symbol) * IntExpr::from(2i64),
                    IntExpr::from(4i64),
                ],
            ),
            ("b", vec![IntExpr::from(symbol), IntExpr::from(4i64)]),
        ];
        for order in [[0usize, 1], [1, 0]] {
            let mut graph = compiled(declaring_inputs(&rows));
            let shapes = call(&[("a", &[6, 4]), ("b", &[3, 4])]);
            let shapes: Vec<_> = order.iter().map(|i| shapes[*i].clone()).collect();
            graph.bind_dims(&shapes).expect("6 is 2*3");
            assert_eq!(graph.dims.get(&symbol), Some(&3));

            let mut graph = compiled(declaring_inputs(&rows));
            let shapes = call(&[("a", &[6, 4]), ("b", &[2, 4])]);
            let shapes: Vec<_> = order.iter().map(|i| shapes[*i].clone()).collect();
            let err = graph.bind_dims(&shapes).expect_err("2*2 is not 6");
            let text = format!("{err:#}");
            assert!(text.contains("\"a\"") && text.contains("axis 0"), "{text}");
            assert!(text.contains('4') && text.contains('6'), "{text}");
            assert_eq!(
                graph.dims.get(&symbol),
                None,
                "a refused call recorded a dimension"
            );
        }
    }

    /// ONE DIMENSION, ONE EXTENT ACROSS THE CALL: two inputs that give one
    /// dimension two extents are refused naming both, rather than the
    /// second silently overwriting the first.
    #[test]
    fn a_dimension_on_two_inputs_must_get_one_extent() {
        let symbol = Symbol::new("s0");
        let rows = [
            ("a", vec![IntExpr::from(symbol), IntExpr::from(4i64)]),
            ("b", vec![IntExpr::from(symbol), IntExpr::from(4i64)]),
        ];
        let mut graph = compiled(declaring_inputs(&rows));
        let err = graph
            .bind_dims(&call(&[("a", &[3, 4]), ("b", &[5, 4])]))
            .expect_err("s0 cannot be both 3 and 5");
        let text = format!("{err:#}");
        assert!(text.contains("\"a\"") && text.contains("\"b\""), "{text}");
        assert!(text.contains('3') && text.contains('5'), "{text}");
        assert_eq!(graph.dims.get(&symbol), None, "a refused call bound a dim");
    }

    /// ALIASING HAS ONE SPELLING: the writeback and the input it mutates
    /// name one buffer id, so the caller addresses one pointer.
    #[test]
    fn a_writeback_and_its_target_are_one_buffer() {
        let translation = translation(&["x"], Some("x"));
        let boundary = bind(
            &translation,
            &row_major(&["x"]),
            &HashMap::new(),
            &HashMap::new(),
        )
        .expect("writeback binds");
        assert_eq!(boundary.output_buffers[0], boundary.input_buffers[0]);
        assert_eq!(
            boundary.bindings.buffers()[&boundary.input_buffers[0]].access,
            Access::ReadWrite
        );
    }

    #[test]
    fn a_mutation_target_that_is_not_a_graph_input_is_refused() {
        let translation = translation(&["x"], Some("elsewhere"));
        let err = bind(
            &translation,
            &row_major(&["x"]),
            &HashMap::new(),
            &HashMap::new(),
        )
        .expect_err("no such input");
        assert!(
            format!("{err:#}").contains("mutates \"elsewhere\""),
            "{err:#}"
        );
    }

    #[test]
    fn an_input_with_no_layout_row_is_refused() {
        let translation = translation(&["x", "y"], None);
        let outputs = row_major_outputs(&translation);
        let err = bind(&translation, &row_major(&["x"]), &outputs, &HashMap::new())
            .expect_err("y has no layout");
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
        let outputs = row_major_outputs(&translation);
        let err = bind(
            &translation,
            &row_major(&["x", "ghost"]),
            &outputs,
            &HashMap::new(),
        )
        .expect_err("no such input");
        assert!(format!("{err:#}").contains("\"ghost\""), "{err:#}");
    }

    /// AN OUTPUT IS BOUND AT THE LAYOUT THE CALLER DECLARED FOR IT, on its
    /// own caller-owned buffer: the tensor handed back carries eager's
    /// strides, never a row-major substitute.
    #[test]
    fn an_output_binds_external_at_its_declared_layout() {
        let translation = translation(&["x"], None);
        let outputs: HashMap<String, DeclaredLayout> = [(
            "out".to_string(),
            declared(BoundaryLayout::ColumnMajor, "column_major", &[]),
        )]
        .into();
        let boundary = bind(&translation, &row_major(&["x"]), &outputs, &HashMap::new())
            .expect("the output binds");
        assert_eq!(
            boundary.bindings.outputs()[0].layout,
            BoundaryLayout::ColumnMajor
        );
        assert_ne!(boundary.output_buffers[0], boundary.input_buffers[0]);
        assert_eq!(
            boundary.output_layouts,
            vec![("column_major".to_string(), Vec::new())]
        );
    }

    /// A WRITEBACK TAKES NO OUTPUT ROW: its layout is its target's, stated
    /// once on the input side, and a second statement here could disagree
    /// with it. `output_layouts` still answers for it, repeating the
    /// target's row, so the caller reads one table for every output.
    #[test]
    fn an_output_row_for_a_writeback_is_refused() {
        let translation = translation(&["x"], Some("x"));
        let outputs = row_major(&["out"]);
        let err = bind(&translation, &row_major(&["x"]), &outputs, &HashMap::new())
            .expect_err("out is a writeback");
        let text = format!("{err:#}");
        assert!(text.contains("output \"out\"") && text.contains("writes back into \"x\""));

        let inputs: HashMap<String, DeclaredLayout> = [(
            "x".to_string(),
            declared(BoundaryLayout::ColumnMajor, "column_major", &[]),
        )]
        .into();
        let boundary = bind(&translation, &inputs, &HashMap::new(), &HashMap::new())
            .expect("the writeback binds");
        assert_eq!(
            boundary.output_layouts,
            vec![("column_major".to_string(), Vec::new())]
        );
    }

    /// A layout row naming a tensor that is not a graph output is a
    /// statement about nothing: refused rather than dropped.
    #[test]
    fn an_output_row_for_a_non_output_is_refused() {
        let translation = translation(&["x"], None);
        let outputs = row_major(&["out", "ghost"]);
        let err = bind(&translation, &row_major(&["x"]), &outputs, &HashMap::new())
            .expect_err("no such output");
        let text = format!("{err:#}");
        assert!(
            text.contains("\"ghost\"") && text.contains("not a graph output"),
            "{text}"
        );
    }

    #[test]
    fn an_output_with_no_layout_row_is_refused() {
        let translation = translation(&["x"], None);
        let err = bind(
            &translation,
            &row_major(&["x"]),
            &HashMap::new(),
            &HashMap::new(),
        )
        .expect_err("out has no layout");
        assert!(
            format!("{err:#}").contains("output \"out\" has no declared boundary layout"),
            "{err:#}"
        );
    }

    /// AN OUTPUT'S ROW IS READ BACK IN THE SPELLING IT ARRIVED IN, so the
    /// caller rebuilds its own binding in its own vocabulary rather than
    /// re-reading the runtime's rendering of the same expression.
    #[test]
    fn output_layouts_echo_the_declared_spelling() {
        let symbol = Symbol::new("s77");
        let mut translation = translation(&["x"], None);
        translation.symbols.insert("s77".to_string(), symbol);
        let strides = vec![
            "Symbol('s77', positive=True, integer=True)".to_string(),
            "Integer(1)".to_string(),
        ];
        let rows = vec![("out".to_string(), "strided".to_string(), strides.clone())];
        let outputs = layout_table(&translation, "output", &rows).expect("the output row is read");
        let BoundaryLayout::Strided { strides: parsed } = &outputs["out"].layout else {
            panic!("expected a strided layout, got {:?}", outputs["out"]);
        };
        assert!(parsed[0].to_symbols().contains(&symbol));
        let boundary = bind(&translation, &row_major(&["x"]), &outputs, &HashMap::new())
            .expect("a strided output binds");
        assert_eq!(
            boundary.output_layouts,
            vec![("strided".to_string(), strides)]
        );
    }

    /// AN OUTPUT ROW IS READ IN THE SAME VOCABULARY AS AN INPUT'S: a stride
    /// naming a dimension the program does not declare is refused naming the
    /// output and the axis, so the refusal says which side of the boundary
    /// the unreadable stride came from.
    #[test]
    fn an_output_stride_naming_an_undeclared_symbol_is_refused() {
        let translation = translation(&["x"], None);
        let rows = vec![(
            "out".to_string(),
            "strided".to_string(),
            vec!["Integer(1)".to_string(), "Symbol('s77')".to_string()],
        )];
        let err = layout_table(&translation, "output", &rows).expect_err("s77 is not declared");
        let text = format!("{err:#}");
        assert!(text.contains("output \"out\""), "{text}");
        assert!(text.contains("axis 1"), "{text}");
    }

    #[test]
    fn an_unknown_layout_tag_is_refused() {
        let translation = translation(&["x"], None);
        let rows = vec![("x".to_string(), "diagonal".to_string(), Vec::new())];
        let err = layout_table(&translation, "input", &rows).expect_err("no such layout");
        assert!(
            format!("{err:#}").contains("unknown boundary layout \"diagonal\""),
            "{err:#}"
        );
    }

    /// A WRITEBACK IS BOUND AT ITS TARGET'S LAYOUT: the sink writes the
    /// caller's own storage, so it states that storage's layout and never
    /// reinterprets it as row-major.
    #[test]
    fn a_writeback_binds_at_its_targets_layout() {
        let translation = translation(&["x"], Some("x"));
        let layouts: HashMap<String, DeclaredLayout> = [(
            "x".to_string(),
            declared(BoundaryLayout::ColumnMajor, "column_major", &[]),
        )]
        .into();
        let boundary = bind(&translation, &layouts, &HashMap::new(), &HashMap::new())
            .expect("a writeback binds at its target's layout");
        assert_eq!(boundary.output_buffers[0], boundary.input_buffers[0]);
        assert_eq!(
            boundary.bindings.outputs()[0].layout,
            BoundaryLayout::ColumnMajor
        );
        assert_eq!(
            boundary.bindings.buffers()[&boundary.input_buffers[0]].access,
            Access::ReadWrite
        );
    }

    /// WHETHER A KERNEL WRITES THAT LAYOUT IS THE SEARCH'S QUESTION, and
    /// today none writes a left-major destination: the answer is a search
    /// that plans nothing and names the output and the layout it is bound
    /// at, which is the text `search()` hands Python unchanged.
    #[test]
    fn a_writeback_the_kernels_cannot_write_is_named_by_the_search() {
        let mut cx = Graph::new();
        let x = cx.named_tensor("x", (2usize, 3usize), DType::F32);
        let out = x + 1.;
        let out_value = out.id.index();
        let shape = vec![IntExpr::from(2i64), IntExpr::from(3i64)];
        let translation = Translation {
            graph: cx,
            inputs: vec![TranslatedInput {
                graph_name: "x".to_string(),
                parameter_name: None,
                kind: InputKind::UserInput {
                    graph_name: "x".to_string(),
                },
                tensor: x.id,
                dtype: DType::F32,
                shape: shape.clone(),
            }],
            outputs: vec![TranslatedOutput {
                graph_name: "out".to_string(),
                tensor: out.id,
                dtype: DType::F32,
                shape,
                mutation_target: Some("x".to_string()),
                returned: true,
            }],
            dims: HashMap::new(),
            symbols: HashMap::new(),
        };
        let layouts: HashMap<String, DeclaredLayout> = [(
            "x".to_string(),
            declared(BoundaryLayout::ColumnMajor, "column_major", &[]),
        )]
        .into();
        let err = compiled_with(translation, &layouts)
            .run_search(Some(1))
            .expect_err("no kernel writes a left-major destination");
        let text = format!("{err:#}");
        assert!(text.contains(&format!("v{out_value}")), "{text}");
        assert!(text.contains("ColumnMajor"), "{text}");
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
        let table = layout_table(&translation, "input", &rows).expect("symbolic strides");
        let BoundaryLayout::Strided { strides } = &table["x"].layout else {
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
        let err = layout_table(&translation, "input", &rows).expect_err("s77 is not declared");
        let text = format!("{err:#}");
        assert!(text.contains("Symbol('s77')"), "{text}");
        assert!(text.contains("axis 1"), "{text}");
    }
}
