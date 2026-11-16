//! A searched plan lowered to what the device launches.
//!
//! The plan search selects has one node per operation, and each node holds
//! an op object. Lowering walks that plan ONCE and flattens it into a list of
//! steps the device can run without any op: a kernel launch (the complete
//! source of one kernel, its pointer arguments and its launch size), a copy,
//! or a host-library call (the call's arguments as plain data).
//!
//! A lowered program is plain data. It can be written out and read back
//! (it derives serde), and a device installs the one that was read back
//! exactly as it installs the one that was lowered in the process. Two
//! things are needed before it can be written: every kernel compiled
//! ([`LoweredProgram::compile_kernels`]), so reading it back runs no
//! compiler, and no host step that is still an op object
//! ([`LoweredProgram::ensure_portable`]).
//!
//! Lowering needs no device and no call. An op's code generator is handed
//! symbolic extents, dtypes and layouts only, so the source it writes cannot
//! depend on a call's sizes or addresses: a kernel reads its dimensions from
//! the parameter block at run time. What does depend on a call (a copy's
//! byte count, a custom launch size) stays an expression here and is
//! evaluated per call by the device.

use std::collections::BTreeSet;

use anyhow::{Result, anyhow, bail, ensure};
use luminal::{
    buffer_tensor_ir::BufferTensorIrOp,
    bufferize::{BufferId, BufferNode, SlotDescriptor},
    layouts::DecodedLayout,
    prelude::FxHashMap,
    shape::{DynMap, Symbol},
};

use crate::{
    arena::{ArenaPlan, ArenaSlice, ArenaStep},
    kernels::{CodegenCtx, KernelLaunch},
    layouts::CudaPlan,
    ops::cublaslt::{
        accumulate_in_place::CublasLtAccumulateInPlace, exec::LtCallTemplate,
        out_of_place::CublasLtOutOfPlace,
    },
    symbolic::{self, Bounds, Expr},
};

/// Threads per block of a default launch.
pub const DEFAULT_BLOCK: u32 = 256;
/// Dynamic shared memory above which a function's limit must be raised.
const SHARED_MEMORY_LIMIT: usize = 48 * 1024;

/// A plan as the steps the device runs, in order. Each step depends on the
/// one before it.
///
/// It is everything a device needs to run the program: no plan, op or
/// search state stands behind it.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct LoweredProgram {
    /// The program's dimensions, in parameter-block order: slot `i` of the
    /// block holds dimension `i`.
    pub dims: Vec<Dimension>,
    /// Every buffer a step names: how many bytes it spans and where it
    /// lives.
    #[serde(with = "buffer_table")]
    pub buffers: FxHashMap<BufferId, BufferRecord>,
    /// The code of every kernel a step launches. Steps that launch the same
    /// kernel share one entry.
    pub kernels: Vec<KernelCode>,
    /// Where the parameter block sits in the arena, and in host staging.
    pub parameters: ArenaSlice,
    pub staging_parameters: ArenaSlice,
    /// Bytes of pinned host staging a call needs.
    pub staging_bytes: usize,
    /// Bytes of device arena a call needs.
    pub arena_bytes: usize,
    pub steps: Vec<Step>,
    /// Where the program leaves each output, one per output slot.
    pub outputs: Vec<OutputRecord>,
}

impl LoweredProgram {
    /// The dimensions in parameter-block order.
    pub fn schema(&self) -> Vec<Symbol> {
        self.dims.iter().map(|dim| dim.symbol).collect()
    }

    /// The record of output slot `slot` (binding order).
    pub fn output(&self, slot: usize) -> Result<&OutputRecord> {
        self.outputs
            .iter()
            .find(|output| output.slot == slot)
            .ok_or_else(|| anyhow!("the program has no output slot {slot}"))
    }

    /// The record of a buffer a step or an output names.
    pub fn buffer(&self, id: &BufferId) -> Result<&BufferRecord> {
        self.buffers
            .get(id)
            .ok_or_else(|| anyhow!("unbound buffer {id:?}"))
    }

    /// A buffer's range in the arena. `None` for caller-owned memory and
    /// for a buffer the program does not hold.
    pub fn slice(&self, id: &BufferId) -> Option<ArenaSlice> {
        self.buffers.get(id).and_then(|record| record.slice)
    }

    /// Refuse, by name, what cannot be written out: a kernel that is still
    /// source, and a host step that is still an op object.
    pub fn ensure_portable(&self) -> Result<()> {
        for step in &self.steps {
            match step {
                Step::Kernel(record) => ensure!(
                    matches!(self.kernels.get(record.code), Some(KernelCode::Ptx(_))),
                    "kernel {} is still source: compile the program's kernels before \
                     writing it out",
                    record.label
                ),
                Step::Host(record) => ensure!(
                    !matches!(record.call, HostCall::Op { .. }),
                    "host operation {} has no call record, so a program that holds it \
                     cannot be written out",
                    record.label
                ),
                Step::Copy(_) => {}
            }
        }
        Ok(())
    }

    /// Compile every kernel that is still source, so that installing the
    /// program runs no compiler. This needs the compiler library and no
    /// device.
    #[cfg(feature = "device")]
    pub fn compile_kernels(&mut self) -> Result<()> {
        for code in &mut self.kernels {
            if let KernelCode::Source(source) = code {
                *code = KernelCode::Ptx(crate::device::compile_kernel(source)?);
            }
        }
        Ok(())
    }
}

/// The code of one kernel's module. Its entry point is `k`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KernelCode {
    /// The complete CUDA source: includes, helpers, one `#define` per
    /// dimension reading it from the parameter block, and the kernel. A
    /// device compiles it the first time it needs it.
    Source(String),
    /// The compiled module as PTX. The driver loads it as it is, on any
    /// device at least as new as the compiler's default target.
    Ptx(String),
}

impl KernelCode {
    /// The code itself, which is also what identifies the kernel.
    pub fn text(&self) -> &str {
        match self {
            Self::Source(text) | Self::Ptx(text) => text,
        }
    }
}

/// The buffer table as a list of `(buffer, record)` rows in a fixed order: a
/// buffer id is not a string, so it cannot key a map in every encoding, and
/// a fixed order makes the same program the same bytes.
mod buffer_table {
    use super::{BufferId, BufferRecord, FxHashMap};
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(
        buffers: &FxHashMap<BufferId, BufferRecord>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let mut rows: Vec<(&BufferId, &BufferRecord)> = buffers.iter().collect();
        rows.sort_by_cached_key(|(id, _)| format!("{id:?}"));
        rows.serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<FxHashMap<BufferId, BufferRecord>, D::Error> {
        Ok(Vec::<(BufferId, BufferRecord)>::deserialize(deserializer)?
            .into_iter()
            .collect())
    }
}

/// One dimension of the program, and the range it was compiled for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Dimension {
    pub symbol: Symbol,
    pub min: usize,
    pub max: usize,
}

/// Where the program leaves one output.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OutputRecord {
    /// The output's slot, in binding order.
    pub slot: usize,
    /// The buffer the program writes the output's bytes into.
    pub buffer: BufferId,
    /// The element strides the program elected for the output, one per axis,
    /// as expressions over the dimensions. `None` when the elected layout
    /// states an offset function instead of strides.
    pub strides: Option<Vec<Expr>>,
}

impl OutputRecord {
    /// The elected element strides at a call's dimensions.
    pub fn strides_at(&self, dims: &DynMap) -> Result<Vec<i64>> {
        self.strides
            .as_ref()
            .ok_or_else(|| anyhow!("output slot {} has no elected strides", self.slot))?
            .iter()
            .map(|stride| symbolic::eval(&stride.0, dims))
            .collect()
    }
}

/// One element stride per axis of a layout, taken from the constructor's own
/// meaning: the contiguous forms state theirs by shape order, a strided chain
/// by its three canonical residues (`coord * stride`, the bare coordinate,
/// the dead axis's zero). `None` for a layout that states an offset function
/// instead of strides, and for a chain summand outside those residues: a
/// stride is never recovered by evaluation.
pub fn elected_strides(layout: &DecodedLayout) -> Option<Vec<Expr>> {
    use luminal::layouts::{
        IntExprTerm as T, LeftMajorContiguousElementLayout as LM,
        RightMajorContiguousElementLayout as RM, StridedElementLayout as ST,
    };
    let extents: Vec<Expr> = layout.shape().0.iter().cloned().map(Expr).collect();
    let rank = extents.len();
    let one = || Expr::from(1usize);
    if layout.has::<RM>() {
        let mut strides: Vec<Expr> = (0..rank).map(|_| one()).collect();
        for axis in (0..rank.saturating_sub(1)).rev() {
            strides[axis] = strides[axis + 1].clone() * extents[axis + 1].clone();
        }
        return Some(strides);
    }
    if layout.has::<LM>() {
        let mut strides: Vec<Expr> = (0..rank).map(|_| one()).collect();
        for axis in 1..rank {
            strides[axis] = strides[axis - 1].clone() * extents[axis - 1].clone();
        }
        return Some(strides);
    }
    let chain = &layout.first::<ST>()?.chain;
    if chain.len() != rank {
        return None;
    }
    // A summand names its own axis FROM THE END; only the dead axis's bare
    // zero has none, and it sits at that axis's position in the chain.
    let axis_of = |axis_from_end: &i64| -> Option<usize> {
        let axis = usize::try_from(*axis_from_end).ok()?;
        (axis < rank).then_some(rank - 1 - axis)
    };
    // A stride is an expression over the dimensions alone.
    fn free_of_coordinates(term: &luminal::layouts::IntExprTerm) -> bool {
        use luminal::layouts::IntExprTerm as T;
        match term {
            T::Lit(_) | T::Var(_) => true,
            T::Coord { .. } => false,
            T::Add(a, b)
            | T::Mul(a, b)
            | T::TruncDiv(a, b)
            | T::TruncRem(a, b)
            | T::CeilDiv(a, b)
            | T::Min(a, b)
            | T::Max(a, b)
            | T::LessThanCast(a, b) => free_of_coordinates(a) && free_of_coordinates(b),
        }
    }
    let mut strides: Vec<Option<Expr>> = vec![None; rank];
    for (position, summand) in chain.iter().enumerate() {
        let (axis, stride) = match summand {
            T::Lit(0) => (position, Expr(T::Lit(0))),
            T::Coord { axis_from_end } => (axis_of(axis_from_end)?, one()),
            T::Mul(a, b) => match (a.as_ref(), b.as_ref()) {
                (T::Coord { axis_from_end }, k) | (k, T::Coord { axis_from_end })
                    if free_of_coordinates(k) =>
                {
                    (axis_of(axis_from_end)?, Expr(k.clone()))
                }
                _ => return None,
            },
            _ => return None,
        };
        if strides[axis].replace(stride).is_some() {
            return None;
        }
    }
    strides.into_iter().collect()
}

/// One buffer of the program.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BufferRecord {
    /// The bytes the buffer spans, as an expression over the dimensions.
    /// `None` for a buffer whose layout states no storage span: no step can
    /// address it.
    pub size: Option<Expr>,
    /// The id a caller supplies a device pointer under, for a boundary
    /// buffer.
    pub lit: Option<i64>,
    /// The buffer's range in the arena. `None` for caller-owned memory,
    /// which is addressed through the pointer the caller supplies.
    pub slice: Option<ArenaSlice>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Step {
    Kernel(KernelRecord),
    Copy(CopyRecord),
    Host(HostRecord),
}

/// One host-library call.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct HostRecord {
    pub label: String,
    pub call: HostCall,
    /// The buffers the call reads, in the order the call takes them.
    pub inputs: Vec<BufferId>,
    pub dest: BufferId,
    /// Scratch the library may use, live only during the call.
    pub workspace: ArenaSlice,
    /// The dimensions the prepared call depends on; `None` for all of them.
    pub capture_dims: Option<Vec<Symbol>>,
}

/// What a host step calls.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HostCall {
    /// A cuBLASLt matrix multiply as plain data. Every host call this
    /// runtime's own operations make is one of these.
    CublasLt(Box<LtCallTemplate>),
    /// A host operation that has no call record: one defined outside this
    /// crate. It is prepared through its own op object, so a program that
    /// holds one runs in the process that lowered it and cannot be written
    /// out ([`LoweredProgram::ensure_portable`] names it).
    #[serde(skip)]
    Op {
        op: Box<dyn BufferTensorIrOp>,
        operand_info: Vec<SlotDescriptor<DecodedLayout>>,
        result_info: Vec<SlotDescriptor<DecodedLayout>>,
    },
}

impl HostCall {
    /// The call record of a host operation: the plain call of either
    /// cuBLASLt operation, or the operation itself.
    fn of(
        op: &dyn BufferTensorIrOp,
        operand_info: &[SlotDescriptor<DecodedLayout>],
        result_info: &[SlotDescriptor<DecodedLayout>],
    ) -> Self {
        let any = op.as_any();
        if let Some(op) = any.downcast_ref::<CublasLtOutOfPlace>() {
            Self::CublasLt(Box::new(op.template()))
        } else if let Some(op) = any.downcast_ref::<CublasLtAccumulateInPlace>() {
            Self::CublasLt(Box::new(op.template()))
        } else {
            Self::Op {
                op: op.clone_bt_box(),
                operand_info: operand_info.to_vec(),
                result_info: result_info.to_vec(),
            }
        }
    }
}

/// One pointer argument of a kernel.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArgRef {
    /// A plan buffer: an arena slice, or a caller's pointer.
    Buffer(BufferId),
    /// The parameter block that holds the call's dimensions.
    Params,
}

/// One kernel launch.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct KernelRecord {
    /// The op that wrote the kernel, for diagnostics only.
    pub label: String,
    /// The kernel's code, as its index in the program's kernel table.
    pub code: usize,
    /// The kernel's pointer arguments, in order.
    pub args: Vec<ArgRef>,
    pub launch: LaunchRecord,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LaunchRecord {
    /// One thread per element at the largest size the dimension ranges
    /// allow, in blocks of [`DEFAULT_BLOCK`]. The kernel guards each thread
    /// against its live element count, so this is fixed for the program.
    Default { grid_x: u32 },
    /// A launch size the op states as expressions over the dimensions,
    /// evaluated per call. `max_dynamic_shared` is set when the shared
    /// memory can exceed the default limit of a function.
    Custom {
        geometry: KernelLaunch,
        max_dynamic_shared: Option<i32>,
    },
}

/// A device-to-device copy of `size` bytes from one plan buffer to another.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct CopyRecord {
    pub src: BufferId,
    pub dst: BufferId,
    /// The byte count, as an expression over the dimensions.
    pub size: Expr,
    pub check: CopyCheck,
}

/// What a copy's byte count must satisfy at a call's dimensions.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CopyCheck {
    /// A whole-buffer copy: the destination holds exactly that many bytes.
    Equal(Expr),
    /// A value copy: it fits in the source and in the destination.
    Capacity(Expr, Expr),
}

impl CopyCheck {
    pub fn check(&self, bytes: usize, dims: &DynMap) -> Result<()> {
        match self {
            Self::Equal(other) => ensure!(bytes == other.eval(dims)?, "copy length mismatch"),
            Self::Capacity(source, dest) => ensure!(
                bytes <= source.eval(dims)? && bytes <= dest.eval(dims)?,
                "value copy exceeds buffer capacity"
            ),
        }
        Ok(())
    }

    pub fn variables(&self, vars: &mut BTreeSet<Symbol>) {
        match self {
            Self::Equal(other) => symbolic::vars(&other.0, vars),
            Self::Capacity(a, b) => {
                symbolic::vars(&a.0, vars);
                symbolic::vars(&b.0, vars);
            }
        }
    }
}

/// The bytes a layout spans, as an expression over the dimensions.
pub fn size(layout: &DecodedLayout) -> Result<Expr> {
    Ok(symbolic::span(layout)?
        * Expr::from(crate::host_buffer::dtype_bytes(
            layout.dtype.ok_or_else(|| anyhow!("missing dtype"))?,
        )?))
}

/// Lower a plan to the steps the device runs. `storage` gives the order the
/// plan's nodes are issued in, and `bounds` the range of each dimension.
pub fn lower(plan: &CudaPlan, storage: &ArenaPlan, bounds: &Bounds) -> Result<LoweredProgram> {
    let dims: Vec<Dimension> = bounds
        .iter()
        .map(|(symbol, (min, max))| Dimension {
            symbol: *symbol,
            min: *min,
            max: *max,
        })
        .collect();
    let schema: Vec<Symbol> = dims.iter().map(|dim| dim.symbol).collect();
    let mut steps = Vec::new();
    let mut outputs = Vec::new();
    let mut kernels: Vec<KernelCode> = Vec::new();
    let mut kernel_of: FxHashMap<String, usize> = FxHashMap::default();
    for step in &storage.steps {
        let ArenaStep::Node(node) = step;
        match &plan.dag[*node] {
            BufferNode::Compute {
                op,
                reads,
                writes,
                operand_info,
                result_info,
                ..
            } => {
                let label = op.label();
                if matches!(label, "BufferAlloc" | "BufferFree") {
                    continue;
                }
                ensure!(
                    writes.len() == 1,
                    "{label}: CUDA requires a single destination"
                );
                ensure!(
                    operand_info.len() == reads.len() && result_info.len() == writes.len(),
                    "{label}: missing slot descriptors"
                );
                if let Some(host) = crate::as_host_op(op.as_ref()) {
                    steps.push(Step::Host(HostRecord {
                        label: label.to_string(),
                        call: HostCall::of(op.as_ref(), operand_info, result_info),
                        inputs: reads
                            .iter()
                            .enumerate()
                            .filter(|(operand, _)| op.operand_reads_memory(*operand))
                            .map(|(_, id)| id.clone())
                            .collect(),
                        dest: writes[0].clone(),
                        workspace: storage.workspaces.get(node).copied().unwrap_or_default(),
                        capture_dims: host.capture_dims(),
                    }));
                } else if let Some(kernel) = crate::as_kernel_op(op.as_ref()) {
                    let codegen = CodegenCtx::from_descriptors(label, operand_info, result_info)?;
                    // Every non-destination operand in plan order, then the
                    // destination, then the parameter block.
                    let args: Vec<ArgRef> = reads[..reads.len() - writes.len()]
                        .iter()
                        .map(|id| ArgRef::Buffer(id.clone()))
                        .chain([ArgRef::Buffer(writes[0].clone()), ArgRef::Params])
                        .collect();
                    for generated in kernel.codegen(&codegen)? {
                        let mut source = crate::kernels::dtype_includes(&{
                            let mut dtypes = codegen.operand_dtypes.clone();
                            dtypes.extend_from_slice(&codegen.dest_dtypes);
                            dtypes
                        });
                        source.push_str(symbolic::CUDA_HELPERS);
                        for (i, s) in schema.iter().enumerate() {
                            source.push_str(&format!(
                                "#define {} params[{i}]\n",
                                symbolic::variable(&s.to_string())
                            ));
                        }
                        source.push_str(&generated.source);
                        let grid_x =
                            u32::try_from(generated.n.capacity(bounds)?.max(1).div_ceil(256))?;
                        ensure!(
                            grid_x <= i32::MAX as u32,
                            "kernel capacity exceeds CUDA grid limit"
                        );
                        let launch = match generated.launch {
                            None => LaunchRecord::Default { grid_x },
                            Some(geometry) => {
                                for expr in geometry.expressions() {
                                    expr.capacity(bounds)?;
                                }
                                let shared = geometry.shared_bytes.capacity(bounds)?;
                                LaunchRecord::Custom {
                                    max_dynamic_shared: (shared > SHARED_MEMORY_LIMIT)
                                        .then(|| i32::try_from(shared))
                                        .transpose()?,
                                    geometry,
                                }
                            }
                        };
                        let code = match kernel_of.get(&source) {
                            Some(code) => *code,
                            None => {
                                kernels.push(KernelCode::Source(source.clone()));
                                kernel_of.insert(source, kernels.len() - 1);
                                kernels.len() - 1
                            }
                        };
                        steps.push(Step::Kernel(KernelRecord {
                            label: label.to_string(),
                            code,
                            args: args.clone(),
                            launch,
                        }));
                    }
                } else {
                    bail!("no CUDA execution interface for {label}");
                }
            }
            BufferNode::BufferCopy { src, dst, .. }
            | BufferNode::BufferCopyValue { src, dst, .. } => {
                let value_layout = plan.dag[*node].copy_value_layout();
                let copy_size = value_layout.unwrap_or(&plan.buffers[src].layout);
                let check = if value_layout.is_some() {
                    CopyCheck::Capacity(
                        size(&plan.buffers[src].layout)?,
                        size(&plan.buffers[dst].layout)?,
                    )
                } else {
                    CopyCheck::Equal(size(&plan.buffers[dst].layout)?)
                };
                steps.push(Step::Copy(CopyRecord {
                    src: src.clone(),
                    dst: dst.clone(),
                    size: size(copy_size)?,
                    check,
                }));
            }
            _ => {}
        }
    }
    // The outputs are read off the whole plan: an output node is not a
    // device step, so the issue order does not hold it.
    for node in plan.dag.node_weights() {
        let BufferNode::BufferOutput { slots } = node else {
            continue;
        };
        for slot in slots {
            outputs.push(OutputRecord {
                slot: slot.index,
                buffer: slot.buffer.clone(),
                strides: elected_strides(&slot.layout),
            });
        }
    }
    outputs.sort_by_key(|output| output.slot);
    let buffers = plan
        .buffers
        .iter()
        .map(|(id, buffer)| {
            (
                id.clone(),
                BufferRecord {
                    size: size(&buffer.layout).ok(),
                    lit: buffer.lit,
                    slice: storage.slices.get(id).copied(),
                },
            )
        })
        .collect();
    Ok(LoweredProgram {
        dims,
        buffers,
        kernels,
        parameters: storage.parameters,
        staging_parameters: storage.staging_parameters,
        staging_bytes: storage.staging_bytes,
        arena_bytes: storage.slab_bytes,
        steps,
        outputs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::cublaslt::{
        CublasLtForm,
        exec::{
            LtActivation, LtComputation, LtMatrixTemplate, LtOrder, LtReductionSchemeMask,
            LtScalars,
        },
    };
    use luminal::{dtype::PlanDtype, index_expr::IotaExpr, layouts::IntExprTerm as T};

    fn matrix(rows: IotaExpr, cols: i64) -> LtMatrixTemplate {
        LtMatrixTemplate {
            dtype: PlanDtype::F32,
            order: LtOrder::Row,
            rows: rows.clone(),
            cols: IotaExpr::Lit(cols),
            ld: IotaExpr::Lit(cols),
            batch_count: IotaExpr::Lit(1),
            batch_stride_in_elements: IotaExpr::Mul(Box::new(rows), Box::new(IotaExpr::Lit(cols))),
        }
    }

    /// A program with one step of each kind, built by hand.
    fn program(kernel: KernelCode) -> LoweredProgram {
        let n = || IotaExpr::Var("n".to_string());
        let bytes = |elements: i64| {
            Expr(T::Mul(
                Box::new(T::Var("n".to_string())),
                Box::new(T::Lit(elements * 4)),
            ))
        };
        let buffer = |index: u32| BufferId::Allocated(index);
        let record = |index: usize, elements: i64, lit: Option<i64>| BufferRecord {
            size: Some(bytes(elements)),
            lit,
            slice: lit.is_none().then_some(ArenaSlice {
                offset: 256 * index,
                bytes: 6 * elements as usize * 4,
            }),
        };
        LoweredProgram {
            dims: vec![Dimension {
                symbol: Symbol::from("n"),
                min: 1,
                max: 6,
            }],
            buffers: [
                (buffer(0), record(0, 8, Some(0))),
                (buffer(1), record(1, 4, None)),
                (buffer(2), record(2, 4, None)),
            ]
            .into_iter()
            .collect(),
            kernels: vec![kernel],
            parameters: ArenaSlice {
                offset: 1024,
                bytes: 8,
            },
            staging_parameters: ArenaSlice {
                offset: 0,
                bytes: 8,
            },
            staging_bytes: 8,
            arena_bytes: 2048,
            steps: vec![
                Step::Host(HostRecord {
                    label: "CublasLtOutOfPlace".to_string(),
                    call: HostCall::CublasLt(Box::new(LtCallTemplate {
                        form: CublasLtForm::Base,
                        computation: LtComputation::Float32,
                        reduction_scheme_mask: LtReductionSchemeMask::All,
                        trans_a: false,
                        trans_b: false,
                        a: matrix(n(), 8),
                        b: matrix(IotaExpr::Lit(8), 4),
                        c: None,
                        d: matrix(n(), 4),
                        scale_dtype: PlanDtype::F32,
                        scalars: LtScalars::Host {
                            alpha_bits: 1f32.to_bits(),
                            beta_bits: 0f32.to_bits(),
                        },
                        activation: LtActivation::Identity,
                        bias: None,
                        in_place: None,
                    })),
                    inputs: vec![buffer(0), buffer(0)],
                    dest: buffer(1),
                    workspace: ArenaSlice {
                        offset: 1280,
                        bytes: 512,
                    },
                    capture_dims: Some(vec![Symbol::from("n")]),
                }),
                Step::Kernel(KernelRecord {
                    label: "Relu".to_string(),
                    code: 0,
                    args: vec![
                        ArgRef::Buffer(buffer(1)),
                        ArgRef::Buffer(buffer(2)),
                        ArgRef::Params,
                    ],
                    launch: LaunchRecord::Default { grid_x: 1 },
                }),
                Step::Copy(CopyRecord {
                    src: buffer(2),
                    dst: buffer(1),
                    size: bytes(4),
                    check: CopyCheck::Equal(bytes(4)),
                }),
            ],
            outputs: vec![OutputRecord {
                slot: 0,
                buffer: buffer(1),
                strides: Some(vec![Expr(T::Lit(4)), Expr(T::Lit(1))]),
            }],
        }
    }

    #[test]
    fn a_program_is_written_out_and_read_back_unchanged() {
        let program = program(KernelCode::Ptx("// ptx".to_string()));
        program.ensure_portable().unwrap();
        let bytes = serde_json::to_vec(&program).unwrap();
        let read_back: LoweredProgram = serde_json::from_slice(&bytes).unwrap();
        // The same program is the same bytes.
        assert_eq!(serde_json::to_vec(&read_back).unwrap(), bytes);

        assert_eq!(read_back.schema(), vec![Symbol::from("n")]);
        assert_eq!(read_back.steps.len(), 3);
        assert_eq!(read_back.buffers.len(), 3);
        let dims: DynMap = [(Symbol::from("n"), 5)].into_iter().collect();
        let output = read_back.output(0).unwrap();
        assert_eq!(output.strides_at(&dims).unwrap(), vec![4, 1]);
        let size = read_back.buffer(&output.buffer).unwrap().size.clone();
        assert_eq!(size.unwrap().eval(&dims).unwrap(), 5 * 4 * 4);
        // The host call resolves at a call's dimensions from the data alone.
        let Step::Host(host) = &read_back.steps[0] else {
            panic!("the first step is the host call");
        };
        let HostCall::CublasLt(call) = &host.call else {
            panic!("the host call is a cuBLASLt record");
        };
        let call = call.resolve(&dims);
        assert_eq!((call.m, call.n, call.k), (5, 4, 8));
    }

    #[test]
    fn a_program_that_cannot_be_written_out_is_refused_by_name() {
        let source = program(KernelCode::Source("// source".to_string()));
        let refusal = source.ensure_portable().unwrap_err().to_string();
        assert!(refusal.contains("kernel Relu is still source"), "{refusal}");
        assert!(read_missing_slot(&source).contains("no output slot 3"));
    }

    fn read_missing_slot(program: &LoweredProgram) -> String {
        program.output(3).unwrap_err().to_string()
    }
}
