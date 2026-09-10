//! CUDA kernel operation trait and shared code generation helpers.
//!
//! Each operation implements `KernelOp` and produces CUDA source
//! with fixed dimensions. Generation needs no GPU; `device` compiles and runs it.

use anyhow::{Result, bail};
use luminal::buffer_tensor_ir::BufferTensorIrOp;
use luminal::bufferize::SlotDescriptor;
use luminal::dtype::PlanDtype;
use luminal::index_expr::IotaExpr;
use luminal::layouts::{
    BitOffsetExpressionLayout as BO, DecodedLayout, ElementOffsetExpressionLayout as EO,
    LeftMajorContiguousElementLayout as LM, RightMajorContiguousElementLayout as RM,
    StridedElementLayout as ST,
};

/// Shapes, data types, and read layouts for one compute node.
///
/// Operands follow plan order, with destinations last. Destinations are also
/// listed separately. All metadata comes from the node's [`SlotDescriptor`]
/// layouts, and all reads use [`layout_read_index`].
#[derive(Debug)]
pub struct CodegenCtx {
    pub operand_dims: Vec<Vec<usize>>,
    pub operand_dtypes: Vec<PlanDtype>,
    pub dest_dims: Vec<Vec<usize>>,
    pub dest_dtypes: Vec<PlanDtype>,
    /// Read layouts in the same order as `operand_dims`.
    /// View layouts include the full mapping to the underlying buffer.
    pub operand_layouts: Vec<DecodedLayout>,
}

impl CodegenCtx {
    /// Read shapes and data types from the node's slot layouts.
    /// Return an error for symbolic dimensions or missing data types.
    pub fn from_descriptors(
        label: &str,
        operand_info: &[SlotDescriptor<DecodedLayout>],
        result_info: &[SlotDescriptor<DecodedLayout>],
    ) -> Result<Self> {
        let dims_of = |slot: &SlotDescriptor<DecodedLayout>, role: &str| -> Result<Vec<usize>> {
            slot.layout.literal_extents().ok_or_else(|| {
                anyhow::anyhow!("{label} {role} has symbolic layout extents (no numeric codegen)")
            })
        };
        let dtype_of = |slot: &SlotDescriptor<DecodedLayout>, role: &str| -> Result<PlanDtype> {
            slot.layout
                .dtype
                .ok_or_else(|| anyhow::anyhow!("{label} {role} carries no dtype fact"))
        };
        let dest_dims: Vec<Vec<usize>> = result_info
            .iter()
            .map(|s| dims_of(s, "dest"))
            .collect::<Result<_>>()?;
        // Kernels write `out[i]`. The rules in `ops/*/match_functional.egg`
        // require contiguous row-major destinations; enforce this there, not here.
        // Views write nothing, and cuBLASLt checks its own destination requirements.
        // See `tests/view_admission.rs` for coverage.
        Ok(CodegenCtx {
            operand_dims: operand_info
                .iter()
                .map(|s| dims_of(s, "operand"))
                .collect::<Result<_>>()?,
            operand_dtypes: operand_info
                .iter()
                .map(|s| dtype_of(s, "operand"))
                .collect::<Result<_>>()?,
            dest_dims,
            dest_dtypes: result_info
                .iter()
                .map(|s| dtype_of(s, "dest"))
                .collect::<Result<_>>()?,
            operand_layouts: operand_info.iter().map(|s| s.layout.clone()).collect(),
        })
    }

    /// Return the layout used to compute this operand's read indices.
    pub fn operand_layout(&self, slot: usize) -> &DecodedLayout {
        &self.operand_layouts[slot]
    }
}

// Reads use each slot's layout, including any view mappings.
// Generated kernels do not check bounds, bit alignment, or duplicate scatter
// indices. Invalid accesses cause undefined behavior; use `compute-sanitizer`
// to debug them. Any future runtime checks should be enabled by a feature flag.

// Simplify read offset expressions, regardless of how the layout is represented.
// For row-major coordinates derived from `i`, an equivalent offset becomes `i`.

/// An offset of the form `constant + Σ coeffs[axis] * c{axis}`.
/// There is one coefficient per axis, numbered from the first dimension.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Affine {
    constant: i64,
    coeffs: Vec<i64>,
}

impl Affine {
    fn zero(rank: usize) -> Self {
        Affine {
            constant: 0,
            coeffs: vec![0; rank],
        }
    }

    fn constant(v: i64, rank: usize) -> Self {
        Affine {
            constant: v,
            coeffs: vec![0; rank],
        }
    }

    /// The coordinate `c{axis}`, with axes numbered from the first dimension.
    fn coord(axis: usize, rank: usize) -> Self {
        let mut coeffs = vec![0; rank];
        coeffs[axis] = 1;
        Affine {
            constant: 0,
            coeffs,
        }
    }

    /// Build `Σ strides[axis] * c{axis}` with a zero constant.
    fn from_strides(strides: &[usize]) -> Option<Self> {
        Some(Affine {
            constant: 0,
            coeffs: strides
                .iter()
                .map(|&s| i64::try_from(s).ok())
                .collect::<Option<_>>()?,
        })
    }

    /// Return the constant if every coordinate coefficient is zero.
    fn as_constant(&self) -> Option<i64> {
        self.coeffs.iter().all(|&c| c == 0).then_some(self.constant)
    }

    /// Add two offsets. Return `None` on overflow so codegen uses the
    /// original expression.
    fn add(self, other: Self) -> Option<Self> {
        Some(Affine {
            constant: self.constant.checked_add(other.constant)?,
            coeffs: self
                .coeffs
                .iter()
                .zip(&other.coeffs)
                .map(|(a, b)| a.checked_add(*b))
                .collect::<Option<_>>()?,
        })
    }

    fn scale(self, k: i64) -> Option<Self> {
        Some(Affine {
            constant: self.constant.checked_mul(k)?,
            coeffs: self
                .coeffs
                .iter()
                .map(|c| c.checked_mul(k))
                .collect::<Option<_>>()?,
        })
    }

    /// Divide every term by `k` without rounding.
    /// Return `None` if any term is not divisible by `k`, or division overflows.
    fn exact_div(self, k: i64) -> Option<Self> {
        if k == 0 || self.constant % k != 0 || self.coeffs.iter().any(|c| c % k != 0) {
            return None;
        }
        Some(Affine {
            constant: self.constant.checked_div(k)?,
            coeffs: self
                .coeffs
                .iter()
                .map(|c| c.checked_div(k))
                .collect::<Option<_>>()?,
        })
    }
}

/// Try to express an integer expression as a constant plus weighted coordinates.
/// Return `None` for unsupported expressions or overflow; codegen then uses
/// the original expression.
fn affine_of_term(expr: &luminal::layouts::IntExprTerm, rank: usize) -> Option<Affine> {
    use luminal::layouts::IntExprTerm as T;
    match expr {
        T::Lit(v) => Some(Affine::constant(*v, rank)),
        T::Var(_) => None,
        T::Coord { axis_from_end } => {
            let axis = usize::try_from(*axis_from_end).ok().filter(|&a| a < rank)?;
            Some(Affine::coord(rank - 1 - axis, rank))
        }
        T::Add(a, b) => affine_of_term(a, rank)?.add(affine_of_term(b, rank)?),
        T::Mul(a, b) => {
            let (a, b) = (affine_of_term(a, rank)?, affine_of_term(b, rank)?);
            match (a.as_constant(), b.as_constant()) {
                (Some(k), _) => b.scale(k),
                (_, Some(k)) => a.scale(k),
                // Multiplying two coordinate-dependent expressions cannot be simplified here.
                _ => None,
            }
        }
        T::TruncDiv(a, b) => {
            let k = affine_of_term(b, rank)?.as_constant()?;
            affine_of_term(a, rank)?.exact_div(k)
        }
        T::TruncRem(_, _)
        | T::CeilDiv(_, _)
        | T::Min(_, _)
        | T::Max(_, _)
        | T::LessThanCast(_, _) => None,
    }
}

/// Try to express the layout's read offset as a constant plus weighted coordinates.
/// The layout must have fixed dimensions equal to `dims`. Prefer contiguous
/// layouts, then strided layouts, then explicit offset expressions.
fn read_affine(layout: &DecodedLayout, dims: &[usize]) -> Option<Affine> {
    let rank = dims.len();
    if layout.literal_extents().as_deref() != Some(dims) {
        return None;
    }
    // Contiguous layouts provide strides directly.
    if layout.has::<RM>() {
        Affine::from_strides(&strides_of(dims))
    } else if layout.has::<LM>() {
        let mut strides = vec![1usize; rank];
        for axis in 1..rank {
            strides[axis] = strides[axis - 1] * dims[axis - 1];
        }
        Affine::from_strides(&strides)
    // Other layouts provide offset expressions.
    } else if let Some(st) = layout.first::<ST>() {
        st.chain.iter().try_fold(Affine::zero(rank), |acc, s| {
            acc.add(affine_of_term(s, rank)?)
        })
    } else if let Some(eo) = layout.first::<EO>() {
        affine_of_term(&eo.offset, rank)
    } else if let Some(bo) = layout.first::<BO>() {
        affine_of_term(&bo.offset, rank)?.exact_div(bo.width.0)
    } else {
        None
    }
}

/// Convert a layout integer expression to C using `long long`.
/// Coordinates use `{prefix}{axis}`, numbered from the first dimension.
/// Return an error for symbolic variables or invalid coordinate axes.
fn lower_layout_term(
    expr: &luminal::layouts::IntExprTerm,
    rank: usize,
    prefix: &str,
) -> Result<String> {
    use luminal::layouts::IntExprTerm as T;
    let rec = |e: &T| lower_layout_term(e, rank, prefix);
    Ok(match expr {
        T::Lit(v) => format!("{v}LL"),
        T::Var(name) => bail!("layout read: symbolic dim `{name}` has no numeric codegen"),
        T::Coord { axis_from_end } => {
            let axis = usize::try_from(*axis_from_end)
                .ok()
                .filter(|&a| a < rank)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "layout read: coordinate axis {axis_from_end} out of rank {rank}"
                    )
                })?;
            format!("{prefix}{}", rank - 1 - axis)
        }
        T::Add(a, b) => format!("({} + {})", rec(a)?, rec(b)?),
        T::Mul(a, b) => format!("({} * {})", rec(a)?, rec(b)?),
        T::TruncDiv(a, b) => format!("({} / {})", rec(a)?, rec(b)?),
        T::TruncRem(a, b) => format!("({} % {})", rec(a)?, rec(b)?),
        T::CeilDiv(a, b) => {
            // CeilDiv is unsupported until its behavior for negative operands is defined.
            let (_, _) = (rec(a)?, rec(b)?);
            bail!("layout read: IntCeilDiv lowering not implemented (fail-closed)")
        }
        T::Min(a, b) => {
            let (a, b) = (rec(a)?, rec(b)?);
            format!("(({a}) < ({b}) ? ({a}) : ({b}))")
        }
        T::Max(a, b) => {
            let (a, b) = (rec(a)?, rec(b)?);
            format!("(({a}) > ({b}) ? ({a}) : ({b}))")
        }
        T::LessThanCast(a, b) => {
            format!("(({}) < ({}) ? 1LL : 0LL)", rec(a)?, rec(b)?)
        }
    })
}

/// How the caller computed the coordinates used by [`layout_read_index`].
///
/// `FlatIndex` means row-major coordinates derived from `i` over `slot_dims`;
/// a matching layout offset can simplify to `i`. `Bound` means independently
/// computed coordinates, so codegen keeps the full offset expression.
#[derive(Debug, Clone, Copy)]
pub enum Coords<'a> {
    FlatIndex { prefix: &'a str },
    Bound { prefix: &'a str },
}

impl<'a> Coords<'a> {
    fn prefix(&self) -> &'a str {
        match self {
            Coords::FlatIndex { prefix } | Coords::Bound { prefix } => prefix,
        }
    }
}

/// Generate C code and an element index for reading an operand's layout.
/// Coordinates use the supplied prefix, with axes numbered from the first dimension.
/// The layout must have fixed dimensions equal to `slot_dims`.
///
/// Return `(code, index_expr)`. If a `FlatIndex` offset simplifies to `i`,
/// return empty code and `"i"`. No runtime bounds checks are generated.
pub fn layout_read_index(
    operand: &str,
    layout: &DecodedLayout,
    slot_dims: &[usize],
    coords: Coords<'_>,
) -> Result<(String, String)> {
    // If the offset equals the row-major index used to compute these
    // coordinates, replace it with `i`.
    if let Coords::FlatIndex { .. } = coords
        && let Some(affine) = read_affine(layout, slot_dims)
    {
        let strides = strides_of(slot_dims);
        let is_flat_index = affine.constant == 0
            && (0..slot_dims.len()).all(|axis| {
                // A size-one axis always has coordinate zero, so its coefficient
                // can be ignored.
                slot_dims[axis] == 1 || i64::try_from(strides[axis]) == Ok(affine.coeffs[axis])
            });
        if is_flat_index {
            return Ok((String::new(), "i".to_string()));
        }
    }
    let in_prefix = coords.prefix();
    let rank = slot_dims.len();
    let idx = format!("{operand}_idx");
    let check_domain = |shape: &luminal::layouts::ShapeTerm| -> Result<()> {
        let extents: Option<Vec<usize>> = shape
            .0
            .iter()
            .map(|e| e.eval_literal().and_then(|v| usize::try_from(v).ok()))
            .collect();
        let Some(extents) = extents else {
            bail!("operand {operand}: layout has symbolic extents (no numeric codegen)");
        };
        if extents != slot_dims {
            bail!(
                "operand {operand}: layout domain {extents:?} differs from the slot's \
                 value extents {slot_dims:?} — refuse, never reinterpret"
            );
        }
        Ok(())
    };
    // Use the same layout preference as `read_affine`.
    // Return an error if no supported representation is available.
    let offset: String = if let Some(rm) = layout.first::<RM>() {
        check_domain(&rm.shape)?;
        let strides = strides_of(slot_dims);
        if rank == 0 {
            "0LL".to_string()
        } else {
            (0..rank)
                .map(|axis| format!("{in_prefix}{axis} * {}LL", strides[axis]))
                .collect::<Vec<_>>()
                .join(" + ")
        }
    } else if let Some(lm) = layout.first::<LM>() {
        check_domain(&lm.shape)?;
        let mut strides = vec![1usize; rank];
        for axis in 1..rank {
            strides[axis] = strides[axis - 1] * slot_dims[axis - 1];
        }
        if rank == 0 {
            "0LL".to_string()
        } else {
            (0..rank)
                .map(|axis| format!("{in_prefix}{axis} * {}LL", strides[axis]))
                .collect::<Vec<_>>()
                .join(" + ")
        }
    } else if let Some(st) = layout.first::<ST>() {
        check_domain(&st.shape)?;
        let summands = st
            .chain
            .iter()
            .map(|s| lower_layout_term(s, rank, in_prefix))
            .collect::<Result<Vec<_>>>()?;
        if summands.is_empty() {
            "0LL".to_string()
        } else {
            summands.join(" + ")
        }
    } else if let Some(eo) = layout.first::<EO>() {
        check_domain(&eo.shape)?;
        lower_layout_term(&eo.offset, rank, in_prefix)?
    } else if let Some(bo) = layout.first::<BO>() {
        check_domain(&bo.shape)?;
        let bits = lower_layout_term(&bo.offset, rank, in_prefix)?;
        let width = bo.width.0;
        // Convert the bit offset to an element index. The compiler must
        // ensure the offset is divisible by the element width.
        let bits_var = format!("{operand}_bits");
        let code = format!(
            "    long long {bits_var} = {bits};\n    long long {idx} = {bits_var} / {width}LL;\n"
        );
        return Ok((code, idx));
    } else {
        bail!(
            "operand {operand}: no lowerable layout spelling — the elected class \
             holds {:?}",
            layout.present()
        );
    };
    Ok((format!("    long long {idx} = {offset};\n"), idx))
}

/// CUDA source for one launch of kernel `k`, with `n` threads.
/// Arguments are the operation's inputs, followed by `out` and `n`.
///
/// `launch` overrides the executor's default geometry (one thread per
/// element in 256-thread blocks): a kernel that organizes its own
/// cooperation — the block-per-output reductions — names its grid and
/// block, and `n` is then whatever the kernel's `n` argument means to it.
#[derive(Debug)]
pub struct KernelSource {
    pub source: String,
    pub n: usize,
    pub launch: Option<LaunchGeometry>,
    /// Skip this launch when the destination IS operand `k`'s storage —
    /// an in-place op whose first launch would copy a buffer onto
    /// itself (scatter's init→dest copy).
    pub skip_when_dest_is_operand: Option<usize>,
}

/// An explicit 1-D launch geometry.
#[derive(Debug, Clone, Copy)]
pub struct LaunchGeometry {
    pub grid: u32,
    pub block: u32,
}

impl KernelSource {
    pub(crate) fn plain(source: String, n: usize) -> Self {
        Self {
            source,
            n,
            launch: None,
            skip_when_dest_is_operand: None,
        }
    }

    pub(crate) fn with_launch(source: String, n: usize, grid: u32, block: u32) -> Self {
        Self {
            source,
            n,
            launch: Some(LaunchGeometry { grid, block }),
            skip_when_dest_is_operand: None,
        }
    }
}

/// Reductions whose reduced extent is at least this long and whose
/// outputs are few run one BLOCK per output element (256 threads striding
/// the axis, then a tree reduction) instead of one thread per output
/// looping the whole axis — the difference between a 20 ms and a 20 µs
/// argmax over a 200k-token vocabulary.
const BLOCK_REDUCE_MIN_EXTENT: usize = 512;
const BLOCK_REDUCE_THREADS: usize = 256;

/// An operation's code generator. Its kernels run in order on one stream;
/// for example, scatter copies the input before writing updates.
/// Generation needs no CUDA device and uses the op's own metadata directly.
pub trait KernelOp: BufferTensorIrOp {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>>;
}

/// Return the CUDA scalar type, or an error for unsupported data types.
pub(crate) fn cuda_type(dtype: PlanDtype) -> Result<&'static str> {
    Ok(match dtype {
        PlanDtype::F32 => "float",
        PlanDtype::Int => "int",
        PlanDtype::Int64 => "long long",
        PlanDtype::Bool | PlanDtype::Bool8 => "unsigned char",
        other => bail!("cuda-lite CL-1 has no device type for {other:?}"),
    })
}

/// Format a number as a CUDA expression accepted by NVRTC.
/// Finite values use scientific notation so C parses them as floating point.
/// NaN and infinities use float bit patterns because NVRTC lacks host math
/// headers. Callers cast the result to the destination type.
pub(crate) fn cuda_f64_literal(v: f64) -> String {
    if v.is_nan() {
        return "__uint_as_float(0x7fc00000u)".to_string();
    }
    if v.is_infinite() {
        return if v.is_sign_positive() {
            "__uint_as_float(0x7f800000u)".to_string()
        } else {
            "__uint_as_float(0xff800000u)".to_string()
        };
    }
    format!("{v:e}")
}

pub(crate) fn numel(dims: &[usize]) -> usize {
    dims.iter().product()
}

/// Generate a two-input elementwise kernel, reading each input through its layout.
pub(crate) fn binary(ctx: &CodegenCtx, expr: &str) -> Result<Vec<KernelSource>> {
    let [a, b, _dest] = ctx.operand_dtypes.as_slice() else {
        bail!(
            "binary op expects two operands + dest, got {}",
            ctx.operand_dtypes.len()
        );
    };
    let (ta, tb) = (cuda_type(*a)?, cuda_type(*b)?);
    let to = cuda_type(ctx.dest_dtypes[0])?;
    let sig = format!("const {ta}* a, const {tb}* b");
    elementwise(ctx, expr, &["a", "b"], &sig, to)
}

/// Generate a one-input elementwise kernel, reading the input through its layout.
pub(crate) fn unary(ctx: &CodegenCtx, expr: &str) -> Result<Vec<KernelSource>> {
    let ta = cuda_type(ctx.operand_dtypes[0])?;
    let to = cuda_type(ctx.dest_dtypes[0])?;
    let sig = format!("const {ta}* a");
    elementwise(ctx, expr, &["a"], &sig, to)
}

// BufferCopy copies whole buffers. Layout conversions use materialize kernels
// selected by egglog rules.

/// Generate one thread per output element, reading inputs through their layouts.
/// The template must refer to each input as `name[i]`; these tokens are replaced
/// with the indices from [`layout_read_index`]. Omit coordinate calculations
/// when every read simplifies to `name[i]`.
///
/// Inputs must match the output shape. Egglog rules require contiguous
/// row-major destinations.
fn elementwise(
    ctx: &CodegenCtx,
    expr: &str,
    names: &[&str],
    sig: &str,
    to: &str,
) -> Result<Vec<KernelSource>> {
    let out_dims = &ctx.dest_dims[0];
    let n = numel(out_dims);
    // Every input must match the output shape, regardless of its layout.
    // Destination layout requirements are enforced by egglog rules.
    for (k, name) in names.iter().enumerate() {
        if &ctx.operand_dims[k] != out_dims {
            bail!(
                "operand {name} value extents {:?} differ from dest extents {:?} — \
                 elementwise templates iterate the dest; refuse, never reinterpret",
                ctx.operand_dims[k],
                out_dims
            );
        }
    }
    let mut chains = String::new();
    let mut rendered = expr.to_string();
    for (k, name) in names.iter().enumerate() {
        // Replace each input read with its layout index. When the index
        // simplifies to `i`, the read stays unchanged.
        let layout = ctx.operand_layout(k);
        let (code, idx) =
            layout_read_index(name, layout, out_dims, Coords::FlatIndex { prefix: "c" })?;
        chains.push_str(&code);
        let flat = format!("{name}[i]");
        if !rendered.contains(&flat) {
            bail!("template expr `{expr}` has no `{flat}` token to rewrite for a composed operand");
        }
        rendered = rendered.replace(&flat, &format!("{name}[{idx}]"));
    }
    if chains.is_empty() {
        // All reads use `i`, so no coordinate calculations are needed.
        let source = format!(
            r#"extern "C" __global__ void k({sig}, {to}* out, unsigned long long n) {{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = {rendered};
}}"#
        );
        return Ok(vec![KernelSource::plain(source, n)]);
    }
    let prelude = coord_prelude(out_dims);
    let source = format!(
        r#"extern "C" __global__ void k({sig}, {to}* out, unsigned long long n) {{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
{prelude}{chains}    out[i] = {rendered};
}}"#
    );
    Ok(vec![KernelSource::plain(source, n)])
}

/// Generate a reduction with one thread per output element, looping over
/// the reduced axis. Axis 0 is the last input dimension.
pub(crate) fn reduce(
    ctx: &CodegenCtx,
    axis_from_end: usize,
    init: &str,
    fold: &str,
) -> Result<Vec<KernelSource>> {
    let in_dims = &ctx.operand_dims[0];
    let ta = cuda_type(ctx.operand_dtypes[0])?;
    let to = cuda_type(ctx.dest_dtypes[0])?;
    if axis_from_end >= in_dims.len() {
        bail!("reduce axis {axis_from_end} out of rank {}", in_dims.len());
    }
    let axis = in_dims.len() - 1 - axis_from_end;
    let extent = in_dims[axis];
    // Count the input elements before and after the reduced axis.
    let inner: usize = in_dims[axis + 1..].iter().product();
    let outer: usize = in_dims[..axis].iter().product();
    let n = outer * inner;
    // Input coordinates combine the output position with the reduction
    // loop index. Use `Coords::Bound`: the input offset cannot simplify
    // to `i`, which indexes the smaller output shape.
    let layout = ctx.operand_layout(0);
    // Compute coordinates outside the reduced axis once before the loop.
    // The loop variable supplies `c{axis}`.
    let mut coords = String::from("    unsigned long long rem = inner;\n");
    for ax in ((axis + 1)..in_dims.len()).rev() {
        coords.push_str(&format!(
            "    long long c{ax} = (long long)(rem % {d}ULL); rem /= {d}ULL;\n",
            d = in_dims[ax]
        ));
    }
    coords.push_str("    rem = outer;\n");
    for ax in (0..axis).rev() {
        coords.push_str(&format!(
            "    long long c{ax} = (long long)(rem % {d}ULL); rem /= {d}ULL;\n",
            d = in_dims[ax]
        ));
    }
    let (chain, idx) = layout_read_index("a", layout, in_dims, Coords::Bound { prefix: "c" })?;
    // Indent the generated index code inside the loop.
    let chain = chain.replace("    ", "        ");
    if inner == 1 && extent >= BLOCK_REDUCE_MIN_EXTENT && n <= 1 << 20 {
        // BLOCK-PER-OUTPUT: block i owns output i; its threads stride the
        // reduced axis, then fold their partials through shared memory.
        // The fold template is applied to (acc, v) exactly as in the loop.
        let threads = BLOCK_REDUCE_THREADS;
        let source = format!(
            r#"extern "C" __global__ void k(const {ta}* a, {to}* out, unsigned long long n) {{
    __shared__ {ta} partial[{threads}];
    unsigned long long i = blockIdx.x;
    if (i >= n) return;
    unsigned long long outer = i / {inner}ULL;
    unsigned long long inner = i % {inner}ULL;
{coords}    {ta} acc = {init};
    for (unsigned long long r = threadIdx.x; r < {extent}ULL; r += {threads}ULL) {{
        long long c{axis} = (long long)r;
{chain}        {ta} v = a[{idx}];
        acc = {fold};
    }}
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned int stride = {threads} / 2; stride > 0; stride >>= 1) {{
        if (threadIdx.x < stride) {{
            {ta} v = partial[threadIdx.x + stride];
            acc = {fold};
            partial[threadIdx.x] = acc;
        }}
        __syncthreads();
    }}
    if (threadIdx.x == 0) out[i] = partial[0];
}}"#
        );
        return Ok(vec![KernelSource::with_launch(
            source,
            n,
            n.max(1) as u32,
            threads as u32,
        )]);
    }
    let source = format!(
        r#"extern "C" __global__ void k(const {ta}* a, {to}* out, unsigned long long n) {{
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long outer = i / {inner}ULL;
    unsigned long long inner = i % {inner}ULL;
{coords}    {ta} acc = {init};
    for (unsigned long long r = 0; r < {extent}ULL; ++r) {{
        long long c{axis} = (long long)r;
{chain}        {ta} v = a[{idx}];
        acc = {fold};
    }}
    out[i] = acc;
}}"#
    );
    Ok(vec![KernelSource::plain(source, n)])
}

/// Convert an [`IotaExpr`] to C using `long long` and coordinates `c0..c{rank-1}`.
/// `Coord(axis_from_end)` reads `c{rank-1-axis_from_end}`.
pub(crate) fn lower_expr(expr: &IotaExpr, rank: usize) -> Result<String> {
    lower_expr_pref(expr, rank, "c")
}

/// Like [`lower_expr`], with a caller-supplied coordinate prefix.
/// `Coord(axis_from_end)` reads `{prefix}{rank-1-axis_from_end}`.
pub(crate) fn lower_expr_pref(expr: &IotaExpr, rank: usize, prefix: &str) -> Result<String> {
    let rec = |e: &IotaExpr| lower_expr_pref(e, rank, prefix);
    Ok(match expr {
        IotaExpr::Lit(v) => format!("{v}LL"),
        IotaExpr::Coord(axis_from_end) => {
            if *axis_from_end >= rank {
                bail!("coordinate axis {axis_from_end} out of rank {rank}");
            }
            format!("{prefix}{}", rank - 1 - axis_from_end)
        }
        IotaExpr::Add(a, b) => format!("({} + {})", rec(a)?, rec(b)?),
        IotaExpr::Mul(a, b) => format!("({} * {})", rec(a)?, rec(b)?),
        IotaExpr::TruncDiv(a, b) => {
            format!("({} / {})", rec(a)?, rec(b)?)
        }
        IotaExpr::TruncRem(a, b) => {
            format!("({} % {})", rec(a)?, rec(b)?)
        }
        IotaExpr::Min(a, b) => {
            let (a, b) = (rec(a)?, rec(b)?);
            format!("(({a}) < ({b}) ? ({a}) : ({b}))")
        }
        IotaExpr::Max(a, b) => {
            let (a, b) = (rec(a)?, rec(b)?);
            format!("(({a}) > ({b}) ? ({a}) : ({b}))")
        }
        IotaExpr::LessThanCast(a, b) => {
            format!("(({}) < ({}) ? 1LL : 0LL)", rec(a)?, rec(b)?)
        }
    })
}

/// Generate row-major coordinates `c0..c{rank-1}` from flat index `i`.
pub(crate) fn coord_prelude(dims: &[usize]) -> String {
    let mut out = String::from("    unsigned long long rem = i;\n");
    for axis in (0..dims.len()).rev() {
        out.push_str(&format!(
            "    long long c{axis} = (long long)(rem % {}ULL); rem /= {}ULL;\n",
            dims[axis], dims[axis]
        ));
    }
    out
}

/// Return row-major strides for `dims`.
pub(crate) fn strides_of(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; dims.len()];
    for k in (0..dims.len().saturating_sub(1)).rev() {
        strides[k] = strides[k + 1] * dims[k + 1];
    }
    strides
}
