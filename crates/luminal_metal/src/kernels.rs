//! Metal Shading Language code generation over elected slot layouts.
//! Reads follow composed layouts; egglog match rules require dense destinations.

use crate::symbolic::Expr;
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

#[derive(Debug)]
pub struct CodegenCtx {
    pub operand_dims: Vec<Vec<Expr>>,
    pub operand_dtypes: Vec<PlanDtype>,
    pub dest_dims: Vec<Vec<Expr>>,
    pub dest_dtypes: Vec<PlanDtype>,
    pub operand_layouts: Vec<DecodedLayout>,
}

impl CodegenCtx {
    pub fn from_descriptors(
        label: &str,
        operand_info: &[SlotDescriptor<DecodedLayout>],
        result_info: &[SlotDescriptor<DecodedLayout>],
    ) -> Result<Self> {
        let dims_of = |slot: &SlotDescriptor<DecodedLayout>, _role: &str| -> Result<Vec<Expr>> {
            Ok(slot.layout.shape().0.iter().cloned().map(Expr).collect())
        };
        let dtype_of = |slot: &SlotDescriptor<DecodedLayout>, role: &str| -> Result<PlanDtype> {
            slot.layout
                .dtype
                .ok_or_else(|| anyhow::anyhow!("{label} {role} carries no dtype fact"))
        };
        let dest_dims: Vec<Vec<Expr>> = result_info
            .iter()
            .map(|s| dims_of(s, "dest"))
            .collect::<Result<_>>()?;
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

    pub fn operand_layout(&self, slot: usize) -> &DecodedLayout {
        &self.operand_layouts[slot]
    }
}

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

    fn coord(axis: usize, rank: usize) -> Self {
        let mut coeffs = vec![0; rank];
        coeffs[axis] = 1;
        Affine {
            constant: 0,
            coeffs,
        }
    }

    fn from_strides(strides: &[usize]) -> Option<Self> {
        Some(Affine {
            constant: 0,
            coeffs: strides
                .iter()
                .map(|&s| i64::try_from(s).ok())
                .collect::<Option<_>>()?,
        })
    }

    fn as_constant(&self) -> Option<i64> {
        self.coeffs.iter().all(|&c| c == 0).then_some(self.constant)
    }

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

fn read_affine(layout: &DecodedLayout, dims: &[usize]) -> Option<Affine> {
    let rank = dims.len();
    if layout.literal_extents().as_deref() != Some(dims) {
        return None;
    }
    if layout.has::<RM>() {
        Affine::from_strides(&literal_strides(dims))
    } else if layout.has::<LM>() {
        let mut strides = vec![1usize; rank];
        for axis in 1..rank {
            strides[axis] = strides[axis - 1] * dims[axis - 1];
        }
        Affine::from_strides(&strides)
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

fn lower_layout_term(
    expr: &luminal::layouts::IntExprTerm,
    rank: usize,
    prefix: &str,
) -> Result<String> {
    use luminal::layouts::IntExprTerm as T;
    let rec = |e: &T| lower_layout_term(e, rank, prefix);
    Ok(match expr {
        T::Lit(v) => crate::symbolic::integer_literal(*v),
        T::Var(name) => crate::symbolic::variable(name),
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
        T::CeilDiv(a, b) => format!("luminal_ceil_div({}, {})", rec(a)?, rec(b)?),
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

pub fn layout_read_index(
    operand: &str,
    layout: &DecodedLayout,
    slot_dims: &[Expr],
    coords: Coords<'_>,
) -> Result<(String, String)> {
    if matches!(coords, Coords::FlatIndex { .. })
        && layout.has::<RM>()
        && layout.shape().0 == slot_dims.iter().map(|e| e.0.clone()).collect::<Vec<_>>()
    {
        return Ok((String::new(), "i".into()));
    }
    if let Coords::FlatIndex { .. } = coords
        && let Some(literals) = slot_dims
            .iter()
            .map(Expr::literal)
            .collect::<Option<Vec<_>>>()
        && let Some(affine) = read_affine(layout, &literals)
    {
        let strides = literal_strides(&literals);
        let is_flat_index = affine.constant == 0
            && (0..slot_dims.len()).all(|axis| {
                literals[axis] == 1 || i64::try_from(strides[axis]) == Ok(affine.coeffs[axis])
            });
        if is_flat_index {
            return Ok((String::new(), "i".to_string()));
        }
    }
    let in_prefix = coords.prefix();
    let rank = slot_dims.len();
    let idx = format!("{operand}_idx");
    let check_domain = |shape: &luminal::layouts::ShapeTerm| -> Result<()> {
        let extents: Vec<Expr> = shape.0.iter().cloned().map(Expr).collect();
        if extents != slot_dims {
            bail!(
                "operand {operand}: layout domain {extents:?} differs from the slot's \
                 value extents {slot_dims:?} — refuse, never reinterpret"
            );
        }
        Ok(())
    };
    let offset: String = if let Some(rm) = layout.first::<RM>() {
        check_domain(&rm.shape)?;
        let strides = strides_of(slot_dims);
        if rank == 0 {
            "0LL".to_string()
        } else {
            (0..rank)
                .map(|axis| format!("{in_prefix}{axis} * {}", strides[axis]))
                .collect::<Vec<_>>()
                .join(" + ")
        }
    } else if let Some(lm) = layout.first::<LM>() {
        check_domain(&lm.shape)?;
        let mut strides = vec![Expr::from(1usize); rank];
        for axis in 1..rank {
            strides[axis] = strides[axis - 1].clone() * slot_dims[axis - 1].clone();
        }
        if rank == 0 {
            "0LL".to_string()
        } else {
            (0..rank)
                .map(|axis| format!("{in_prefix}{axis} * {}", strides[axis]))
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
        let bits_var = format!("{operand}_bits");
        let code =
            format!("    long {bits_var} = {bits};\n    long {idx} = {bits_var} / {width};\n");
        return Ok((code, idx));
    } else {
        bail!(
            "operand {operand}: no lowerable layout spelling — the elected class \
             holds {:?}",
            layout.present()
        );
    };
    Ok((format!("    long {idx} = {offset};\n"), idx))
}

#[derive(Debug)]
/// MSL function `k`: DPS operand pointers in slot order, followed by a
/// `device const long* params` argument. Dimensions use `symbolic::variable`.
pub struct KernelSource {
    pub source: String,
    pub n: Expr,
    pub launch: Option<KernelLaunch>,
}

impl KernelSource {
    pub fn plain(source: String, n: Expr) -> Self {
        Self {
            source,
            n,
            launch: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct KernelLaunch {
    pub grid: [Expr; 3],
    pub block: [Expr; 3],
    pub shared_bytes: Expr,
}
impl KernelLaunch {
    pub fn linear(n: Expr, block: usize) -> Self {
        Self {
            grid: [
                Expr(luminal::layouts::IntExprTerm::CeilDiv(
                    Box::new(n.0),
                    Box::new(Expr::from(block).0),
                )),
                1usize.into(),
                1usize.into(),
            ],
            block: [block.into(), 1usize.into(), 1usize.into()],
            shared_bytes: 0usize.into(),
        }
    }
}

/// Generate execution directly from the operation carried by the buffer plan.
pub trait KernelOp: BufferTensorIrOp {
    fn codegen(&self, ctx: &CodegenCtx) -> Result<Vec<KernelSource>>;
}

pub(crate) fn metal_type(dtype: PlanDtype) -> Result<&'static str> {
    Ok(match dtype {
        PlanDtype::F32 => "float",
        PlanDtype::F16 => "half",
        PlanDtype::Int => "int",
        PlanDtype::Int64 => "long",
        PlanDtype::Bool | PlanDtype::Bool8 => "unsigned char",
        other => bail!("Metal has no device type for {other:?}"),
    })
}

pub(crate) fn metal_f64_literal(v: f64) -> String {
    if v.is_nan() {
        return "as_type<float>(0x7fc00000u)".to_string();
    }
    if v.is_infinite() {
        return if v.is_sign_positive() {
            "as_type<float>(0x7f800000u)".to_string()
        } else {
            "as_type<float>(0xff800000u)".to_string()
        };
    }
    format!("{v:e}")
}

pub(crate) fn numel(dims: &[Expr]) -> Expr {
    dims.iter().product()
}

pub(crate) fn binary(ctx: &CodegenCtx, expr: &str) -> Result<Vec<KernelSource>> {
    let [a, b, _dest] = ctx.operand_dtypes.as_slice() else {
        bail!(
            "binary op expects two operands + dest, got {}",
            ctx.operand_dtypes.len()
        );
    };
    let (ta, tb) = (metal_type(*a)?, metal_type(*b)?);
    let to = metal_type(ctx.dest_dtypes[0])?;
    let sig = format!("device const {ta}* a, device const {tb}* b");
    elementwise(ctx, expr, &["a", "b"], &sig, to)
}

pub(crate) fn unary(ctx: &CodegenCtx, expr: &str) -> Result<Vec<KernelSource>> {
    let ta = metal_type(ctx.operand_dtypes[0])?;
    let to = metal_type(ctx.dest_dtypes[0])?;
    let sig = format!("device const {ta}* a");
    elementwise(ctx, expr, &["a"], &sig, to)
}

fn elementwise(
    ctx: &CodegenCtx,
    expr: &str,
    names: &[&str],
    sig: &str,
    to: &str,
) -> Result<Vec<KernelSource>> {
    let out_dims = &ctx.dest_dims[0];
    let n = numel(out_dims);
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
        let source = format!(
            r#"kernel void k({sig}, device {to}* out, device const long* params, uint gid [[thread_position_in_grid]]) {{
    const ulong n = {n};
    ulong i = gid;
    if (i < n) out[i] = {rendered};
}}"#
        );
        return Ok(vec![KernelSource::plain(source, n)]);
    }
    let prelude = coord_prelude(out_dims);
    let source = format!(
        r#"kernel void k({sig}, device {to}* out, device const long* params, uint gid [[thread_position_in_grid]]) {{
    const ulong n = {n};
    ulong i = gid;
    if (i >= n) return;
{prelude}{chains}    out[i] = {rendered};
}}"#
    );
    Ok(vec![KernelSource::plain(source, n)])
}

pub(crate) fn reduce(
    ctx: &CodegenCtx,
    axis_from_end: usize,
    init: &str,
    fold: &str,
) -> Result<Vec<KernelSource>> {
    reduce_impl(ctx, axis_from_end, init, fold, false)
}

pub(crate) fn reduce_product(ctx: &CodegenCtx, axis_from_end: usize) -> Result<Vec<KernelSource>> {
    reduce_impl(ctx, axis_from_end, "0.0f", "acc + v", true)
}

fn reduce_impl(
    ctx: &CodegenCtx,
    axis_from_end: usize,
    init: &str,
    fold: &str,
    product: bool,
) -> Result<Vec<KernelSource>> {
    let in_dims = &ctx.operand_dims[0];
    let ta = metal_type(ctx.operand_dtypes[0])?;
    let to = metal_type(ctx.dest_dtypes[0])?;
    if axis_from_end >= in_dims.len() {
        bail!("reduce axis {axis_from_end} out of rank {}", in_dims.len());
    }
    let axis = in_dims.len() - 1 - axis_from_end;
    let extent = in_dims[axis].clone();
    let inner: Expr = in_dims[axis + 1..].iter().product();
    let outer: Expr = in_dims[..axis].iter().product();
    let n = outer * inner.clone();
    let layout = ctx.operand_layout(0);
    let mut coords = String::from("    ulong rem = inner;\n");
    for ax in ((axis + 1)..in_dims.len()).rev() {
        coords.push_str(&format!(
            "    long c{ax} = (long)(rem % {d}); rem /= {d};\n",
            d = in_dims[ax]
        ));
    }
    coords.push_str("    rem = outer;\n");
    for ax in (0..axis).rev() {
        coords.push_str(&format!(
            "    long c{ax} = (long)(rem % {d}); rem /= {d};\n",
            d = in_dims[ax]
        ));
    }
    let (chain, idx) = layout_read_index("a", layout, in_dims, Coords::Bound { prefix: "c" })?;
    let mut chain = chain.replace("    ", "        ");
    let (signature, value) = if product {
        anyhow::ensure!(
            ctx.operand_dims[1] == *in_dims,
            "dot operands have different shapes"
        );
        let tb = metal_type(ctx.operand_dtypes[1])?;
        let (other_chain, other_idx) = layout_read_index(
            "b",
            ctx.operand_layout(1),
            in_dims,
            Coords::Bound { prefix: "c" },
        )?;
        chain.push_str(&other_chain.replace("    ", "        "));
        (
            format!("device const {ta}* a, device const {tb}* b"),
            format!("a[{idx}] * b[{other_idx}]"),
        )
    } else {
        (format!("device const {ta}* a"), format!("a[{idx}]"))
    };
    let source = format!(
        r#"kernel void k({signature}, device {to}* out, device const long* params, uint gid [[thread_position_in_grid]]) {{
    const ulong n = {n};
    ulong i = gid;
    if (i >= n) return;
    ulong outer = i / {inner};
    ulong inner = i % {inner};
{coords}    {ta} acc = {init};
    for (ulong r = 0; r < {extent}; ++r) {{
        long c{axis} = (long)r;
{chain}        {ta} v = {value};
        acc = {fold};
    }}
    out[i] = acc;
}}"#
    );
    Ok(vec![KernelSource::plain(source, n)])
}

pub(crate) fn lower_expr(expr: &IotaExpr, rank: usize) -> Result<String> {
    lower_expr_pref(expr, rank, "c")
}

pub(crate) fn lower_expr_pref(expr: &IotaExpr, rank: usize, prefix: &str) -> Result<String> {
    let rec = |e: &IotaExpr| lower_expr_pref(e, rank, prefix);
    Ok(match expr {
        IotaExpr::Lit(v) => crate::symbolic::integer_literal(*v),
        IotaExpr::Var(name) => crate::symbolic::variable(name),
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
        IotaExpr::CeilDiv(a, b) => format!("luminal_ceil_div({}, {})", rec(a)?, rec(b)?),
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

pub(crate) fn coord_prelude(dims: &[Expr]) -> String {
    let mut out = String::from("    ulong rem = i;\n");
    for axis in (0..dims.len()).rev() {
        out.push_str(&format!(
            "    long c{axis} = (long)(rem % {}); rem /= {};\n",
            dims[axis], dims[axis]
        ));
    }
    out
}

fn literal_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; dims.len()];
    for k in (0..dims.len().saturating_sub(1)).rev() {
        strides[k] = strides[k + 1] * dims[k + 1];
    }
    strides
}

pub(crate) fn strides_of(dims: &[Expr]) -> Vec<Expr> {
    let mut strides = vec![Expr::from(1usize); dims.len()];
    for k in (0..dims.len().saturating_sub(1)).rev() {
        strides[k] = strides[k + 1].clone() * dims[k + 1].clone();
    }
    strides
}

/// Validate the installed plan's executable inventory without touching a GPU.
/// This checks code generation; it never selects or rewrites operations.
pub(crate) fn validate_plan(plan: &crate::MetalPlan) -> Result<()> {
    use luminal::buffer_tensor_ir::{BufferAlloc, BufferFree};
    use luminal::bufferize::BufferNode;
    for node in plan.dag.node_weights() {
        if let BufferNode::Compute {
            op,
            operand_info,
            result_info,
            reads,
            writes,
            ..
        } = node
        {
            if op.as_any().is::<BufferAlloc>() || op.as_any().is::<BufferFree>() {
                continue;
            }
            anyhow::ensure!(
                result_info.len() == 1
                    && operand_info.len() == reads.len()
                    && result_info.len() == writes.len(),
                "{}: malformed Metal kernel slots",
                op.label()
            );
            let ctx = CodegenCtx::from_descriptors(op.label(), operand_info, result_info)?;
            let kernel = crate::as_kernel_op(op.as_ref())
                .ok_or_else(|| anyhow::anyhow!("{} has no Metal kernel interface", op.label()))?;
            kernel.codegen(&ctx)?;
        }
    }
    Ok(())
}
