//! Resolved vendor descriptors and buffer bounds checks.
//! Egglog chooses every descriptor, order, transpose and optional argument.
//! Rust evaluates those expressions and checks the actual backing capacities;
//! it never recovers a layout or rebinds the selected destination.
use super::CublasLtForm;
use anyhow::{Result, bail};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LtOrder {
    /// `CUBLASLT_ORDER_ROW`: element `(r, c)` at `r*ld + c`; `ld` is the
    /// ROW pitch and the descriptor reaches `ld*(rows-1) + cols`.
    Row,
    /// `CUBLASLT_ORDER_COL`: element `(r, c)` at `c*ld + r`; `ld` is the
    /// COLUMN pitch and the descriptor reaches `ld*(cols-1) + rows`.
    Col,
}

/// One descriptor's geometry: `rows x cols` with leading dimension `ld`
/// and its storage [`LtOrder`], all resolved literals in elements.
/// The elected egglog descriptor supplies every field unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LtDesc {
    pub rows: i64,
    pub cols: i64,
    pub ld: i64,
    pub order: LtOrder,
    pub batch_count: i64,
    pub batch_stride_in_elements: i64,
}

impl LtDesc {
    /// A ROW-order descriptor (the operand convention).
    pub fn row(rows: i64, cols: i64, ld: i64) -> Self {
        Self {
            rows,
            cols,
            ld,
            order: LtOrder::Row,
            batch_count: 1,
            batch_stride_in_elements: 0,
        }
    }

    /// A COL-order descriptor.
    pub fn col(rows: i64, cols: i64, ld: i64) -> Self {
        Self {
            rows,
            cols,
            ld,
            order: LtOrder::Col,
            batch_count: 1,
            batch_stride_in_elements: 0,
        }
    }

    /// How many elements past the base pointer this descriptor
    /// dereferences: `ld * (major_lines - 1) + minor_extent`, where the
    /// MAJOR lines are rows in ROW order and columns in COL order.
    /// `None` on i64 overflow.
    fn reach(&self) -> Option<i64> {
        let (major, minor) = match self.order {
            LtOrder::Row => (self.rows, self.cols),
            LtOrder::Col => (self.cols, self.rows),
        };
        self.ld
            .checked_mul(major - 1)?
            .checked_add(minor)?
            .checked_add(
                self.batch_stride_in_elements
                    .checked_mul(self.batch_count - 1)?,
            )
    }
}

/// An actual C input, with its independently proved layout and storage type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LtCMatrixInput {
    pub descriptor: LtDesc,
    pub data_type: luminal::dtype::PlanDtype,
    pub operand: usize,
}

/// The optional bias's physical type and literal operand position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LtBiasInput {
    pub data_type: luminal::dtype::PlanDtype,
    pub operand: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LtActivation {
    Identity,
    RectifiedLinear,
    GaussianErrorLinear,
}

/// The explicitly admitted input-rounding and computation contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LtComputation {
    Float32,
    TensorFloat32,
}

impl LtComputation {
    pub fn egglog_constructor(self) -> &'static str {
        match self {
            Self::Float32 => "CublasLtFloat32Computation",
            Self::TensorFloat32 => "CublasLtTensorFloat32Computation",
        }
    }
}

/// Host values retain their FP32 bits, including signed zero and NaN payloads.
/// Device values are ordinary read operands; replay reads their current bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LtScalars {
    Host {
        alpha_bits: u32,
        beta_bits: u32,
    },
    Device {
        alpha_operand: usize,
        beta_operand: usize,
    },
}

impl LtScalars {
    pub fn host(alpha: f32, beta: f32) -> Self {
        Self::Host {
            alpha_bits: alpha.to_bits(),
            beta_bits: beta.to_bits(),
        }
    }
    pub fn host_values(self) -> Option<(f32, f32)> {
        match self {
            Self::Host {
                alpha_bits,
                beta_bits,
            } => Some((f32::from_bits(alpha_bits), f32::from_bits(beta_bits))),
            Self::Device { .. } => None,
        }
    }
    pub fn input_count(self) -> usize {
        if matches!(self, Self::Device { .. }) {
            2
        } else {
            0
        }
    }
    pub fn pointer_mode_constructor(self) -> &'static str {
        match self {
            Self::Host { .. } => "CublasLtHostScalarPointerMode",
            Self::Device { .. } => "CublasLtDeviceScalarPointerMode",
        }
    }
}

/// A fully resolved vendor call. Every descriptor and scalar argument is elected
/// in egglog; dispatch only validates backing capacities and vendor availability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LtCall {
    pub form: CublasLtForm,
    pub computation: LtComputation,
    pub m: i64,
    pub n: i64,
    pub k: i64,
    pub trans_a: bool,
    pub trans_b: bool,
    pub a_dtype: luminal::dtype::PlanDtype,
    pub b_dtype: luminal::dtype::PlanDtype,
    pub d_dtype: luminal::dtype::PlanDtype,
    pub a: LtDesc,
    pub b: LtDesc,
    /// Absent C is represented explicitly; dispatch supplies the vendor's
    /// required placeholder descriptor without inventing a logical input.
    pub c: Option<LtCMatrixInput>,
    pub d: LtDesc,
    pub scale_dtype: luminal::dtype::PlanDtype,
    pub scalars: LtScalars,
    pub activation: LtActivation,
    /// Physical bias storage, structurally absent on non-bias forms.
    pub bias: Option<LtBiasInput>,
}

pub fn validate_ld_bounds(who: &str, desc: &LtDesc, elems: usize) -> Result<()> {
    validate_descriptor_bounds(who, desc, elems, false)
}

fn validate_descriptor_bounds(
    who: &str,
    desc: &LtDesc,
    elems: usize,
    broadcast_c: bool,
) -> Result<()> {
    if desc.rows < 1 || desc.cols < 1 {
        bail!(
            "cuBLASLt {who}: empty descriptor geometry {}x{} — refused before dispatch",
            desc.rows,
            desc.cols
        );
    }
    if desc.ld < 0 || (desc.ld == 0 && !broadcast_c) {
        bail!(
            "cuBLASLt {who}: ld {} < 1 — refused before dispatch",
            desc.ld
        );
    }
    if !(1..=i64::from(i32::MAX)).contains(&desc.batch_count) {
        bail!(
            "cuBLASLt {who}: batch count {} must fit a positive int32",
            desc.batch_count
        );
    }
    if desc.batch_stride_in_elements < 0 {
        bail!("cuBLASLt {who}: negative batch stride — refused before dispatch");
    }
    let needed = desc.reach().ok_or_else(|| {
        anyhow::anyhow!("cuBLASLt {who}: the descriptor's element reach overflows i64 — refused")
    })?;
    if needed as i128 > elems as i128 {
        bail!(
            "cuBLASLt {who}: descriptor {}x{} ld {} ({:?} order) needs {} elements \
             but the buffer holds {} — out-of-bounds access refused BEFORE dispatch \
             (the library's own ld check is vacuous at a single major line, \
             i.e. rows==1 in ROW order)",
            desc.rows,
            desc.cols,
            desc.ld,
            desc.order,
            needed,
            elems
        );
    }
    Ok(())
}

pub fn assert_bias_destination_order(call: &LtCall, who: &str) -> Result<()> {
    if (call.bias.is_some() || call.activation != LtActivation::Identity)
        && call.d.order != LtOrder::Col
    {
        bail!(
            "cuBLASLt {who}: unreachable: bias or activation requires a column-order D; \
             a bias form ({}) reached the executor with a {:?}-order D descriptor \
             ({}x{} ld {}). The library refuses BIAS/RELU_BIAS on a ROW-order D \
             (CUBLAS_STATUS_NOT_SUPPORTED, measured on the A100 2026-08-28) — \
             refused BEFORE dispatch, no bytes move",
            format_args!("{:?}", call.form),
            call.d.order,
            call.d.rows,
            call.d.cols,
            call.d.ld
        );
    }
    Ok(())
}

impl LtCall {
    /// Validate every descriptor against its backing buffer's element
    /// count — the pre-dispatch gate (contract 4). `elems` follows the elected
    /// read order; the destination's capacity is checked separately.
    pub fn validate_against(&self, operand_elems: &[usize], dest_elems: usize) -> Result<()> {
        if self.scale_dtype != luminal::dtype::PlanDtype::F32 {
            bail!(
                "cuBLASLt: admitted compute modes require FP32 alpha/beta, got {:?}",
                self.scale_dtype
            );
        }
        let arity = self.form.lit_arity() + self.scalars.input_count();
        if operand_elems.len() != arity {
            bail!(
                "cuBLASLt {}: {} operand buffers for Lit arity {}",
                format_args!("{:?}", self.form),
                operand_elems.len(),
                arity
            );
        }
        validate_ld_bounds("A", &self.a, operand_elems[0])?;
        validate_ld_bounds("B", &self.b, operand_elems[1])?;
        validate_ld_bounds("D", &self.d, dest_elems)?;
        if let LtScalars::Device {
            alpha_operand,
            beta_operand,
        } = self.scalars
        {
            for (name, operand) in [("alpha", alpha_operand), ("beta", beta_operand)] {
                if operand_elems.get(operand).is_none_or(|&len| len < 1) {
                    bail!("cuBLASLt: device {name} operand {operand} needs one FP32 element");
                }
            }
        } else if self.c.is_none() && self.scalars.host_values().unwrap().1 != 0.0 {
            bail!("cuBLASLt: absent C requires beta zero");
        }
        if let Some(c) = self.c {
            let Some(&elems) = operand_elems.get(c.operand) else {
                bail!(
                    "cuBLASLt: C operand index {} out of operand range",
                    c.operand
                );
            };
            // C alone may broadcast a packed vector with leading dimension
            // zero. D still goes through the nonzero-leading-dimension gate.
            validate_descriptor_bounds("C", &c.descriptor, elems, true)?;
        }
        if let Some(bias) = self.bias {
            let i = bias.operand;
            let Some(&elems) = operand_elems.get(i) else {
                bail!("cuBLASLt: bias operand index {i} out of operand range");
            };
            // The bias vector is length m: one entry per vendor D row.
            if (elems as i128) < self.m as i128 {
                bail!(
                    "cuBLASLt bias: buffer holds {elems} elements, epilogue reads m = {} \
                     — refused before dispatch",
                    self.m
                );
            }
        }
        Ok(())
    }
}
