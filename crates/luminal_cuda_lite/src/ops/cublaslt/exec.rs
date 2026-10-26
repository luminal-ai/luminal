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
}

impl LtDesc {
    /// A ROW-order descriptor (the operand convention).
    pub fn row(rows: i64, cols: i64, ld: i64) -> Self {
        Self {
            rows,
            cols,
            ld,
            order: LtOrder::Row,
        }
    }

    /// A COL-order descriptor.
    pub fn col(rows: i64, cols: i64, ld: i64) -> Self {
        Self {
            rows,
            cols,
            ld,
            order: LtOrder::Col,
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
        self.ld.checked_mul(major - 1)?.checked_add(minor)
    }
}

/// Where the C pointer comes from. The C DESCRIPTOR always exists
/// (contract 3); only the pointer source varies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CSource {
    /// No-C forms: pass the D pointer as C (beta = 0.0f, C never read).
    AliasD,
    /// C-fold forms: the Lit operand at this index (contract order
    /// `[a, b, c, bias?]` puts c at 2); beta = 1.0f.
    Operand(usize),
}

/// The fully-resolved host call: every number the dispatch needs,
/// nothing the dispatch may reinterpret. NO scalar fields beyond the
/// structural `beta_is_one` — see module doc contract 2.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LtCall {
    pub form: CublasLtForm,
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
    /// ALWAYS present (contract 3): on the no-C forms this mirrors `d`.
    pub c: LtDesc,
    pub d: LtDesc,
    pub c_source: CSource,
    /// beta is STRUCTURAL: `true` exactly on the C-fold forms.
    pub beta_is_one: bool,
    pub relu: bool,
    /// Lit operand index of the bias vector, on the bias forms.
    pub bias_operand: Option<usize>,
}

pub fn validate_ld_bounds(who: &str, desc: &LtDesc, elems: usize) -> Result<()> {
    if desc.rows < 1 || desc.cols < 1 {
        bail!(
            "cuBLASLt {who}: empty descriptor geometry {}x{} — refused before dispatch",
            desc.rows,
            desc.cols
        );
    }
    if desc.ld < 1 {
        bail!(
            "cuBLASLt {who}: ld {} < 1 — refused before dispatch",
            desc.ld
        );
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
    if call.bias_operand.is_some() && call.d.order != LtOrder::Col {
        bail!(
            "cuBLASLt {who}: unreachable: bias requires a column-order D; \
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
    /// count — the pre-dispatch gate (contract 4). `elems` in Lit
    /// operand order `[a, b, c?, bias?]`, then the destination.
    pub fn validate_against(&self, operand_elems: &[usize], dest_elems: usize) -> Result<()> {
        if operand_elems.len() != self.form.lit_arity() {
            bail!(
                "cuBLASLt {}: {} operand buffers for Lit arity {}",
                format_args!("{:?}", self.form),
                operand_elems.len(),
                self.form.lit_arity()
            );
        }
        validate_ld_bounds("A", &self.a, operand_elems[0])?;
        validate_ld_bounds("B", &self.b, operand_elems[1])?;
        validate_ld_bounds("D", &self.d, dest_elems)?;
        match self.c_source {
            CSource::AliasD => validate_ld_bounds("C(=D)", &self.c, dest_elems)?,
            CSource::Operand(i) => {
                let Some(&elems) = operand_elems.get(i) else {
                    bail!("cuBLASLt: C operand index {i} out of operand range");
                };
                validate_ld_bounds("C", &self.c, elems)?;
            }
        }
        if let Some(i) = self.bias_operand {
            let Some(&elems) = operand_elems.get(i) else {
                bail!("cuBLASLt: bias operand index {i} out of operand range");
            };
            // The bias vector is length m (one entry per D row —
            // independent of storage order; this is why the ROW
            // convention was chosen over the CM-swap trick).
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
