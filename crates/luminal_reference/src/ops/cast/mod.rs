//! The dtype-changing materializer: `out[i] = convert(in[i])` to the target
//! dtype carried in the egglog term. Never a view — the element bit width
//! changes underfoot, so the result is always fresh storage in the
//! canonical layout minted for the OUTPUT's dtype. Functional only.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, ToDps};

/// `CastGeneric(input) -> out`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cast;

impl OpSlotNames for Cast {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for Cast {
    fn label(&self) -> &str {
        "CastGeneric"
    }
}

impl Bufferizable for Cast {}

impl ToDps for Cast {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        Some(Box::new(CastDps))
    }
}

impl LayoutIrOp for Cast {}

/// Destination-passing form of [`Cast`]:
///
/// ```text
/// CastGeneric(input: read, dest0: write-only ↔ out0) -> out0
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CastDps;

impl OpSlotNames for CastDps {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            1 => "dest0".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for CastDps {
    fn label(&self) -> &str {
        "CastGeneric" // DPS forms keep the IR name
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        operand != 1 // dest0 is write-only
    }
}

impl Bufferizable for CastDps {
    fn alias_info(&self) -> Vec<luminal::layout_ir::AliasInfo> {
        vec![luminal::layout_ir::AliasInfo {
            operand: 1,
            result: 0,
            sharing: luminal::layout_ir::Sharing::Must,
        }]
    }
}

impl ToDps for CastDps {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // already DPS — keeps the rewrite pass idempotent
    }
}

impl LayoutIrOp for CastDps {}

// ---------------------------------------------------------------------------
// Matchers
// ---------------------------------------------------------------------------

/// Matches `LayoutTensorOpCastGeneric` enodes and produces [`Cast`]
/// instances. Metadata children: `dtype` at child 1 (the target), and
/// `out_layout` at child 2.
#[derive(Debug, Clone, Copy, Default)]
pub struct CastMatcher;

impl OpMatcher for CastMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpCastGeneric"
    }

    fn snippets(&self) -> Vec<luminal::egglog_snippet::EgglogSnippet> {
        vec![
            luminal::egglog_snippet::EgglogSnippet {
                category: luminal::egglog_snippet::SpliceCategory::LayoutOpConstructors,
                text: include_str!("match_functional_constructor.egg"),
            },
            luminal::egglog_snippet::EgglogSnippet {
                category: luminal::egglog_snippet::SpliceCategory::Match,
                text: include_str!("match_functional.egg"),
            },
        ]
    }

    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[("dtype", 1), ("out_layout", 2)]
    }

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(Cast)
    }
}

// ---------------------------------------------------------------------------
// ---- kernel ----
// Reference-runtime execution for this op, dispatched by TypeId from the
// label->fn table in `crate::kernels` (op-folder ruling
// 2026-08-13: everything about an op lives in the op's folder).
// ---------------------------------------------------------------------------

use luminal::buffer_tensor_ir::{ReferenceKernelCtx, TypedBuffer};

pub(crate) fn kernel(
    _op: &dyn BufferTensorIrOp,
    ctx: &mut ReferenceKernelCtx,
) -> anyhow::Result<()> {
    // The conversion is driven by the BUFFER types the plan annotated —
    // the op needs no dtype field of its own. Covered pairs only;
    // anything else refuses loudly (never a silent reinterpretation).
    // The 2026-08-11 cast policy: int -> float is CHECKED-EXACT
    // (conservative |v| <= 2^24, loud outside it); float -> int is a
    // REFUSAL (a lossy read is an explicit op, never a cast — the
    // Bool8 projection rule generalized); bool -> number is the exact
    // 0/1 indicator bridge; number -> bool is always the refusal.
    //
    // NARROW INTEGERS (ruling 2026-09-02, main #399's carve-out): five
    // integer widths would be 25 hand-written pair arms, so any cast
    // TOUCHING I8/U8/I16 routes through `narrow_cast` below, which
    // states the policy once. The wide arms in this match are
    // untouched.
    const F32_EXACT_INT: i64 = 1 << 24;
    if is_narrow(&ctx.operands[0]) || is_narrow(&ctx.dests[0]) {
        return narrow_cast(ctx);
    }
    match (&ctx.operands[0], &mut ctx.dests[0]) {
        // Same-type: value-preserving copies.
        (TypedBuffer::F32(input), TypedBuffer::F32(dest)) => {
            anyhow::ensure!(input.len() == dest.len(), "cast length mismatch");
            dest.copy_from_slice(input);
        }
        (TypedBuffer::I32(input), TypedBuffer::I32(dest)) => {
            anyhow::ensure!(input.len() == dest.len(), "cast length mismatch");
            dest.copy_from_slice(input);
        }
        (TypedBuffer::I64(input), TypedBuffer::I64(dest)) => {
            anyhow::ensure!(input.len() == dest.len(), "cast length mismatch");
            dest.copy_from_slice(input);
        }
        (TypedBuffer::Bool8(input), TypedBuffer::Bool8(dest)) => {
            anyhow::ensure!(input.len() == dest.len(), "cast length mismatch");
            dest.copy_from_slice(input);
        }
        // The indicator bridges: bool -> number is exactly 0 / 1.
        (TypedBuffer::Bool8(input), TypedBuffer::F32(dest)) => {
            anyhow::ensure!(input.len() == dest.len(), "cast length mismatch");
            for (out, code) in dest.iter_mut().zip(input) {
                // The Bool8 invariant, enforced at the read: only the
                // two legal codes exist; anything else is ill-formed
                // data, not a truthy byte.
                anyhow::ensure!(*code <= 1, "Bool8 buffer holds ill-formed code {code}");
                *out = f32::from(*code);
            }
        }
        (TypedBuffer::Bool8(input), TypedBuffer::I32(dest)) => {
            anyhow::ensure!(input.len() == dest.len(), "cast length mismatch");
            for (out, code) in dest.iter_mut().zip(input) {
                anyhow::ensure!(*code <= 1, "Bool8 buffer holds ill-formed code {code}");
                *out = i32::from(*code);
            }
        }
        (TypedBuffer::Bool8(input), TypedBuffer::I64(dest)) => {
            anyhow::ensure!(input.len() == dest.len(), "cast length mismatch");
            for (out, code) in dest.iter_mut().zip(input) {
                anyhow::ensure!(*code <= 1, "Bool8 buffer holds ill-formed code {code}");
                *out = i64::from(*code);
            }
        }
        // Int -> float: checked-exact (conservative), loud outside it.
        (TypedBuffer::I32(input), TypedBuffer::F32(dest)) => {
            anyhow::ensure!(input.len() == dest.len(), "cast length mismatch");
            for (out, value) in dest.iter_mut().zip(input) {
                anyhow::ensure!(
                    (i64::from(*value)).abs() <= F32_EXACT_INT,
                    "cast i32 -> f32 loses exactness at value {value} \
                     (|v| <= 2^24 by the conservative-exact ruling)"
                );
                *out = *value as f32;
            }
        }
        (TypedBuffer::I64(input), TypedBuffer::F32(dest)) => {
            anyhow::ensure!(input.len() == dest.len(), "cast length mismatch");
            for (out, value) in dest.iter_mut().zip(input) {
                anyhow::ensure!(
                    value.abs() <= F32_EXACT_INT,
                    "cast i64 -> f32 loses exactness at value {value} \
                     (|v| <= 2^24 by the conservative-exact ruling)"
                );
                *out = *value as f32;
            }
        }
        // Int widenings/narrowings: exact or loud.
        (TypedBuffer::I32(input), TypedBuffer::I64(dest)) => {
            anyhow::ensure!(input.len() == dest.len(), "cast length mismatch");
            for (out, value) in dest.iter_mut().zip(input) {
                *out = i64::from(*value);
            }
        }
        (TypedBuffer::I64(input), TypedBuffer::I32(dest)) => {
            anyhow::ensure!(input.len() == dest.len(), "cast length mismatch");
            for (out, value) in dest.iter_mut().zip(input) {
                *out = i32::try_from(*value).map_err(|_| {
                    anyhow::anyhow!("cast i64 -> i32 out of range at value {value}")
                })?;
            }
        }
        // fp8: the QUANTIZE direction is the model's own explicit
        // step (quantization is model definition, ruling 2026-08-12) —
        // E4M3FN semantics: round-to-nearest-even, SATURATE to ±448
        // (the clamp handles the crate's non-FN overflow behavior;
        // agreement with the checkpoint codec is pinned exhaustively
        // by test). Widening back is exact.
        (TypedBuffer::F32(input), TypedBuffer::F8E4M3(dest)) => {
            anyhow::ensure!(input.len() == dest.len(), "cast length mismatch");
            for (out, value) in dest.iter_mut().zip(input) {
                *out = if value.is_nan() {
                    float8::F8E4M3::from_bits(0x7F)
                } else {
                    float8::F8E4M3::from_f32(value.clamp(-448.0, 448.0))
                };
            }
        }
        (TypedBuffer::F8E4M3(input), TypedBuffer::F32(dest)) => {
            anyhow::ensure!(input.len() == dest.len(), "cast length mismatch");
            for (out, code) in dest.iter_mut().zip(input) {
                *out = code.to_f32();
            }
        }
        (TypedBuffer::F8E4M3(input), TypedBuffer::F8E4M3(dest)) => {
            anyhow::ensure!(input.len() == dest.len(), "cast length mismatch");
            dest.copy_from_slice(input);
        }
        // Float -> int is a REFUSAL: rounding/truncation is a lossy
        // read and must be an explicit op with ruled semantics, never
        // a cast (the F32 -> Bool8 projection rule generalized).
        (TypedBuffer::F32(_), TypedBuffer::I32(_) | TypedBuffer::I64(_)) => {
            anyhow::bail!(
                "cast f32 -> int is not a reinterpretation: a rounding \
                 or truncation is a lossy read and must appear as an \
                 explicit op in the model, never as a cast"
            );
        }
        (
            TypedBuffer::F32(_) | TypedBuffer::I32(_) | TypedBuffer::I64(_),
            TypedBuffer::Bool8(_),
        ) => {
            anyhow::bail!(
                "cast number -> Bool8 is not a reinterpretation: the != 0 \
                 reading is a PROJECTION and must appear as an explicit \
                 comparison in the model (LessThan), never as a cast"
            );
        }
        (input, dest) => anyhow::bail!(
            "cast has no ({} -> {}) arm",
            input.type_name(),
            dest.type_name()
        ),
    }
    Ok(())
}

fn is_narrow(buffer: &TypedBuffer) -> bool {
    matches!(
        buffer,
        TypedBuffer::I8(_) | TypedBuffer::U8(_) | TypedBuffer::I16(_)
    )
}

/// Any integer buffer read as i64. LOSSLESS at every integer width this
/// runtime has (I8/U8/I16/I32/I64 all fit in i64), so the reading never
/// loses a value — only the write side can, and only where the policy
/// below says it may.
fn read_integer(buffer: &TypedBuffer) -> Option<Vec<i64>> {
    Some(match buffer {
        TypedBuffer::I8(values) => values.iter().map(|v| i64::from(*v)).collect(),
        TypedBuffer::U8(values) => values.iter().map(|v| i64::from(*v)).collect(),
        TypedBuffer::I16(values) => values.iter().map(|v| i64::from(*v)).collect(),
        TypedBuffer::I32(values) => values.iter().map(|v| i64::from(*v)).collect(),
        TypedBuffer::I64(values) => values.clone(),
        _ => return None,
    })
}

/// Every cast with a narrow integer on either side. The policy, stated
/// once (carve-out 2026-09-02 — Austin to confirm at review):
///
/// * int -> NARROW int TRUNCATES (`as`), which is main #399's semantics
///   and torch's: at 8 and 16 bits a narrowing conversion is a defined
///   wrap, not an error.
/// * int -> `Int` / `Int64` stays CHECKED — the non-wrapping ruling of
///   2026-08-11 is untouched for the wide types, so `I64 -> I32` still
///   refuses out of range.
/// * NARROW int -> float is exact BY WIDTH (|v| <= 32767, far inside the
///   f32 2^24 exactness bound), so it satisfies the checked-exact
///   int -> float rule with nothing left to check.
/// * `Bool8` -> narrow int is the exact 0/1 indicator bridge.
/// * float -> NARROW int is REFUSED, exactly like every other
///   float -> int. This is the ONE place main's unchecked `as` is not
///   carried: the carve-out is about integer WIDTH semantics, and it is
///   not a licence to make a lossy float read implicit (cast policy
///   2026-08-11). A rounding or truncation stays an explicit op.
/// * narrow int -> `Bool8` is REFUSED: `!= 0` is a projection, never a
///   cast.
fn narrow_cast(ctx: &mut ReferenceKernelCtx) -> anyhow::Result<()> {
    let source_name = ctx.operands[0].type_name();
    let dest_name = ctx.dests[0].type_name();
    let values: Vec<i64> = match &ctx.operands[0] {
        TypedBuffer::Bool8(codes) => {
            let mut values = Vec::with_capacity(codes.len());
            for code in codes {
                anyhow::ensure!(*code <= 1, "Bool8 buffer holds ill-formed code {code}");
                values.push(i64::from(*code));
            }
            values
        }
        other => read_integer(other).ok_or_else(|| {
            anyhow::anyhow!(
                "cast {source_name} -> {dest_name} is not a reinterpretation: \
                 a rounding or truncation is a lossy read and must appear as \
                 an explicit op in the model, never as a cast"
            )
        })?,
    };
    anyhow::ensure!(values.len() == ctx.dests[0].len(), "cast length mismatch");
    match &mut ctx.dests[0] {
        TypedBuffer::I8(dest) => {
            for (out, value) in dest.iter_mut().zip(&values) {
                *out = *value as i8;
            }
        }
        TypedBuffer::U8(dest) => {
            for (out, value) in dest.iter_mut().zip(&values) {
                *out = *value as u8;
            }
        }
        TypedBuffer::I16(dest) => {
            for (out, value) in dest.iter_mut().zip(&values) {
                *out = *value as i16;
            }
        }
        TypedBuffer::I32(dest) => {
            for (out, value) in dest.iter_mut().zip(&values) {
                *out = i32::try_from(*value).map_err(|_| {
                    anyhow::anyhow!("cast {source_name} -> i32 out of range at value {value}")
                })?;
            }
        }
        TypedBuffer::I64(dest) => {
            for (out, value) in dest.iter_mut().zip(&values) {
                *out = *value;
            }
        }
        TypedBuffer::F32(dest) => {
            for (out, value) in dest.iter_mut().zip(&values) {
                *out = *value as f32;
            }
        }
        TypedBuffer::F64(dest) => {
            for (out, value) in dest.iter_mut().zip(&values) {
                *out = *value as f64;
            }
        }
        TypedBuffer::Bool8(_) => anyhow::bail!(
            "cast {source_name} -> Bool8 is not a reinterpretation: the != 0 \
             reading is a PROJECTION and must appear as an explicit \
             comparison in the model (LessThan), never as a cast"
        ),
        dest => anyhow::bail!("cast has no ({source_name} -> {}) arm", dest.type_name()),
    }
    Ok(())
}
