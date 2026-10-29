//! Exact encoding access for Boolean conversion kernels and read-only bit views.
use crate::TypedBuffer;
use crate::narrow_float::NarrowFloat;
use crate::typed_buffer::narrow_float_match;
use anyhow::{Result, anyhow, ensure};
use luminal::{
    dtype::PlanDtype,
    layouts::{BitOffsetExpressionLayout, DecodedLayout},
};

pub(crate) fn code(data: &TypedBuffer, i: usize) -> Result<u64> {
    ensure!(i < data.len(), "bit view reads outside backing storage");
    Ok(match data {
        TypedBuffer::F32(v) => u64::from(v[i].to_bits()),
        // Logical TF32 has nineteen encoding bits in a 32-bit backing word.
        TypedBuffer::TF32(v) => u64::from(v[i].to_bits() >> 13),
        TypedBuffer::F64(v) => v[i].to_bits(),
        TypedBuffer::I32(v) => u64::from(v[i] as u32),
        TypedBuffer::I64(v) => v[i] as u64,
        TypedBuffer::U64(v) => v[i],
        TypedBuffer::I8(v) => u64::from(v[i] as u8),
        TypedBuffer::U8(v) => u64::from(v[i]),
        TypedBuffer::I16(v) => u64::from(v[i] as u16),
        TypedBuffer::Bool8(v) => {
            ensure!(v[i] <= 1, "ill-formed Bool8 code");
            u64::from(v[i])
        }
        narrow @ (TypedBuffer::F8E4M3FN(_)
        | TypedBuffer::F16(_)
        | TypedBuffer::Bf16(_)
        | TypedBuffer::F8E5M2(_)
        | TypedBuffer::F8E4M3FNUZ(_)
        | TypedBuffer::F8E5M2FNUZ(_)) => {
            narrow_float_match!(narrow, v => v[i].to_code(), _other => unreachable!())
        }
    })
}
pub(crate) fn set_code(data: &mut TypedBuffer, i: usize, code: u64) -> Result<()> {
    ensure!(i < data.len(), "bit encoding write outside destination");
    match data {
        TypedBuffer::F32(v) => v[i] = f32::from_bits(code as u32),
        TypedBuffer::TF32(v) => v[i] = crate::Tf32::from_precision_bits(code)?,
        TypedBuffer::F64(v) => v[i] = f64::from_bits(code),
        TypedBuffer::I32(v) => v[i] = code as i32,
        TypedBuffer::I64(v) => v[i] = code as i64,
        TypedBuffer::U64(v) => v[i] = code,
        TypedBuffer::I8(v) => v[i] = code as i8,
        TypedBuffer::U8(v) => v[i] = code as u8,
        TypedBuffer::I16(v) => v[i] = code as i16,
        TypedBuffer::Bool8(v) => {
            ensure!(code <= 1, "ill-formed Bool8 encoding {code}");
            v[i] = code as u8;
        }
        narrow @ (TypedBuffer::F8E4M3FN(_)
        | TypedBuffer::F16(_)
        | TypedBuffer::Bf16(_)
        | TypedBuffer::F8E5M2(_)
        | TypedBuffer::F8E4M3FNUZ(_)
        | TypedBuffer::F8E5M2FNUZ(_)) => {
            narrow_float_match!(narrow, v => v[i] = NarrowFloat::from_code(code), _other => unreachable!())
        }
    }
    Ok(())
}

/// Resolve dimension variables before evaluating a per-element bit address.
fn resolved(
    term: &luminal::layouts::IntExprTerm,
    dims: &luminal::shape::DynMap,
) -> Result<luminal::layouts::IntExprTerm> {
    use luminal::layouts::IntExprTerm as E;
    let go = |x: &E| resolved(x, dims).map(Box::new);
    Ok(match term {
        E::Var(_) => E::Lit(term.eval_dims(dims)?),
        E::Lit(_) | E::Coord { .. } => term.clone(),
        E::Add(a, b) => E::Add(go(a)?, go(b)?),
        E::Mul(a, b) => E::Mul(go(a)?, go(b)?),
        E::TruncDiv(a, b) => E::TruncDiv(go(a)?, go(b)?),
        E::TruncRem(a, b) => E::TruncRem(go(a)?, go(b)?),
        E::CeilDiv(a, b) => E::CeilDiv(go(a)?, go(b)?),
        E::Min(a, b) => E::Min(go(a)?, go(b)?),
        E::Max(a, b) => E::Max(go(a)?, go(b)?),
        E::LessThanCast(a, b) => E::LessThanCast(go(a)?, go(b)?),
    })
}

/// Materialize a consumer's logical values from an elected read-only bit view.
/// The owning buffer keeps its original typed representation. In particular,
/// internal one-bit Bool storage uses one host byte per logical bit; it is NOT
/// byte-addressed Bool8 and must not be interpreted as such.
pub(crate) fn read_view(
    data: &TypedBuffer,
    owner: PlanDtype,
    layout: &DecodedLayout,
    dims: &luminal::shape::DynMap,
) -> Result<TypedBuffer> {
    let dtype = layout
        .dtype
        .ok_or_else(|| anyhow!("bit view has no dtype"))?;
    let shape = layout.extents_with(dims)?;
    let n = shape.iter().try_fold(1usize, |a, b| {
        a.checked_mul(*b)
            .ok_or_else(|| anyhow!("bit view volume overflow"))
    })?;
    let mut out = TypedBuffer::zeroed(dtype, n)?;
    let width = usize::try_from(dtype.egglog_bits())?;
    let owner_width = usize::try_from(owner.egglog_bits())?;
    ensure!(
        width == usize::try_from(layout.width_bits())?,
        "bit view dtype/access width mismatch"
    );
    let bit_expr = layout
        .first::<BitOffsetExpressionLayout>()
        .map(|l| resolved(&l.offset, dims))
        .transpose()?;
    let mut coords = vec![0; shape.len()];
    for i in 0..n {
        let mut rem = i;
        for (c, d) in coords.iter_mut().zip(&shape).rev() {
            *c = rem % *d;
            rem /= *d;
        }
        let offset = match &bit_expr {
            Some(expr) => usize::try_from(expr.eval_at(&coords)?)?,
            None => layout
                .element_index(&coords)?
                .checked_mul(width)
                .ok_or_else(|| anyhow!("bit offset overflow"))?,
        };
        let mut value = 0u64;
        for bit in 0..width {
            let address = offset
                .checked_add(bit)
                .ok_or_else(|| anyhow!("bit address overflow"))?;
            value |= ((code(data, address / owner_width)? >> (address % owner_width)) & 1) << bit;
        }
        set_code(&mut out, i, value)?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use luminal::layouts::{BitWidthTerm, IntExprTerm as E, ShapeTerm};

    #[test]
    fn tf32_precision_codes_preserve_exact_bits_in_full_word_storage() {
        let mut values = TypedBuffer::TF32(vec![crate::Tf32::default(); 6]);
        for (i, bits) in [0, 1 << 18, 1, 0x1fc00, 0x1fc01, 0x7ffff]
            .into_iter()
            .enumerate()
        {
            set_code(&mut values, i, bits).unwrap();
            assert_eq!(code(&values, i).unwrap(), bits);
            assert_eq!(
                u64::from(values.as_tf32().unwrap()[i].to_bits()),
                bits << 13
            );
        }
        assert_eq!(values.byte_len(), 24);
        assert!(set_code(&mut values, 0, 1 << 19).is_err());
    }

    #[test]
    fn bit_reads_handle_subbyte_origins_and_skip_unaccessed_bits() {
        let layout = DecodedLayout::of(
            BitOffsetExpressionLayout {
                offset: E::Mul(Box::new(E::Coord { axis_from_end: 0 }), Box::new(E::Lit(2))),
                shape: ShapeTerm(vec![E::Lit(8)]),
                width: BitWidthTerm(1),
            },
            Some(PlanDtype::Bool),
        );
        let codes = [0xacu8, 0x53];
        let bits = read_view(
            &TypedBuffer::U8(codes.to_vec()),
            PlanDtype::U8,
            &layout,
            &Default::default(),
        )
        .unwrap();
        let word = u16::from_le_bytes(codes);
        assert_eq!(
            *bits.as_bool8().unwrap(),
            (0..8)
                .map(|i| ((word >> (2 * i)) & 1) as u8)
                .collect::<Vec<_>>()
        );
        let layout = DecodedLayout::of(
            BitOffsetExpressionLayout {
                offset: E::Lit(3),
                shape: ShapeTerm(vec![E::Lit(1)]),
                width: BitWidthTerm(8),
            },
            Some(PlanDtype::U8),
        );
        let packed = read_view(
            &TypedBuffer::U8(codes.to_vec()),
            PlanDtype::U8,
            &layout,
            &Default::default(),
        )
        .unwrap();
        assert_eq!(*packed.as_u8().unwrap(), vec![(word >> 3) as u8]);
        assert!(
            read_view(
                &TypedBuffer::U8(vec![0]),
                PlanDtype::U8,
                &layout,
                &Default::default()
            )
            .is_err()
        );
    }
}
