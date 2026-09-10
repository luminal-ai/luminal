//! Runtime evaluation, conservative capacity bounds, and MSL shape expressions.

use anyhow::{Result, anyhow, bail, ensure};
use luminal::layouts::{self, IntExprTerm as T, LayoutFacts};
use luminal::shape::{DynMap, Symbol};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub type Bounds = BTreeMap<Symbol, (usize, usize)>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expr(pub T);
impl From<usize> for Expr {
    fn from(v: usize) -> Self {
        Self(T::Lit(i64::try_from(v).expect("dimension exceeds i64")))
    }
}
impl std::ops::Mul for Expr {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self(T::Mul(Box::new(self.0), Box::new(rhs.0))).simplify()
    }
}
impl std::iter::Product for Expr {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(1usize.into(), |a, b| a * b)
    }
}
impl<'a> std::iter::Product<&'a Expr> for Expr {
    fn product<I: Iterator<Item = &'a Expr>>(iter: I) -> Self {
        iter.cloned().product()
    }
}
impl Expr {
    fn simplify(self) -> Self {
        self.0
            .eval_literal()
            .map(|v| Self(T::Lit(v)))
            .unwrap_or(self)
    }
    pub fn eval(&self, dims: &DynMap) -> Result<usize> {
        usize::try_from(if let Some(value) = self.0.eval_literal() {
            value
        } else {
            eval(&self.0, dims)?
        })
        .map_err(Into::into)
    }
    pub fn capacity(&self, bounds: &Bounds) -> Result<usize> {
        let (lo, hi) = interval(&self.0, bounds)?;
        ensure!(lo >= 0, "negative dimension bound {lo} for {:?}", self.0);
        Ok(usize::try_from(hi)?)
    }
    pub fn literal(&self) -> Option<usize> {
        self.0.eval_literal().and_then(|v| usize::try_from(v).ok())
    }
}
impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&metal(&self.0, &|_| panic!("coordinate in dimension")))
    }
}

pub fn variable(name: &str) -> String {
    let mut s = String::from("luminal_dim_");
    for b in name.bytes() {
        use std::fmt::Write;
        write!(s, "{b:02x}").unwrap();
    }
    s
}
pub fn metal(t: &T, coord: &dyn Fn(i64) -> String) -> String {
    let go = |e: &T| metal(e, coord);
    match t {
        T::Lit(v) => integer_literal(*v),
        T::Var(s) => variable(s),
        T::Coord { axis_from_end } => coord(*axis_from_end),
        T::Add(a, b) => format!("({}+{})", go(a), go(b)),
        T::Mul(a, b) => format!("({}*{})", go(a), go(b)),
        T::TruncDiv(a, b) => format!("({}/{})", go(a), go(b)),
        T::TruncRem(a, b) => format!("({}%{})", go(a), go(b)),
        T::CeilDiv(a, b) => format!("luminal_ceil_div({}, {})", go(a), go(b)),
        T::Min(a, b) => format!("luminal_min({}, {})", go(a), go(b)),
        T::Max(a, b) => format!("luminal_max({}, {})", go(a), go(b)),
        T::LessThanCast(a, b) => format!("({}<{} ? 1LL:0LL)", go(a), go(b)),
    }
}
pub const METAL_HELPERS: &str = "inline long luminal_min(long a,long b){return a<b?a:b;}\ninline long luminal_max(long a,long b){return a>b?a:b;}\ninline long luminal_ceil_div(long a,long b){return a/b + (a%b != 0 && ((a>0)==(b>0)));}\n";

pub fn eval(t: &T, dims: &DynMap) -> Result<i64> {
    let bounds = dims.iter().map(|(s, v)| (*s, (*v, *v))).collect();
    let (lo, hi) = interval(t, &bounds)?;
    ensure!(lo == hi, "expression did not evaluate exactly");
    Ok(lo)
}
pub fn interval(t: &T, bounds: &Bounds) -> Result<(i64, i64)> {
    let checked =
        |v: i128| i64::try_from(v).map_err(|_| anyhow!("shape expression overflow: {t:?}"));
    let (lo, hi) = match t {
        T::Lit(v) => (*v, *v),
        T::Var(s) => {
            let (lo, hi) = bounds
                .get(&Symbol::from(s.as_str()))
                .ok_or_else(|| anyhow!("unbound dimension `{s}`"))?;
            (i64::try_from(*lo)?, i64::try_from(*hi)?)
        }
        T::Coord { .. } => bail!("coordinate-dependent allocation span: {t:?}"),
        T::Add(a, b) => {
            let (a, z) = interval(a, bounds)?;
            let (b, y) = interval(b, bounds)?;
            (
                checked(a as i128 + b as i128)?,
                checked(z as i128 + y as i128)?,
            )
        }
        T::Mul(a, b) => {
            let (a, z) = interval(a, bounds)?;
            let (b, y) = interval(b, bounds)?;
            let v = [
                a as i128 * b as i128,
                a as i128 * y as i128,
                z as i128 * b as i128,
                z as i128 * y as i128,
            ];
            (
                checked(*v.iter().min().unwrap())?,
                checked(*v.iter().max().unwrap())?,
            )
        }
        T::TruncDiv(a, b) | T::CeilDiv(a, b) => {
            let (a, z) = interval(a, bounds)?;
            let (b, y) = interval(b, bounds)?;
            ensure!(b > 0 || y < 0, "divisor interval contains zero: {t:?}");
            let mut v = vec![];
            for n in [a, z] {
                for d in [b, y] {
                    let (n, d) = (n as i128, d as i128);
                    let q = n / d;
                    v.push(
                        q + if matches!(t, T::CeilDiv(..)) && n % d != 0 && ((n > 0) == (d > 0)) {
                            1
                        } else {
                            0
                        },
                    );
                }
            }
            (
                checked(*v.iter().min().unwrap())?,
                checked(*v.iter().max().unwrap())?,
            )
        }
        T::TruncRem(a, b) => {
            let (a, z) = interval(a, bounds)?;
            let (b, y) = interval(b, bounds)?;
            ensure!(b > 0 || y < 0, "divisor interval contains zero: {t:?}");
            if a == z && b == y {
                let r = checked(a as i128 % b as i128)?;
                (r, r)
            } else {
                let m = (b as i128).abs().max((y as i128).abs()) - 1;
                (
                    if a < 0 {
                        checked((-m).max(a as i128))?
                    } else {
                        0
                    },
                    if z > 0 { checked(m.min(z as i128))? } else { 0 },
                )
            }
        }
        T::Min(a, b) => {
            let (a, z) = interval(a, bounds)?;
            let (b, y) = interval(b, bounds)?;
            (a.min(b), z.min(y))
        }
        T::Max(a, b) => {
            let (a, z) = interval(a, bounds)?;
            let (b, y) = interval(b, bounds)?;
            (a.max(b), z.max(y))
        }
        T::LessThanCast(a, b) => {
            let (a, z) = interval(a, bounds)?;
            let (b, y) = interval(b, bounds)?;
            if z < b {
                (1, 1)
            } else if a >= y {
                (0, 0)
            } else {
                (0, 1)
            }
        }
    };
    ensure!(lo <= hi, "invalid dimension interval {lo}..{hi}");
    Ok((lo, hi))
}
pub fn vars(t: &T, out: &mut BTreeSet<Symbol>) {
    match t {
        T::Var(s) => {
            out.insert(Symbol::from(s.as_str()));
        }
        T::Lit(_) | T::Coord { .. } => {}
        T::Add(a, b)
        | T::Mul(a, b)
        | T::TruncDiv(a, b)
        | T::TruncRem(a, b)
        | T::CeilDiv(a, b)
        | T::Min(a, b)
        | T::Max(a, b)
        | T::LessThanCast(a, b) => {
            vars(a, out);
            vars(b, out);
        }
    }
}
pub fn substitute(t: &T, dims: &DynMap) -> Result<T> {
    let go = |t: &T| substitute(t, dims).map(Box::new);
    Ok(match t {
        T::Var(_) => T::Lit(eval(t, dims)?),
        T::Lit(_) | T::Coord { .. } => t.clone(),
        T::Add(a, b) => T::Add(go(a)?, go(b)?),
        T::Mul(a, b) => T::Mul(go(a)?, go(b)?),
        T::TruncDiv(a, b) => T::TruncDiv(go(a)?, go(b)?),
        T::TruncRem(a, b) => T::TruncRem(go(a)?, go(b)?),
        T::CeilDiv(a, b) => T::CeilDiv(go(a)?, go(b)?),
        T::Min(a, b) => T::Min(go(a)?, go(b)?),
        T::Max(a, b) => T::Max(go(a)?, go(b)?),
        T::LessThanCast(a, b) => T::LessThanCast(go(a)?, go(b)?),
    })
}
pub fn resolve_layout(l: &layouts::DecodedLayout, dims: &DynMap) -> Result<layouts::DecodedLayout> {
    let shape = layouts::ShapeTerm(
        l.shape()
            .0
            .iter()
            .map(|e| substitute(e, dims))
            .collect::<Result<_>>()?,
    );
    let mut spellings: Vec<std::sync::Arc<dyn LayoutFacts>> = vec![];
    if let Some(s) = l.first::<layouts::RightMajorContiguousElementLayout>() {
        spellings.push(std::sync::Arc::new(
            layouts::RightMajorContiguousElementLayout {
                shape: shape.clone(),
                width: s.width,
            },
        ));
    }
    if let Some(s) = l.first::<layouts::LeftMajorContiguousElementLayout>() {
        spellings.push(std::sync::Arc::new(
            layouts::LeftMajorContiguousElementLayout {
                shape: shape.clone(),
                width: s.width,
            },
        ));
    }
    if let Some(s) = l.first::<layouts::StridedElementLayout>() {
        spellings.push(std::sync::Arc::new(layouts::StridedElementLayout {
            shape: shape.clone(),
            width: s.width,
            chain: s
                .chain
                .iter()
                .map(|e| substitute(e, dims))
                .collect::<Result<_>>()?,
        }));
    }
    if let Some(s) = l.first::<layouts::ElementOffsetExpressionLayout>() {
        spellings.push(std::sync::Arc::new(
            layouts::ElementOffsetExpressionLayout {
                shape: shape.clone(),
                width: s.width,
                offset: substitute(&s.offset, dims)?,
            },
        ));
    }
    if let Some(s) = l.first::<layouts::BitOffsetExpressionLayout>() {
        spellings.push(std::sync::Arc::new(layouts::BitOffsetExpressionLayout {
            shape,
            width: s.width,
            offset: substitute(&s.offset, dims)?,
        }));
    }
    ensure!(
        !spellings.is_empty(),
        "unsupported layout {:?}",
        l.present()
    );
    let mut out = layouts::DecodedLayout::of_spellings(spellings, l.dtype);
    out.class = l.class.clone();
    Ok(out)
}
pub fn span(l: &layouts::DecodedLayout) -> Result<Expr> {
    l.spellings
        .iter()
        .find_map(|s| s.span_elements())
        .map(Expr)
        .ok_or_else(|| anyhow!("layout {:?} has no storage span", l.present()))
}
pub fn bytes(l: &layouts::DecodedLayout, dims: &DynMap) -> Result<usize> {
    span(l)?
        .eval(dims)?
        .checked_mul(crate::host_buffer::dtype_bytes(
            l.dtype.ok_or_else(|| anyhow!("layout has no dtype"))?,
        )?)
        .ok_or_else(|| anyhow!("buffer byte size overflow"))
}
pub fn capacity_bytes(l: &layouts::DecodedLayout, bounds: &Bounds) -> Result<usize> {
    span(l)?
        .capacity(bounds)?
        .checked_mul(crate::host_buffer::dtype_bytes(
            l.dtype.ok_or_else(|| anyhow!("layout has no dtype"))?,
        )?)
        .ok_or_else(|| anyhow!("buffer capacity overflow"))
}

pub fn iota_term(e: &luminal::index_expr::IotaExpr) -> Option<T> {
    use luminal::index_expr::IotaExpr as I;
    let go = |e: &I| iota_term(e).map(Box::new);
    Some(match e {
        I::Lit(v) => T::Lit(*v),
        I::Var(s) => T::Var(s.clone()),
        I::Coord(a) => T::Coord {
            axis_from_end: *a as i64,
        },
        I::Add(a, b) => T::Add(go(a)?, go(b)?),
        I::Mul(a, b) => T::Mul(go(a)?, go(b)?),
        I::TruncDiv(a, b) => T::TruncDiv(go(a)?, go(b)?),
        I::TruncRem(a, b) => T::TruncRem(go(a)?, go(b)?),
        I::CeilDiv(a, b) => T::CeilDiv(go(a)?, go(b)?),
        I::Min(a, b) => T::Min(go(a)?, go(b)?),
        I::Max(a, b) => T::Max(go(a)?, go(b)?),
        I::LessThanCast(a, b) => T::LessThanCast(go(a)?, go(b)?),
    })
}

#[derive(Debug, Clone, Default)]
pub struct ShapeEnv {
    pub bounds: Bounds,
    pub values: DynMap,
}

pub fn integer_literal(v: i64) -> String {
    if v == i64::MIN {
        "(-9223372036854775807LL - 1LL)".into()
    } else {
        format!("{v}LL")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn var() -> T {
        T::Var("n".into())
    }
    fn lit(v: i64) -> Box<T> {
        Box::new(T::Lit(v))
    }
    #[test]
    fn capacities_cover_interior_extrema_and_arithmetic() {
        let terms = [
            T::Mul(
                Box::new(var()),
                Box::new(T::Add(lit(10), Box::new(T::Mul(lit(-1), Box::new(var()))))),
            ),
            T::TruncRem(Box::new(var()), lit(4)),
            T::CeilDiv(Box::new(var()), lit(4)),
            T::Min(Box::new(var()), lit(5)),
            T::Max(Box::new(var()), lit(5)),
            T::LessThanCast(Box::new(var()), lit(5)),
        ];
        let bounds = [('n'.into(), (1, 9))].into_iter().collect();
        for term in terms {
            let capacity = Expr(term.clone()).capacity(&bounds).unwrap();
            for n in 1..=9 {
                let dims = [('n'.into(), n)].into_iter().collect();
                assert!(usize::try_from(eval(&term, &dims).unwrap()).unwrap() <= capacity);
            }
        }
    }
    #[test]
    fn invalid_or_unbounded_allocations_refuse() {
        assert!(Expr(var()).capacity(&Bounds::new()).is_err());
        assert!(Expr(T::Lit(-1)).capacity(&Bounds::new()).is_err());
        assert!(
            Expr(T::Mul(lit(i64::MAX), lit(2)))
                .capacity(&Bounds::new())
                .is_err()
        );
        let bounds = [('n'.into(), (0, 9))].into_iter().collect();
        assert!(
            Expr(T::TruncDiv(lit(1), Box::new(var())))
                .capacity(&bounds)
                .is_err()
        );
        assert!(
            Expr(T::Var("n".into()))
                .capacity(&[('n'.into(), (9, 1))].into_iter().collect())
                .is_err()
        );
    }
    #[test]
    fn runtime_division_agrees_with_integer_semantics() {
        for a in -9i64..=9 {
            for b in -4i64..=4 {
                if b != 0 {
                    assert_eq!(
                        eval(&T::TruncDiv(lit(a), lit(b)), &Default::default()).unwrap(),
                        a / b
                    );
                    assert_eq!(
                        eval(&T::TruncRem(lit(a), lit(b)), &Default::default()).unwrap(),
                        a % b
                    );
                    assert_eq!(
                        eval(&T::CeilDiv(lit(a), lit(b)), &Default::default()).unwrap(),
                        (a as f64 / b as f64).ceil() as i64
                    );
                }
            }
        }
    }
}
