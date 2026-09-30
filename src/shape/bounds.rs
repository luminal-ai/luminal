//! The dimension domain of one compiled program.
use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, anyhow, ensure};

use super::{DynMap, Symbol};

/// An inclusive, representable interval for a dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DimensionRange {
    min: usize,
    max: usize,
}

impl DimensionRange {
    pub fn new(min: usize, max: usize) -> Result<Self> {
        ensure!(min <= max, "empty dimension range [{min}, {max}]");
        i64::try_from(max).map_err(|_| anyhow!("dimension range exceeds i64: {max}"))?;
        Ok(Self { min, max })
    }

    pub fn exact(value: usize) -> Result<Self> {
        Self::new(value, value)
    }

    pub fn min(self) -> usize {
        self.min
    }
    pub fn max(self) -> usize {
        self.max
    }

    pub fn contains(self, value: usize) -> bool {
        self.min <= value && value <= self.max
    }

    pub fn intersect(self, other: Self) -> Result<Self> {
        Self::new(self.min.max(other.min), self.max.min(other.max))
    }
}

/// Complete bounds for the symbolic dimensions of one program.
/// Profiling and execution assignments are supplied separately.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DimensionBounds(BTreeMap<Symbol, DimensionRange>);

impl DimensionBounds {
    pub fn new(ranges: impl IntoIterator<Item = (Symbol, DimensionRange)>) -> Result<Self> {
        let mut result = BTreeMap::new();
        for (symbol, range) in ranges {
            ensure!(
                result.insert(symbol, range).is_none(),
                "duplicate dimension `{symbol}`"
            );
        }
        Ok(Self(result))
    }

    pub fn from_ranges(ranges: impl IntoIterator<Item = (Symbol, (usize, usize))>) -> Result<Self> {
        Self::new(
            ranges
                .into_iter()
                .map(|(s, (lo, hi))| Ok((s, DimensionRange::new(lo, hi)?)))
                .collect::<Result<Vec<_>>>()?,
        )
    }

    pub fn exact(values: &DynMap) -> Result<Self> {
        Self::from_ranges(values.iter().map(|(&s, &v)| (s, (v, v))))
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Symbol, &DimensionRange)> {
        self.0.iter()
    }
    pub fn get(&self, symbol: &Symbol) -> Option<DimensionRange> {
        self.0.get(symbol).copied()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn symbols(&self) -> BTreeSet<Symbol> {
        self.0.keys().copied().collect()
    }

    /// Check all required values; unrelated application dimensions are allowed.
    pub fn validate_values(&self, values: &DynMap) -> Result<()> {
        for (symbol, range) in &self.0 {
            let value = values
                .get(symbol)
                .ok_or_else(|| anyhow!("missing dimension `{symbol}`"))?;
            ensure!(
                range.contains(*value),
                "dimension `{symbol}`={value} is outside [{}, {}]",
                range.min,
                range.max
            );
        }
        Ok(())
    }

    pub fn project(&self, symbols: &BTreeSet<Symbol>) -> Result<Self> {
        Self::new(
            symbols
                .iter()
                .map(|s| {
                    Ok((
                        *s,
                        self.get(s)
                            .ok_or_else(|| anyhow!("missing bounds for dimension `{s}`"))?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?,
        )
    }

    pub fn validate_symbols(&self, symbols: &BTreeSet<Symbol>) -> Result<()> {
        let provided = self.symbols();
        let missing: Vec<_> = symbols.difference(&provided).collect();
        let extra: Vec<_> = provided.difference(symbols).collect();
        ensure!(
            missing.is_empty() && extra.is_empty(),
            "dimension bounds differ from program: missing {missing:?}, unused {extra:?}"
        );
        Ok(())
    }

    pub fn signed_intervals(&self) -> crate::layouts::DimBounds {
        self.0
            .iter()
            .map(|(&s, r)| (s, (r.min as i64, r.max as i64)))
            .collect()
    }

    pub fn ranges(&self) -> BTreeMap<Symbol, (usize, usize)> {
        self.0.iter().map(|(&s, r)| (s, (r.min, r.max))).collect()
    }

    /// Ordinary function facts, emitted before saturation.
    pub fn egglog_seeds(&self) -> String {
        let mut text = String::new();
        for (symbol, range) in &self.0 {
            let symbol = symbol.egglog_literal();
            use std::fmt::Write;
            writeln!(
                text,
                "(set (lower-bound-of (IntVar {symbol})) (bigint {}))",
                range.min
            )
            .unwrap();
            writeln!(
                text,
                "(set (upper-bound-of (IntVar {symbol})) (bigint {}))",
                range.max
            )
            .unwrap();
        }
        text
    }
}

/// Read dimension references from a bound program before adding bounds or
/// running rewrites. Includes index/stride/value expressions and contracts.
pub fn program_dimensions(program: &str) -> Result<BTreeSet<Symbol>> {
    use egglog::ast::{Expr, Literal};
    let commands = egglog::EGraph::default()
        .parse_program(None, program)
        .map_err(|e| anyhow!("parsing dimension references: {e}"))?;
    let mut names = BTreeSet::new();
    for command in commands {
        command.visit_exprs(&mut |expr| {
            if let Expr::Call(_, name, args) = &expr
                && name == "IntVar"
                && let [Expr::Lit(_, Literal::String(name))] = args.as_slice()
            {
                names.insert(name.clone());
            }
            expr
        });
    }
    names
        .into_iter()
        .map(|name| Symbol::try_new_dim(name).map_err(Into::into))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_domain_and_assignments() {
        assert!(DimensionRange::new(9, 2).is_err());
        assert!(DimensionRange::new(0, usize::MAX).is_err());
        let bounds =
            DimensionBounds::from_ranges([("n".into(), (0, 9)), ("exact".into(), (1, 1))]).unwrap();
        assert!(
            bounds
                .validate_values(&[("n".into(), 0), ("exact".into(), 1)].into_iter().collect())
                .is_ok()
        );
        assert!(
            bounds
                .validate_values(
                    &[("n".into(), 10), ("exact".into(), 1)]
                        .into_iter()
                        .collect()
                )
                .is_err()
        );
        assert!(
            bounds
                .validate_values(&[("n".into(), 3)].into_iter().collect())
                .is_err()
        );
        assert!(
            DimensionRange::new(2, 8)
                .unwrap()
                .intersect(DimensionRange::exact(1).unwrap())
                .is_err()
        );
    }

    #[test]
    fn reads_nested_and_escaped_symbols_without_saturation() {
        let symbol = Symbol::from("stride\"\\\n");
        let text = format!(
            "(let x (Mul (IntVar {}) (Add (IntVar \"n\") (ILit 1))))",
            symbol.egglog_literal()
        );
        let symbols = program_dimensions(&text).unwrap();
        assert_eq!(symbols, [symbol, "n".into()].into_iter().collect());
        let bounds =
            DimensionBounds::from_ranges([(symbol, (1, 8)), ("n".into(), (2, 9))]).unwrap();
        bounds.validate_symbols(&symbols).unwrap();
        assert_eq!(program_dimensions(&bounds.egglog_seeds()).unwrap(), symbols);
        assert!(
            DimensionBounds::default()
                .validate_symbols(&symbols)
                .is_err()
        );
    }
}
