//! Validated application domains and dispatch of independently compiled values.
use crate::shape::{DimensionBounds, DynMap};
use anyhow::{Result, anyhow, ensure};

#[derive(Clone, Debug)]
pub struct BucketSpec {
    pub label: String,
    bounds: DimensionBounds,
    profile_dims: DynMap,
}

impl BucketSpec {
    pub fn new(bounds: DimensionBounds, profile_dims: DynMap) -> Result<Self> {
        bounds.validate_values(&profile_dims)?;
        ensure!(
            profile_dims.len() == bounds.symbols().len(),
            "profiling assignment contains unused dimensions"
        );
        Ok(Self {
            label: String::new(),
            bounds,
            profile_dims,
        })
    }

    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    pub fn bounds(&self) -> &DimensionBounds {
        &self.bounds
    }
    pub fn profile_dims(&self) -> &DynMap {
        &self.profile_dims
    }
}

/// A disjoint collection of application-owned values. Construction validates
/// whole domains; gaps are permitted. No compilation occurs on a dispatch miss.
#[derive(Debug)]
pub struct BucketSet<T> {
    entries: Vec<(BucketSpec, T)>,
}

impl<T> BucketSet<T> {
    pub fn new(entries: Vec<(BucketSpec, T)>) -> Result<Self> {
        ensure!(!entries.is_empty(), "no application buckets");
        let symbols = entries[0].0.bounds.symbols();
        for (i, (spec, _)) in entries.iter().enumerate() {
            ensure!(
                spec.bounds.symbols() == symbols,
                "bucket {i} uses different dimensions"
            );
            for (j, (previous, _)) in entries[..i].iter().enumerate() {
                let overlap = spec.bounds.iter().all(|(symbol, range)| {
                    let other = previous.bounds.get(symbol).unwrap();
                    range.min() <= other.max() && other.min() <= range.max()
                });
                ensure!(!overlap, "application buckets {j} and {i} overlap");
            }
        }
        Ok(Self { entries })
    }

    pub fn entries(&self) -> &[(BucketSpec, T)] {
        &self.entries
    }

    pub fn select_index(&self, dims: &DynMap) -> Result<usize> {
        for symbol in self.entries[0].0.bounds.symbols() {
            ensure!(dims.contains_key(&symbol), "missing dimension `{symbol}`");
        }
        self.entries
            .iter()
            .position(|(spec, _)| spec.bounds.validate_values(dims).is_ok())
            .ok_or_else(|| anyhow!("no application bucket covers dimensions {dims:?}"))
    }

    pub fn select(&self, dims: &DynMap) -> Result<&T> {
        Ok(&self.entries[self.select_index(dims)?].1)
    }

    pub fn select_mut(&mut self, dims: &DynMap) -> Result<&mut T> {
        let index = self.select_index(dims)?;
        Ok(&mut self.entries[index].1)
    }

    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.entries.iter_mut().map(|(_, value)| value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(q: (usize, usize), c: (usize, usize)) -> BucketSpec {
        BucketSpec::new(
            DimensionBounds::from_ranges([("q".into(), q), ("c".into(), c)]).unwrap(),
            [("q".into(), q.0), ("c".into(), c.0)].into_iter().collect(),
        )
        .unwrap()
    }

    #[test]
    fn multidimensional_overlap_and_gaps() {
        let buckets = BucketSet::new(vec![
            (spec((1, 1), (1, 4096)), 0),
            (spec((2, 128), (1, 4096)), 1),
        ])
        .unwrap();
        for q in [1, 2, 128] {
            let dims = [("q".into(), q), ("c".into(), 4096)].into_iter().collect();
            assert_eq!(*buckets.select(&dims).unwrap(), usize::from(q > 1));
        }
        assert!(
            buckets
                .select(&[("q".into(), 129), ("c".into(), 4)].into_iter().collect())
                .is_err()
        );
        assert!(
            buckets
                .select(&[("q".into(), 1)].into_iter().collect())
                .is_err()
        );
        assert!(
            BucketSet::new(vec![(spec((1, 2), (1, 4)), ()), (spec((2, 3), (4, 8)), ())]).is_err()
        );
        // Domain ordering need not be sorted and a gap is legal.
        assert!(
            BucketSet::new(vec![(spec((1, 2), (5, 8)), ()), (spec((1, 2), (1, 3)), ())]).is_ok()
        );
    }

    #[test]
    fn static_domains_and_invalid_profiles() {
        let spec = BucketSpec::new(Default::default(), Default::default()).unwrap();
        assert!(BucketSet::new(vec![(spec.clone(), ())]).is_ok());
        assert!(BucketSet::new(vec![(spec.clone(), ()), (spec, ())]).is_err());
        assert!(BucketSet::<()>::new(vec![]).is_err());
        assert!(
            BucketSpec::new(
                DimensionBounds::from_ranges([("q".into(), (2, 3))]).unwrap(),
                [("q".into(), 1)].into_iter().collect()
            )
            .is_err()
        );
    }
}
