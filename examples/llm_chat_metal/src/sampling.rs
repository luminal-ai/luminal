use anyhow::{Result, ensure};
use rand::{Rng, SeedableRng, rngs::StdRng};

pub struct Sampler {
    temperature: f32,
    top_p: f32,
    rng: StdRng,
}
impl Sampler {
    pub fn new(temperature: f32, top_p: f32, seed: u64) -> Result<Self> {
        ensure!(
            temperature.is_finite() && temperature >= 0.,
            "temperature must be finite and nonnegative"
        );
        ensure!(
            top_p.is_finite() && top_p > 0. && top_p <= 1.,
            "top-p must be in (0,1]"
        );
        Ok(Self {
            temperature,
            top_p,
            rng: StdRng::seed_from_u64(seed),
        })
    }
    pub fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        ensure!(
            !logits.is_empty() && logits.len() <= u32::MAX as usize,
            "invalid logits length"
        );
        ensure!(
            logits.iter().all(|v| !v.is_nan() && *v != f32::INFINITY),
            "model produced NaN/+inf logits"
        );
        let (best, &max) = logits
            .iter()
            .enumerate()
            .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
            .unwrap();
        ensure!(max.is_finite(), "model produced no finite logits");
        if self.temperature == 0. {
            return Ok(best as u32);
        }
        let mut probabilities: Vec<_> = logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, ((v as f64 - max as f64) / self.temperature as f64).exp()))
            .collect();
        probabilities.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let total: f64 = probabilities.iter().map(|p| p.1).sum();
        let cutoff = total * self.top_p as f64;
        let mut mass = 0.;
        let mut keep = 0;
        for p in &probabilities {
            mass += p.1;
            keep += 1;
            if mass >= cutoff {
                break;
            }
        }
        let mut draw = self.rng.random::<f64>() * mass;
        for &(id, p) in &probabilities[..keep] {
            if draw < p {
                return Ok(id as u32);
            }
            draw -= p;
        }
        Ok(probabilities[keep - 1].0 as u32)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn greedy_ties_invalid_logits_and_seeded_sampling() {
        assert_eq!(
            Sampler::new(0., 1., 0)
                .unwrap()
                .sample(&[2., 2., 1.])
                .unwrap(),
            0
        );
        assert!(
            Sampler::new(0., 1., 0)
                .unwrap()
                .sample(&[f32::NAN])
                .is_err()
        );
        let mut a = Sampler::new(0.8, 0.9, 7).unwrap();
        let mut b = Sampler::new(0.8, 0.9, 7).unwrap();
        for _ in 0..20 {
            assert_eq!(
                a.sample(&[0., 1., 2.]).unwrap(),
                b.sample(&[0., 1., 2.]).unwrap()
            );
        }
        assert_eq!(
            Sampler::new(1., 0.01, 0)
                .unwrap()
                .sample(&[0., 1., 2.])
                .unwrap(),
            2
        );
    }
}
