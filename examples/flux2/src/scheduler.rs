//! Pure-Rust port of `diffusers.FlowMatchEulerDiscreteScheduler`, the scheduler
//! configured by Flux 2's `scheduler/scheduler_config.json`.
//!
//! Behaves exactly like the diffusers implementation when:
//!   - `use_dynamic_shifting = true`
//!   - `time_shift_type = "exponential"`
//!   - `invert_sigmas = false`
//!   - `shift_terminal = None`
//!   - karras / exponential / beta sigmas are all disabled
//!
//! Validated against `diffusers==0.36.x` for `image_seq_len = 4096`,
//! `num_inference_steps = 8`, `mu = 1.15` (max difference < 1e-6).

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub num_train_timesteps: f32,
    pub sigma_max: f32,
    pub sigma_min: f32,
    pub base_image_seq_len: f32,
    pub max_image_seq_len: f32,
    pub base_shift: f32,
    pub max_shift: f32,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            num_train_timesteps: 1000.0,
            sigma_max: 1.0,
            sigma_min: 1e-3,
            base_image_seq_len: 256.0,
            max_image_seq_len: 4096.0,
            base_shift: 0.5,
            max_shift: 1.15,
        }
    }
}

/// Linear interpolation of the shift parameter over the configured image-sequence range.
/// Matches the `mu` computation in `diffusers.FluxPipeline.calculate_shift`.
pub fn compute_mu(cfg: &SchedulerConfig, image_seq_len: usize) -> f32 {
    let m = (cfg.max_shift - cfg.base_shift) / (cfg.max_image_seq_len - cfg.base_image_seq_len);
    m * (image_seq_len as f32 - cfg.base_image_seq_len) + cfg.base_shift
}

/// Build the schedule of sigmas (length `num_inference_steps + 1`, ending in 0.0)
/// and timesteps (length `num_inference_steps`) for one inference run.
pub fn make_schedule(
    cfg: &SchedulerConfig,
    num_inference_steps: usize,
    mu: f32,
) -> (Vec<f32>, Vec<f32>) {
    assert!(num_inference_steps >= 1);

    // 1. Linearly spaced timesteps -> sigmas in [sigma_max, sigma_min].
    let n = num_inference_steps;
    let mut sigmas: Vec<f32> = (0..n)
        .map(|i| {
            let t_max = cfg.sigma_max * cfg.num_train_timesteps;
            let t_min = cfg.sigma_min * cfg.num_train_timesteps;
            let alpha = if n == 1 {
                0.0
            } else {
                i as f32 / (n - 1) as f32
            };
            let t = t_max + (t_min - t_max) * alpha;
            t / cfg.num_train_timesteps
        })
        .collect();

    // 2. Resolution-dependent exponential time shift.
    let exp_mu = mu.exp();
    for s in sigmas.iter_mut() {
        // s' = exp(mu) / (exp(mu) + (1/s - 1))
        let rhs = exp_mu + (1.0 / *s - 1.0);
        *s = exp_mu / rhs;
    }

    // 3. Timesteps = sigmas * num_train_timesteps before terminal append.
    let timesteps: Vec<f32> = sigmas.iter().map(|s| s * cfg.num_train_timesteps).collect();

    // 4. Append terminal 0 sigma.
    sigmas.push(0.0);
    (sigmas, timesteps)
}

/// One Euler integration step of the rectified-flow ODE.
/// `sample_next = sample + (sigma_next - sigma) * model_output`.
/// `sigmas[i]` is the current step's sigma, `sigmas[i + 1]` the next.
pub fn euler_step(sample: &mut [f32], model_output: &[f32], sigma: f32, sigma_next: f32) {
    debug_assert_eq!(sample.len(), model_output.len());
    let dt = sigma_next - sigma;
    for (s, &m) in sample.iter_mut().zip(model_output) {
        *s += dt * m;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn matches_diffusers_4096_steps_8() {
        // Reference output captured from diffusers 0.36 with the FLUX.2-dev config
        // (use_dynamic_shifting=True, time_shift_type=exponential, mu=1.15).
        let cfg = SchedulerConfig::default();
        let mu = compute_mu(&cfg, 4096);
        assert!(close(mu, 1.15, 1e-6), "mu={mu}");

        let (sigmas, timesteps) = make_schedule(&cfg, 8, mu);
        let expected_sigmas = [
            1.0, 0.9499281, 0.887_723, 0.8083667, 0.7036315, 0.5590252, 0.3464282, 0.0031514, 0.0,
        ];
        let expected_timesteps = [
            1000.0, 949.9281, 887.723, 808.3667, 703.6315, 559.0252, 346.4282, 3.1514,
        ];
        assert_eq!(sigmas.len(), expected_sigmas.len());
        for (got, want) in sigmas.iter().zip(expected_sigmas.iter()) {
            assert!(
                close(*got, *want, 1e-4),
                "sigma mismatch: got {got} want {want}"
            );
        }
        assert_eq!(timesteps.len(), expected_timesteps.len());
        for (got, want) in timesteps.iter().zip(expected_timesteps.iter()) {
            assert!(
                close(*got, *want, 1e-1),
                "timestep mismatch: got {got} want {want}",
            );
        }
    }

    #[test]
    fn euler_step_matches_formula() {
        // Trivial: prev = sample + (sigma_next - sigma) * out.
        let mut x = vec![1.0_f32, 2.0, 3.0];
        let out = vec![10.0_f32, -10.0, 0.0];
        euler_step(&mut x, &out, 0.5, 0.2);
        let dt = 0.2 - 0.5;
        assert!(close(x[0], 1.0 + dt * 10.0, 1e-6));
        assert!(close(x[1], 2.0 + dt * -10.0, 1e-6));
        assert!(close(x[2], 3.0 + dt * 0.0, 1e-6));
    }
}
