//! Optimizers as HLIR graph transformations.
//!
//! An optimizer builds its update rule directly into the graph:
//! `w_new = f(w, grad, state)` becomes ordinary HLIR ops marked as outputs,
//! so the update compiles and fuses with the forward+backward pass on any
//! runtime. Optimizer state (momentum buffers, Adam moments) enters the graph
//! as inputs and exits as outputs; the host loop feeds each step's state back
//! in, exactly like it rebinds weights.
//!
//! Step-dependent scalars (the learning rate, Adam's bias-corrected step
//! size) are scalar graph inputs fed from [`Optimizer::scalar_values`], so
//! learning-rate schedules never require recompiling.
//!
//! ```no_run
//! use luminal::prelude::*;
//! use luminal_training::{Backward, Optimizer, SGD};
//!
//! let mut cx = Graph::new();
//! let w = cx.tensor((3, 4));
//! let x = cx.tensor((2, 3));
//! let loss = x.matmul(w).sum((0, 1));
//! let grads = cx.backward(loss, &[w]);
//! let opt = SGD::new(1e-2).momentum(0.9);
//! let step = opt.build(&mut cx, &[w], &grads);
//! // ...compile; each iteration: feed w, state_in, and opt.scalar_values(t),
//! // execute, then read step.new_params / step.state_out back.
//! ```

use luminal::prelude::*;

/// Handle to an optimizer's in-graph update step. All tensors here are
/// already marked as outputs where the host needs to read them.
pub struct OptimizerStep {
    /// Updated parameters, aligned with the `params` passed to `build`.
    /// Read these after `execute` and feed them back as the params next step.
    pub new_params: Vec<GraphTensor>,
    /// State buffers to feed each step (initialize from `state_init`).
    pub state_in: Vec<GraphTensor>,
    /// Updated state buffers, aligned with `state_in`. Read after `execute`
    /// and feed back into `state_in` next step.
    pub state_out: Vec<GraphTensor>,
    /// Scalar inputs to feed with [`Optimizer::scalar_values`] each step.
    pub scalar_in: Vec<GraphTensor>,
    /// Initial values for `state_in` (zeros).
    pub state_init: Vec<Vec<f32>>,
}

pub trait Optimizer {
    /// Append this optimizer's update ops to the graph. Call after
    /// `backward`, before compiling.
    fn build(&self, cx: &mut Graph, params: &[GraphTensor], grads: &[GraphTensor])
    -> OptimizerStep;

    /// Scalar values to feed into `scalar_in` for iteration `step`
    /// (0-indexed), aligned by position.
    fn scalar_values(&self, step: usize) -> Vec<f32>;
}

/// Broadcast a scalar (0-dim) input across `dims` as a free stride-0 view.
fn broadcast_scalar(mut s: GraphTensor, dims: &[Expression]) -> GraphTensor {
    for (i, d) in dims.iter().enumerate() {
        s = s.expand_dim(i, *d);
    }
    s
}

fn static_size(p: &GraphTensor) -> usize {
    p.dims()
        .iter()
        .map(|d| {
            d.to_usize()
                .expect("optimizer state requires statically-shaped parameters")
        })
        .product()
}

/// Stochastic gradient descent with optional momentum and (coupled) weight
/// decay, matching `torch.optim.SGD` semantics (dampening 0, non-Nesterov):
///
/// ```text
/// g      = grad + weight_decay · w
/// buf    = momentum · buf + g
/// w_new  = w − lr · buf
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SGD {
    pub lr: f32,
    pub momentum: f32,
    pub weight_decay: f32,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
        }
    }
    pub fn momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }
    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for SGD {
    fn build(
        &self,
        cx: &mut Graph,
        params: &[GraphTensor],
        grads: &[GraphTensor],
    ) -> OptimizerStep {
        assert_eq!(params.len(), grads.len());
        let lr = cx.tensor(());
        let mut step = OptimizerStep {
            new_params: vec![],
            state_in: vec![],
            state_out: vec![],
            scalar_in: vec![lr],
            state_init: vec![],
        };
        for (w, g) in params.iter().zip(grads) {
            let dims = w.dims();
            let mut g_eff = *g;
            if self.weight_decay != 0.0 {
                g_eff += *w * self.weight_decay;
            }
            let direction = if self.momentum != 0.0 {
                let buf = cx.tensor(dims.clone());
                let buf_new = (buf * self.momentum + g_eff).output();
                step.state_in.push(buf);
                step.state_out.push(buf_new);
                step.state_init.push(vec![0.0; static_size(w)]);
                buf_new
            } else {
                g_eff
            };
            let new_w = (*w - broadcast_scalar(lr, &dims) * direction).output();
            step.new_params.push(new_w);
        }
        step
    }

    fn scalar_values(&self, _step: usize) -> Vec<f32> {
        vec![self.lr]
    }
}

/// AdamW: Adam with decoupled weight decay, using the Adam paper's efficient
/// formulation (bias correction folded into the fed step size `α_t`):
///
/// ```text
/// m      = β₁·m + (1−β₁)·g
/// v      = β₂·v + (1−β₂)·g²
/// α_t    = lr · √(1−β₂ᵗ) / (1−β₁ᵗ)          (computed host-side, fed as a scalar)
/// w_new  = w − α_t · m/(√v + ε) − lr·λ·w
/// ```
///
/// Plain Adam is `weight_decay = 0` (the default).
#[derive(Debug, Clone, Copy)]
pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl AdamW {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }
    pub fn betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }
    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }
    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for AdamW {
    fn build(
        &self,
        cx: &mut Graph,
        params: &[GraphTensor],
        grads: &[GraphTensor],
    ) -> OptimizerStep {
        assert_eq!(params.len(), grads.len());
        let alpha_t = cx.tensor(());
        let mut step = OptimizerStep {
            new_params: vec![],
            state_in: vec![],
            state_out: vec![],
            scalar_in: vec![alpha_t],
            state_init: vec![],
        };
        // Decoupled decay uses the unscheduled base lr, fed separately so
        // schedules on α_t don't have to alter the decay strength.
        let lr = if self.weight_decay != 0.0 {
            let lr = cx.tensor(());
            step.scalar_in.push(lr);
            Some(lr)
        } else {
            None
        };
        for (w, g) in params.iter().zip(grads) {
            let dims = w.dims();
            let size = static_size(w);
            let m_in = cx.tensor(dims.clone());
            let v_in = cx.tensor(dims.clone());
            let m_new = (m_in * self.beta1 + *g * (1.0 - self.beta1)).output();
            let v_new = (v_in * self.beta2 + *g * *g * (1.0 - self.beta2)).output();
            let update = m_new / (v_new.sqrt() + self.eps);
            let mut new_w = *w - broadcast_scalar(alpha_t, &dims) * update;
            if let Some(lr) = lr {
                new_w -= broadcast_scalar(lr, &dims) * (*w * self.weight_decay);
            }
            step.new_params.push(new_w.output());
            step.state_in.push(m_in);
            step.state_in.push(v_in);
            step.state_out.push(m_new);
            step.state_out.push(v_new);
            step.state_init.push(vec![0.0; size]);
            step.state_init.push(vec![0.0; size]);
        }
        step
    }

    fn scalar_values(&self, step: usize) -> Vec<f32> {
        let t = (step + 1) as i32; // bias correction is 1-indexed
        let alpha_t = self.lr * (1.0 - self.beta2.powi(t)).sqrt() / (1.0 - self.beta1.powi(t));
        let mut v = vec![alpha_t];
        if self.weight_decay != 0.0 {
            v.push(self.lr);
        }
        v
    }
}
