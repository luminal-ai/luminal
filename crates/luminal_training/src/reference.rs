//! Zero-copy training-loop utilities for [`ReferenceRuntime`].
//!
//! After an `execute`, updated parameters and optimizer state live in the
//! runtime's Output buffers, and every Input slot has been consumed. Instead
//! of cloning the new values out to the host and feeding them back with
//! `set_data`, [`OptimizerStep::rebind`] *moves* each Output buffer into the
//! corresponding Input slot — a map-entry move, no data copy — so weights and
//! optimizer state stay resident in the runtime across steps. The host only
//! feeds per-step data (batches, step-size scalars).
//!
//! Evaluation runs execute the same graph and therefore consume the resident
//! Input buffers (and produce bogus "updated" params from the eval batch).
//! Wrap them with [`snapshot_inputs`] / [`restore_inputs`], which clone only
//! for the eval — training steps stay copy-free.

use luminal::hlir::{Input, Output, ReferenceData, ReferenceRuntime};
use luminal::prelude::*;

use crate::optim::OptimizerStep;

/// Find the runtime-local node for the Input created from HLIR node `id`.
fn find_input(rt: &ReferenceRuntime, id: NodeIndex) -> NodeIndex {
    rt.graph
        .node_indices()
        .find(|n| {
            (**rt.graph[*n])
                .as_any()
                .downcast_ref::<Input>()
                .is_some_and(|i| i.node == id.index())
        })
        .unwrap_or_else(|| panic!("{id:?} is not an Input node in the compiled graph"))
}

/// Find the runtime-local Output node whose source is HLIR node `id`.
fn find_output(rt: &ReferenceRuntime, id: NodeIndex) -> NodeIndex {
    rt.graph
        .node_indices()
        .find(|n| {
            (**rt.graph[*n])
                .as_any()
                .downcast_ref::<Output>()
                .is_some_and(|o| o.node == id.index())
        })
        .unwrap_or_else(|| panic!("{id:?} has no Output node in the compiled graph"))
}

/// Move the buffer produced at Output `from` (an HLIR node marked `.output()`)
/// into the Input slot of HLIR node `to`, without copying the data.
pub fn rebind_buffer(rt: &mut ReferenceRuntime, from: NodeIndex, to: NodeIndex) {
    let out = find_output(rt, from);
    let inp = find_input(rt, to);
    let buf = rt
        .buffers
        .remove(&out)
        .unwrap_or_else(|| panic!("no buffer at output {from:?} — did execute run?"));
    rt.buffers.insert(inp, buf);
}

/// Clone the resident Input buffers for `tensors` (e.g. before an eval
/// execute, which will consume them).
pub fn snapshot_inputs(rt: &ReferenceRuntime, tensors: &[GraphTensor]) -> Vec<ReferenceData> {
    tensors
        .iter()
        .map(|t| {
            let inp = find_input(rt, t.id);
            rt.buffers
                .get(&inp)
                .unwrap_or_else(|| panic!("no resident buffer for input {:?}", t.id))
                .clone()
        })
        .collect()
}

/// Re-insert previously snapshotted buffers into their Input slots. Clones
/// from the snapshot so the same snapshot can restore repeatedly (e.g. one
/// snapshot, many eval batches).
pub fn restore_inputs(rt: &mut ReferenceRuntime, tensors: &[GraphTensor], bufs: &[ReferenceData]) {
    assert_eq!(tensors.len(), bufs.len());
    for (t, b) in tensors.iter().zip(bufs) {
        let inp = find_input(rt, t.id);
        rt.buffers.insert(inp, b.clone());
    }
}

impl OptimizerStep {
    /// Advance one training step without copying: move each updated-parameter
    /// buffer into its parameter's Input slot and each updated state buffer
    /// into its state Input slot, ready for the next `execute`. Call after
    /// `execute`, with `params` aligned as passed to [`Optimizer::build`].
    ///
    /// [`Optimizer::build`]: crate::Optimizer::build
    pub fn rebind(&self, rt: &mut ReferenceRuntime, params: &[GraphTensor]) {
        assert_eq!(params.len(), self.new_params.len());
        for (np, p) in self.new_params.iter().zip(params) {
            rebind_buffer(rt, np.id, p.id);
        }
        for (so, si) in self.state_out.iter().zip(&self.state_in) {
            rebind_buffer(rt, so.id, si.id);
        }
    }
}

/// A compiled training session on the reference runtime.
///
/// `new` appends the backward pass and the optimizer's update rule to the
/// graph, compiles it, and zeroes the optimizer state. From then on weights
/// and optimizer state stay resident in the runtime: [`Trainer::step`]
/// advances them by rebinding output buffers to input slots (no copies).
/// Feed each parameter once with [`Trainer::set_data`] before stepping, and
/// mark any tensors you want to [`Trainer::read`] as `.output()` before
/// calling `new`.
pub struct Trainer<O> {
    pub rt: ReferenceRuntime,
    dyn_map: FxHashMap<char, usize>,
    step: OptimizerStep,
    params: Vec<GraphTensor>,
    resident: Vec<GraphTensor>,
    loss_out: GraphTensor,
    opt: O,
    t: usize,
}

impl<O: crate::Optimizer> Trainer<O> {
    pub fn new(cx: &mut Graph, loss: GraphTensor, params: &[GraphTensor], opt: O) -> Self {
        let grads = crate::backward(cx, loss, params);
        let step = opt.build(cx, params, &grads);
        let loss_out = loss.output();
        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        let opts = CompileOptions::default().search_graph_limit(1);
        let mut rt = cx.search(ReferenceRuntime::default(), opts);
        for (s, v) in step.state_in.iter().zip(&step.state_init) {
            rt.set_data(s.id, v.clone());
        }
        Self {
            rt,
            dyn_map: cx.dyn_map.clone(),
            params: params.to_vec(),
            resident: params.iter().chain(&step.state_in).copied().collect(),
            step,
            loss_out,
            opt,
            t: 0,
        }
    }

    /// Feed an input tensor's buffer for the next execute.
    pub fn set_data(&mut self, t: GraphTensor, data: Vec<f32>) {
        self.rt.set_data(t.id, data);
    }

    /// Execute on the currently-fed batch without advancing weights (this
    /// consumes the resident buffers — wrap with [`Trainer::eval`] if
    /// training continues afterwards). Returns the loss.
    pub fn forward(&mut self) -> f32 {
        for (s, v) in self
            .step
            .scalar_in
            .iter()
            .zip(self.opt.scalar_values(self.t))
        {
            self.rt.set_data(s.id, vec![v]);
        }
        self.rt.execute(&self.dyn_map);
        self.rt.get_f32(self.loss_out.id)[0]
    }

    /// Forward + backward + optimizer update on the currently-fed batch;
    /// rebinds the updated weights/state for the next step and returns loss.
    pub fn step(&mut self) -> f32 {
        let loss = self.forward();
        self.step.rebind(&mut self.rt, &self.params);
        self.t += 1;
        loss
    }

    /// Run `f` (e.g. an eval forward) with the resident weights and state
    /// snapshotted before and restored after, so it cannot disturb training.
    pub fn eval<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        let snap = snapshot_inputs(&self.rt, &self.resident);
        let out = f(self);
        restore_inputs(&mut self.rt, &self.resident, &snap);
        out
    }

    /// Read an output-marked tensor's buffer from the last execute.
    pub fn read(&self, t: GraphTensor) -> Vec<f32> {
        self.rt.get_f32(t.id).clone()
    }

    /// Feed `inputs` and take one training step. Returns the loss.
    pub fn step_with(&mut self, inputs: &[(GraphTensor, Vec<f32>)]) -> f32 {
        for (t, d) in inputs {
            self.set_data(*t, d.clone());
        }
        self.step()
    }

    /// Feed `inputs`, run one evaluation forward pass (guarded by
    /// [`Trainer::eval`] so training state is untouched), and read `out`.
    pub fn eval_forward(
        &mut self,
        inputs: &[(GraphTensor, Vec<f32>)],
        out: GraphTensor,
    ) -> Vec<f32> {
        self.eval(|tr| {
            for (t, d) in inputs {
                tr.set_data(*t, d.clone());
            }
            tr.forward();
            tr.read(out)
        })
    }
}
