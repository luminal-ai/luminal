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
