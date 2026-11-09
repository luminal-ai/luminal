//! Coordinator-owned implementation search for collective programs.
//!
//! Rank zero owns the entire compiler and sends already-bufferized physical
//! plans to profiling workers. Workers never assemble, saturate, extract,
//! DPS-rewrite, bufferize, sample, rank, or select implementations. They only
//! instantiate the coordinator's exact plan in their local CUDA context and
//! enter the same fixed warmup/trial sequence, so every authored collective
//! is launched in the same order on every rank.

use anyhow::{Context, Result};
#[cfg(feature = "device")]
use anyhow::{anyhow, ensure};
use serde::{Deserialize, Serialize};

#[cfg_attr(not(any(feature = "device", test)), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProfileRun {
    Warmup,
    Trial,
}

/// Authoritative launch sequence for every rank: one warmup followed by a
/// fixed, nonzero number of timed trials. Neither timing nor early-stop state
/// can alter this sequence on an individual rank.
#[cfg_attr(not(any(feature = "device", test)), allow(dead_code))]
pub(crate) fn fixed_profile_runs(trials: usize) -> impl Iterator<Item = ProfileRun> {
    std::iter::once(ProfileRun::Warmup).chain(std::iter::repeat_n(ProfileRun::Trial, trials.max(1)))
}

#[derive(Serialize, Deserialize)]
#[cfg(any(feature = "device", test))]
pub(crate) struct Begin {
    pub(crate) trials: usize,
    pub(crate) arena_budget_bytes: usize,
}

#[derive(Serialize, Deserialize)]
#[cfg(any(feature = "device", test))]
enum Handshake {
    Begin(Box<Begin>),
    Abort(String),
}

#[derive(Serialize, Deserialize)]
#[cfg(any(feature = "device", test))]
pub(crate) enum Command {
    Candidate(Vec<u8>),
    Finish(Vec<u8>),
    Abort(String),
}

#[cfg_attr(not(feature = "device"), allow(dead_code))]
#[derive(Serialize, Deserialize)]
enum WireOp {
    Registered { name: String, payload: Vec<u8> },
    BufferAlloc,
    BufferFree,
}

#[cfg_attr(not(feature = "device"), allow(dead_code))]
#[derive(Serialize, Deserialize)]
enum WireNode {
    BufferInput {
        slots: Vec<luminal::bufferize::InputBinding>,
    },
    Compute {
        op: WireOp,
        reads: Vec<luminal::bufferize::BufferId>,
        writes: Vec<luminal::bufferize::BufferId>,
        ties: Vec<(usize, usize)>,
        operand_info: Vec<luminal::bufferize::SlotDescriptor<luminal::layouts::DecodedLayout>>,
        result_info: Vec<luminal::bufferize::SlotDescriptor<luminal::layouts::DecodedLayout>>,
    },
    BufferCopy {
        src: luminal::bufferize::BufferId,
        dst: luminal::bufferize::BufferId,
    },
    BufferCopyValue {
        src: luminal::bufferize::BufferId,
        dst: luminal::bufferize::BufferId,
        layout: luminal::layouts::DecodedLayout,
    },
    BufferOutput {
        slots: Vec<luminal::bufferize::OutputBinding<luminal::layouts::DecodedLayout>>,
    },
}

#[cfg_attr(not(feature = "device"), allow(dead_code))]
#[derive(Serialize, Deserialize)]
struct WirePlan {
    nodes: Vec<WireNode>,
    edges: Vec<(u32, u32, luminal::bufferize::BufferEdge)>,
    buffers: Vec<(
        luminal::bufferize::BufferId,
        luminal::bufferize::Buffer<luminal::layouts::DecodedLayout>,
    )>,
    value_buffer: Vec<(
        luminal::prelude::egraph_serialize::ClassId,
        luminal::bufferize::BufferId,
    )>,
    outputs: Vec<u32>,
}

pub(crate) type WireDecoders = std::collections::HashMap<String, crate::op::DecodeWireOp>;

#[cfg_attr(not(feature = "device"), allow(dead_code))]
fn encode_op(op: &dyn luminal::buffer_tensor_ir::BufferTensorIrOp) -> Result<WireOp> {
    match op.label() {
        "BufferAlloc" => return Ok(WireOp::BufferAlloc),
        "BufferFree" => return Ok(WireOp::BufferFree),
        _ => {}
    }
    let interface = op
        .runtime_interface()
        .and_then(|interface| interface.downcast_ref::<crate::CudaOpInterface>())
        .with_context(|| format!("{} has no CUDA wire interface", op.label()))?;
    let (name, encode, _) = interface.wire_parts().with_context(|| {
        format!(
            "{} is not registered for collective plan transport",
            op.label()
        )
    })?;
    Ok(WireOp::Registered {
        name: name.into(),
        payload: encode(op)?,
    })
}

#[cfg_attr(not(feature = "device"), allow(dead_code))]
fn decode_op(
    op: WireOp,
    decoders: &WireDecoders,
) -> Result<Box<dyn luminal::buffer_tensor_ir::BufferTensorIrOp>> {
    Ok(match op {
        WireOp::Registered { name, payload } => decoders.get(&name).with_context(|| {
            format!("collective worker registry has no wire decoder for {name}")
        })?(&payload)?,
        WireOp::BufferAlloc => Box::new(luminal::buffer_tensor_ir::BufferAlloc),
        WireOp::BufferFree => Box::new(luminal::buffer_tensor_ir::BufferFree),
    })
}

#[cfg_attr(not(feature = "device"), allow(dead_code))]
pub(crate) fn encode_plan(plan: &crate::layouts::CudaPlan) -> Result<Vec<u8>> {
    use petgraph::visit::EdgeRef;
    let mut ids = std::collections::HashMap::new();
    let mut nodes = Vec::with_capacity(plan.dag.node_count());
    for node in plan.dag.node_indices() {
        let dense = u32::try_from(nodes.len()).context("collective plan has too many nodes")?;
        ids.insert(node, dense);
        nodes.push(match &plan.dag[node] {
            luminal::bufferize::BufferNode::BufferInput { slots } => WireNode::BufferInput {
                slots: slots.clone(),
            },
            luminal::bufferize::BufferNode::Compute {
                op,
                reads,
                writes,
                ties,
                operand_info,
                result_info,
            } => WireNode::Compute {
                op: encode_op(op.as_ref())?,
                reads: reads.clone(),
                writes: writes.clone(),
                ties: ties.clone(),
                operand_info: operand_info.clone(),
                result_info: result_info.clone(),
            },
            luminal::bufferize::BufferNode::BufferCopy { src, dst } => WireNode::BufferCopy {
                src: src.clone(),
                dst: dst.clone(),
            },
            luminal::bufferize::BufferNode::BufferCopyValue { src, dst, layout } => {
                WireNode::BufferCopyValue {
                    src: src.clone(),
                    dst: dst.clone(),
                    layout: layout.clone(),
                }
            }
            luminal::bufferize::BufferNode::BufferOutput { slots } => WireNode::BufferOutput {
                slots: slots.clone(),
            },
        });
    }
    let edges = plan
        .dag
        .edge_references()
        .map(|edge| {
            Ok((
                ids[&edge.source()],
                ids[&edge.target()],
                edge.weight().clone(),
            ))
        })
        .collect::<Result<_>>()?;
    let wire = WirePlan {
        nodes,
        edges,
        buffers: plan
            .buffers
            .iter()
            .map(|(id, buffer)| (id.clone(), buffer.clone()))
            .collect(),
        value_buffer: plan
            .value_buffer
            .iter()
            .map(|(value, buffer)| (value.clone(), buffer.clone()))
            .collect(),
        outputs: plan.outputs.iter().map(|node| ids[node]).collect(),
    };
    Ok(serde_json::to_vec(&wire)?)
}

#[cfg_attr(not(feature = "device"), allow(dead_code))]
pub(crate) fn decode_plan(
    bytes: &[u8],
    decoders: &WireDecoders,
) -> Result<crate::layouts::CudaPlan> {
    let wire: WirePlan =
        serde_json::from_slice(bytes).context("decode collective physical plan")?;
    let mut dag = petgraph::graph::DiGraph::new();
    let mut nodes = Vec::with_capacity(wire.nodes.len());
    for node in wire.nodes {
        let node = match node {
            WireNode::BufferInput { slots } => {
                luminal::bufferize::BufferNode::BufferInput { slots }
            }
            WireNode::Compute {
                op,
                reads,
                writes,
                ties,
                operand_info,
                result_info,
            } => luminal::bufferize::BufferNode::Compute {
                op: decode_op(op, decoders)?,
                reads,
                writes,
                ties,
                operand_info,
                result_info,
            },
            WireNode::BufferCopy { src, dst } => {
                luminal::bufferize::BufferNode::BufferCopy { src, dst }
            }
            WireNode::BufferCopyValue { src, dst, layout } => {
                luminal::bufferize::BufferNode::BufferCopyValue { src, dst, layout }
            }
            WireNode::BufferOutput { slots } => {
                luminal::bufferize::BufferNode::BufferOutput { slots }
            }
        };
        nodes.push(dag.add_node(node));
    }
    for (source, target, edge) in wire.edges {
        let source = *nodes
            .get(usize::try_from(source)?)
            .context("wire edge source out of range")?;
        let target = *nodes
            .get(usize::try_from(target)?)
            .context("wire edge target out of range")?;
        dag.add_edge(source, target, edge);
    }
    Ok(luminal::bufferize::BufferIrGraph {
        dag,
        buffers: wire.buffers.into_iter().collect(),
        value_buffer: wire.value_buffer.into_iter().collect(),
        outputs: wire
            .outputs
            .into_iter()
            .map(|node| {
                nodes
                    .get(usize::try_from(node)?)
                    .copied()
                    .context("wire output node out of range")
            })
            .collect::<Result<_>>()?,
    })
}

#[cfg(feature = "device")]
#[doc(hidden)]
pub struct Coordinator {
    group: std::rc::Rc<crate::collective::NcclCommunicator>,
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
    begun: bool,
    finished: bool,
}

#[cfg(feature = "device")]
impl Coordinator {
    pub(crate) fn new(
        group: std::rc::Rc<crate::collective::NcclCommunicator>,
        stream: std::sync::Arc<cudarc::driver::CudaStream>,
    ) -> Result<Self> {
        ensure!(
            group.rank() == 0,
            "collective search coordinator must be rank 0"
        );
        Ok(Self {
            group,
            stream,
            begun: false,
            finished: false,
        })
    }

    pub(crate) fn begin(
        &mut self,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        trials: usize,
        arena_budget_bytes: usize,
    ) -> Result<()> {
        ensure!(!self.begun, "collective search began twice");
        let payload = bincode::serialize(&Handshake::Begin(Box::new(Begin {
            trials: trials.max(1),
            arena_budget_bytes,
        })))?;
        self.group.broadcast_bytes(stream, Some(&payload))?;
        self.begun = true;
        Ok(())
    }

    pub(crate) fn candidate(
        &self,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        plan: &crate::layouts::CudaPlan,
    ) -> Result<()> {
        ensure!(self.begun, "collective search has not begun");
        self.send(stream, &Command::Candidate(encode_plan(plan)?))
    }

    pub(crate) fn finish(
        &mut self,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        plan: &crate::layouts::CudaPlan,
    ) -> Result<()> {
        ensure!(self.begun, "collective search has not begun");
        self.send(stream, &Command::Finish(encode_plan(plan)?))?;
        self.finished = true;
        ensure!(
            self.group.all_true(stream, true)?,
            "a collective profiling worker could not install the selected finalist"
        );
        Ok(())
    }

    pub(crate) fn all_ready(
        &self,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        local: bool,
    ) -> Result<bool> {
        self.group.all_true(stream, local)
    }

    pub(crate) fn max_nanos(
        &self,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        local: u64,
    ) -> Result<u64> {
        self.group.max_u64(stream, local)
    }

    fn send(
        &self,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        command: &Command,
    ) -> Result<()> {
        let payload = bincode::serialize(command)?;
        self.group.broadcast_bytes(stream, Some(&payload))?;
        Ok(())
    }
}

#[cfg(feature = "device")]
impl Drop for Coordinator {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let message = "collective search coordinator exited before selecting a finalist";
        let payload = if self.begun {
            bincode::serialize(&Command::Abort(message.into()))
        } else {
            bincode::serialize(&Handshake::Abort(message.into()))
        };
        if let Ok(payload) = payload {
            let _ = self.group.broadcast_bytes(&self.stream, Some(&payload));
        }
        self.finished = true;
    }
}

#[cfg(feature = "device")]
pub(crate) struct Worker {
    group: std::rc::Rc<crate::collective::NcclCommunicator>,
    pub(crate) begin: Begin,
}

#[cfg(feature = "device")]
impl Worker {
    pub(crate) fn receive(
        group: std::rc::Rc<crate::collective::NcclCommunicator>,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    ) -> Result<Self> {
        ensure!(
            group.rank() != 0,
            "rank 0 is the collective search coordinator"
        );
        let payload = group.broadcast_bytes(stream, None)?;
        let begin =
            match bincode::deserialize(&payload).context("decode collective search handshake")? {
                Handshake::Begin(begin) => *begin,
                Handshake::Abort(error) => {
                    return Err(anyhow!("collective search coordinator aborted: {error}"));
                }
            };
        Ok(Self { group, begin })
    }

    pub(crate) fn next(
        &self,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        decoders: &WireDecoders,
    ) -> Result<WorkerCommand> {
        let payload = self.group.broadcast_bytes(stream, None)?;
        let command: Command =
            bincode::deserialize(&payload).context("decode collective search command")?;
        match command {
            Command::Candidate(encoded) => {
                Ok(WorkerCommand::Candidate(decode_plan(&encoded, decoders)?))
            }
            Command::Finish(encoded) => Ok(WorkerCommand::Finish(decode_plan(&encoded, decoders)?)),
            Command::Abort(error) => Err(anyhow!("collective search coordinator aborted: {error}")),
        }
    }

    pub(crate) fn all_ready(
        &self,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        local: bool,
    ) -> Result<bool> {
        self.group.all_true(stream, local)
    }

    pub(crate) fn max_nanos(
        &self,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        local: u64,
    ) -> Result<u64> {
        self.group.max_u64(stream, local)
    }
}

#[cfg(feature = "device")]
pub(crate) enum WorkerCommand {
    Candidate(crate::layouts::CudaPlan),
    Finish(crate::layouts::CudaPlan),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_plan_wire_round_trips_ops_and_decoded_layouts() {
        use luminal::bufferize::{BufferId, BufferNode, SlotDescriptor};
        use luminal::dtype::PlanDtype;
        use luminal::layouts::{
            BitWidthTerm, DecodedLayout, IntExprTerm, RightMajorContiguousElementLayout, ShapeTerm,
        };

        let layout = DecodedLayout::of(
            RightMajorContiguousElementLayout {
                shape: ShapeTerm(vec![IntExprTerm::Lit(2), IntExprTerm::Lit(3)]),
                width: BitWidthTerm(32),
            },
            Some(PlanDtype::F32),
        );
        let buffer = BufferId::Allocated(0);
        let slot = SlotDescriptor {
            value: "value".into(),
            buffer: buffer.clone(),
            layout,
        };
        let mut dag = petgraph::graph::DiGraph::new();
        dag.add_node(BufferNode::Compute {
            op: Box::new(crate::ops::add::AddFunctionalDps),
            reads: vec![buffer.clone()],
            writes: vec![buffer],
            ties: vec![(0, 0)],
            operand_info: vec![slot.clone()],
            result_info: vec![slot],
        });
        let plan = luminal::bufferize::BufferIrGraph {
            dag,
            buffers: Default::default(),
            value_buffer: Default::default(),
            outputs: vec![],
        };
        let interface = crate::CudaOpInterface::kernel_wire::<crate::ops::add::AddFunctionalDps>();
        let (name, _, decode) = interface.wire_parts().unwrap();
        let decoders = [(name.into(), decode)].into_iter().collect();
        let decoded = decode_plan(&encode_plan(&plan).unwrap(), &decoders).unwrap();
        let BufferNode::Compute {
            op,
            operand_info,
            result_info,
            ..
        } = &decoded.dag[petgraph::graph::NodeIndex::new(0)]
        else {
            panic!("wire changed the compute node")
        };
        assert_eq!(op.label(), "AddFunctionalGeneric");
        assert_eq!(operand_info[0].layout, result_info[0].layout);
        assert_eq!(operand_info[0].layout.dtype, Some(PlanDtype::F32));
        assert_eq!(operand_info[0].layout.literal_extents(), Some(vec![2, 3]));
    }

    #[test]
    fn profiling_sequence_is_fixed_and_never_has_zero_trials() {
        assert_eq!(
            fixed_profile_runs(0).collect::<Vec<_>>(),
            [ProfileRun::Warmup, ProfileRun::Trial]
        );
        assert_eq!(
            fixed_profile_runs(3).collect::<Vec<_>>(),
            [
                ProfileRun::Warmup,
                ProfileRun::Trial,
                ProfileRun::Trial,
                ProfileRun::Trial,
            ]
        );
    }

    #[test]
    fn snapshot_handshake_and_candidate_commands_round_trip() {
        let payload = bincode::serialize(&Handshake::Begin(Box::new(Begin {
            trials: 3,
            arena_budget_bytes: 4096,
        })))
        .unwrap();
        let Handshake::Begin(begin) = bincode::deserialize(&payload).unwrap() else {
            panic!("begin handshake changed variant")
        };
        assert_eq!(begin.trials, 3);
        assert_eq!(begin.arena_budget_bytes, 4096);

        let payload = bincode::serialize(&Command::Candidate(vec![2, 0, 4])).unwrap();
        let Command::Candidate(candidate) = bincode::deserialize(&payload).unwrap() else {
            panic!("candidate command changed variant")
        };
        assert_eq!(candidate, [2, 0, 4]);
    }
}
