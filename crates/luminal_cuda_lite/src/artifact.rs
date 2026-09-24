//! Stable, device-free wire format for an installed CUDA-lite plan.
//!
//! CUDA modules, streams, pointers, arenas and graph captures are deliberately
//! absent.  Loading reconstructs the typed plan; device installation and
//! kernel compilation remain lazy on the newly bound runtime.

use anyhow::{Context, Result, bail, ensure};
use luminal::buffer_tensor_ir::{BufferAlloc, BufferFree, BufferTensorIrOp};
use luminal::bufferize::{
    Buffer, BufferEdge, BufferId, BufferIrGraph, BufferNode, InputBinding, OutputBinding,
    SlotDescriptor,
};
use luminal::egglog_utils::eclass::Spellings;
use luminal::layouts::{
    BitOffsetExpressionLayout, DecodedLayout, ElementOffsetExpressionLayout, Layout, LayoutFacts,
    LeftMajorContiguousElementLayout, RightMajorContiguousElementLayout, StridedElementLayout,
};
use luminal::prelude::egraph_serialize::ClassId;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use crate::runtime::SearchedPlanTemplate;

pub const ARTIFACT_SCHEMA: u32 = 1;
pub const OP_REGISTRY_ABI: u32 = 1;

#[derive(Debug, Serialize, Deserialize)]
struct Envelope {
    schema: u32,
    op_registry_abi: u32,
    crate_version: String,
    fingerprint: String,
    plan: PlanWire,
    kernel_sources: Vec<KernelSourcesWire>,
    bounds: crate::symbolic::Bounds,
    device_budget_bytes: Option<usize>,
}

pub fn serialize(template: &SearchedPlanTemplate, fingerprint: &str) -> Result<Vec<u8>> {
    let envelope = Envelope {
        schema: ARTIFACT_SCHEMA,
        op_registry_abi: OP_REGISTRY_ABI,
        crate_version: env!("CARGO_PKG_VERSION").to_string(),
        fingerprint: fingerprint.to_string(),
        plan: PlanWire::encode(&template.plan)?,
        kernel_sources: kernel_sources(&template.plan, &template.bounds)?,
        bounds: template.bounds.clone(),
        device_budget_bytes: template.device_budget_bytes,
    };
    serde_json::to_vec(&envelope).map_err(Into::into)
}

pub fn deserialize(bytes: &[u8], fingerprint: &str) -> Result<SearchedPlanTemplate> {
    let envelope: Envelope = serde_json::from_slice(bytes).context("decoding plan artifact")?;
    ensure!(
        envelope.schema == ARTIFACT_SCHEMA,
        "plan artifact schema {} is not supported (expected {})",
        envelope.schema,
        ARTIFACT_SCHEMA
    );
    ensure!(
        envelope.op_registry_abi == OP_REGISTRY_ABI,
        "plan artifact op-registry ABI {} is not supported (expected {})",
        envelope.op_registry_abi,
        OP_REGISTRY_ABI
    );
    ensure!(
        envelope.crate_version == env!("CARGO_PKG_VERSION"),
        "plan artifact was written by luminal_cuda_lite {}, running {}",
        envelope.crate_version,
        env!("CARGO_PKG_VERSION")
    );
    ensure!(
        envelope.fingerprint == fingerprint,
        "plan artifact fingerprint mismatch: stored {}, requested {}",
        envelope.fingerprint,
        fingerprint
    );
    let plan = envelope.plan.decode()?;
    ensure!(
        kernel_sources(&plan, &envelope.bounds)? == envelope.kernel_sources,
        "plan artifact kernel sources do not match the installed plan under the current op registry"
    );
    Ok(SearchedPlanTemplate {
        plan,
        bounds: envelope.bounds,
        device_budget_bytes: envelope.device_budget_bytes,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct KernelSourcesWire {
    node: usize,
    label: String,
    sources: Vec<String>,
}

fn kernel_sources(
    plan: &BufferIrGraph<DecodedLayout>,
    bounds: &crate::symbolic::Bounds,
) -> Result<Vec<KernelSourcesWire>> {
    let schema: Vec<_> = bounds.keys().copied().collect();
    let mut sources = Vec::new();
    for node in plan.dag.node_indices() {
        let BufferNode::Compute {
            op,
            operand_info,
            result_info,
            ..
        } = &plan.dag[node]
        else {
            continue;
        };
        let Some(kernel) = crate::as_kernel_op(op.as_ref()) else {
            continue;
        };
        let label = op.label();
        let codegen =
            crate::kernels::CodegenCtx::from_descriptors(label, operand_info, result_info)?;
        let generated = kernel
            .codegen(&codegen)?
            .into_iter()
            .map(|generated| {
                let mut source = crate::kernels::dtype_includes(&{
                    let mut dtypes = codegen.operand_dtypes.clone();
                    dtypes.extend_from_slice(&codegen.dest_dtypes);
                    dtypes
                });
                source.push_str(crate::symbolic::CUDA_HELPERS);
                for (index, symbol) in schema.iter().enumerate() {
                    source.push_str(&format!(
                        "#define {} params[{index}]\n",
                        crate::symbolic::variable(&symbol.to_string())
                    ));
                }
                source.push_str(&generated.source);
                source
            })
            .collect();
        sources.push(KernelSourcesWire {
            node: node.index(),
            label: label.to_string(),
            sources: generated,
        });
    }
    Ok(sources)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum LayoutSpelling {
    RightMajor(RightMajorContiguousElementLayout),
    LeftMajor(LeftMajorContiguousElementLayout),
    Strided(StridedElementLayout),
    ElementOffset(ElementOffsetExpressionLayout),
    BitOffset(BitOffsetExpressionLayout),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LayoutWire {
    class: ClassId,
    dtype: Option<luminal::dtype::PlanDtype>,
    spellings: Vec<LayoutSpelling>,
}

impl LayoutWire {
    fn encode(layout: &DecodedLayout) -> Self {
        let mut spellings = Vec::new();
        spellings.extend(
            layout
                .spellings
                .all::<RightMajorContiguousElementLayout>()
                .into_iter()
                .cloned()
                .map(LayoutSpelling::RightMajor),
        );
        spellings.extend(
            layout
                .spellings
                .all::<LeftMajorContiguousElementLayout>()
                .into_iter()
                .cloned()
                .map(LayoutSpelling::LeftMajor),
        );
        spellings.extend(
            layout
                .spellings
                .all::<StridedElementLayout>()
                .into_iter()
                .cloned()
                .map(LayoutSpelling::Strided),
        );
        spellings.extend(
            layout
                .spellings
                .all::<ElementOffsetExpressionLayout>()
                .into_iter()
                .cloned()
                .map(LayoutSpelling::ElementOffset),
        );
        spellings.extend(
            layout
                .spellings
                .all::<BitOffsetExpressionLayout>()
                .into_iter()
                .cloned()
                .map(LayoutSpelling::BitOffset),
        );
        Self {
            class: layout.class.clone(),
            dtype: layout.dtype,
            spellings,
        }
    }

    fn decode(self) -> Result<DecodedLayout> {
        let decoded: Vec<Arc<dyn LayoutFacts>> = self
            .spellings
            .into_iter()
            .map(|spelling| match spelling {
                LayoutSpelling::RightMajor(value) => Arc::new(value) as Arc<dyn LayoutFacts>,
                LayoutSpelling::LeftMajor(value) => Arc::new(value),
                LayoutSpelling::Strided(value) => Arc::new(value),
                LayoutSpelling::ElementOffset(value) => Arc::new(value),
                LayoutSpelling::BitOffset(value) => Arc::new(value),
            })
            .collect();
        ensure!(!decoded.is_empty(), "artifact layout has no spelling");
        Ok(DecodedLayout {
            class: self.class,
            dtype: self.dtype,
            spellings: Spellings::<Layout>::from_decoded(decoded),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BufferWire {
    id: BufferId,
    access: luminal::layout_ir::Access,
    freed_by: luminal::layout_ir::FreedBy,
    owner: luminal::bufferize::Owner,
    label: String,
    lit: Option<i64>,
    backs: ClassId,
    layout: LayoutWire,
}

impl BufferWire {
    fn encode(buffer: &Buffer<DecodedLayout>) -> Self {
        Self {
            id: buffer.id.clone(),
            access: buffer.access,
            freed_by: buffer.freed_by,
            owner: buffer.owner,
            label: buffer.label.clone(),
            lit: buffer.lit,
            backs: buffer.backs.clone(),
            layout: LayoutWire::encode(&buffer.layout),
        }
    }

    fn decode(self) -> Result<Buffer<DecodedLayout>> {
        Ok(Buffer {
            id: self.id,
            access: self.access,
            freed_by: self.freed_by,
            owner: self.owner,
            label: self.label,
            lit: self.lit,
            backs: self.backs,
            layout: self.layout.decode()?,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SlotWire {
    value: ClassId,
    buffer: BufferId,
    layout: LayoutWire,
}

impl SlotWire {
    fn encode(slot: &SlotDescriptor<DecodedLayout>) -> Self {
        Self {
            value: slot.value.clone(),
            buffer: slot.buffer.clone(),
            layout: LayoutWire::encode(&slot.layout),
        }
    }

    fn decode(self) -> Result<SlotDescriptor<DecodedLayout>> {
        Ok(SlotDescriptor {
            value: self.value,
            buffer: self.buffer,
            layout: self.layout.decode()?,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OutputWire {
    index: usize,
    value: ClassId,
    buffer: BufferId,
    layout: LayoutWire,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum NodeWire {
    Input {
        slots: Vec<(ClassId, BufferId)>,
    },
    Compute {
        op: Box<OpWire>,
        reads: Vec<BufferId>,
        writes: Vec<BufferId>,
        ties: Vec<(usize, usize)>,
        operand_info: Vec<SlotWire>,
        result_info: Vec<SlotWire>,
    },
    Copy {
        src: BufferId,
        dst: BufferId,
    },
    Output {
        slots: Vec<OutputWire>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EdgeWire {
    source: usize,
    target: usize,
    buffer: BufferId,
    port: String,
    kind: luminal::bufferize::EdgeKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PlanWire {
    nodes: Vec<NodeWire>,
    edges: Vec<EdgeWire>,
    buffers: Vec<BufferWire>,
    value_buffer: Vec<(ClassId, BufferId)>,
    outputs: Vec<usize>,
}

impl PlanWire {
    fn encode(plan: &BufferIrGraph<DecodedLayout>) -> Result<Self> {
        let nodes = plan
            .dag
            .node_weights()
            .map(|node| match node {
                BufferNode::BufferInput { slots } => Ok(NodeWire::Input {
                    slots: slots
                        .iter()
                        .map(|slot| (slot.value.clone(), slot.buffer.clone()))
                        .collect(),
                }),
                BufferNode::Compute {
                    op,
                    reads,
                    writes,
                    ties,
                    operand_info,
                    result_info,
                } => Ok(NodeWire::Compute {
                    op: Box::new(OpWire::encode(&**op)?),
                    reads: reads.clone(),
                    writes: writes.clone(),
                    ties: ties.clone(),
                    operand_info: operand_info.iter().map(SlotWire::encode).collect(),
                    result_info: result_info.iter().map(SlotWire::encode).collect(),
                }),
                BufferNode::BufferCopy { src, dst } => Ok(NodeWire::Copy {
                    src: src.clone(),
                    dst: dst.clone(),
                }),
                BufferNode::BufferOutput { slots } => Ok(NodeWire::Output {
                    slots: slots
                        .iter()
                        .map(|slot| OutputWire {
                            index: slot.index,
                            value: slot.value.clone(),
                            buffer: slot.buffer.clone(),
                            layout: LayoutWire::encode(&slot.layout),
                        })
                        .collect(),
                }),
            })
            .collect::<Result<_>>()?;
        let edges = plan
            .dag
            .edge_references()
            .map(|edge| EdgeWire {
                source: edge.source().index(),
                target: edge.target().index(),
                buffer: edge.weight().buffer.clone(),
                port: edge.weight().port.clone(),
                kind: edge.weight().kind,
            })
            .collect();
        Ok(Self {
            nodes,
            edges,
            buffers: plan.buffers.values().map(BufferWire::encode).collect(),
            value_buffer: plan
                .value_buffer
                .iter()
                .map(|(value, buffer)| (value.clone(), buffer.clone()))
                .collect(),
            outputs: plan.outputs.iter().map(|node| node.index()).collect(),
        })
    }

    fn decode(self) -> Result<BufferIrGraph<DecodedLayout>> {
        let mut dag = DiGraph::new();
        for node in self.nodes {
            let node = match node {
                NodeWire::Input { slots } => BufferNode::BufferInput {
                    slots: slots
                        .into_iter()
                        .map(|(value, buffer)| InputBinding { value, buffer })
                        .collect(),
                },
                NodeWire::Compute {
                    op,
                    reads,
                    writes,
                    ties,
                    operand_info,
                    result_info,
                } => BufferNode::Compute {
                    op: (*op).decode()?,
                    reads,
                    writes,
                    ties,
                    operand_info: operand_info
                        .into_iter()
                        .map(SlotWire::decode)
                        .collect::<Result<_>>()?,
                    result_info: result_info
                        .into_iter()
                        .map(SlotWire::decode)
                        .collect::<Result<_>>()?,
                },
                NodeWire::Copy { src, dst } => BufferNode::BufferCopy { src, dst },
                NodeWire::Output { slots } => BufferNode::BufferOutput {
                    slots: slots
                        .into_iter()
                        .map(|slot| {
                            Ok(OutputBinding {
                                index: slot.index,
                                value: slot.value,
                                buffer: slot.buffer,
                                layout: slot.layout.decode()?,
                            })
                        })
                        .collect::<Result<_>>()?,
                },
            };
            dag.add_node(node);
        }
        for edge in self.edges {
            ensure!(
                edge.source < dag.node_count() && edge.target < dag.node_count(),
                "artifact edge {} -> {} is outside {} nodes",
                edge.source,
                edge.target,
                dag.node_count()
            );
            dag.add_edge(
                NodeIndex::new(edge.source),
                NodeIndex::new(edge.target),
                BufferEdge {
                    buffer: edge.buffer,
                    port: edge.port,
                    kind: edge.kind,
                },
            );
        }
        let buffers: HashMap<BufferId, Buffer<DecodedLayout>> = self
            .buffers
            .into_iter()
            .map(|wire| {
                let id = wire.id.clone();
                Ok((id, wire.decode()?))
            })
            .collect::<Result<_>>()?;
        let outputs = self
            .outputs
            .into_iter()
            .map(|index| {
                ensure!(
                    index < dag.node_count(),
                    "artifact output node {index} is absent"
                );
                Ok(NodeIndex::new(index))
            })
            .collect::<Result<_>>()?;
        Ok(BufferIrGraph {
            dag,
            buffers,
            value_buffer: self.value_buffer.into_iter().collect::<BTreeMap<_, _>>(),
            outputs,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum OpWire {
    Add,
    Cast,
    Ceil,
    Div,
    Exp,
    Exp2,
    Floor,
    LessThan,
    Log2,
    Copy,
    Mod,
    Mul,
    Recip,
    Round,
    Select,
    Sin,
    Sqrt,
    Trunc,
    TruncCast,
    TruncDiv,
    TruncRem,
    ReduceMax(i64),
    ReduceSum(i64),
    Gather(usize),
    Scatter(usize),
    Constant(f64),
    Iota(Option<luminal::index_expr::IotaExpr>),
    IndexMap(Option<Vec<luminal::index_expr::IotaExpr>>),
    CublasLt(Box<crate::ops::cublaslt::CublasLtDps>),
    BufferAlloc,
    BufferFree,
}

macro_rules! zst_ops {
    ($op:expr, $($variant:ident => $ty:path),+ $(,)?) => {{
        $(if $op.as_any().is::<$ty>() { return Ok(OpWire::$variant); })+
    }};
}

impl OpWire {
    fn encode(op: &dyn BufferTensorIrOp) -> Result<Self> {
        zst_ops!(op,
            Add => crate::ops::add::AddFunctionalDps,
            Cast => crate::ops::cast::CastDps,
            Ceil => crate::ops::ceil::CeilFunctionalDps,
            Div => crate::ops::div::DivFunctionalDps,
            Exp => crate::ops::exp::ExpFunctionalDps,
            Exp2 => crate::ops::exp2::Exp2FunctionalDps,
            Floor => crate::ops::floor::FloorFunctionalDps,
            LessThan => crate::ops::less_than::LessThanDps,
            Log2 => crate::ops::log2::Log2FunctionalDps,
            Copy => crate::ops::materialize_layout_copy::MaterializeLayoutCopyDps,
            Mod => crate::ops::modulo::ModFunctionalDps,
            Mul => crate::ops::mul::MulFunctionalDps,
            Recip => crate::ops::recip::RecipFunctionalDps,
            Round => crate::ops::round::RoundFunctionalDps,
            Select => crate::ops::select::SelectFunctionalDps,
            Sin => crate::ops::sin::SinFunctionalDps,
            Sqrt => crate::ops::sqrt::SqrtFunctionalDps,
            Trunc => crate::ops::trunc::TruncFunctionalDps,
            TruncCast => crate::ops::trunc_cast::TruncCastDps,
            TruncDiv => crate::ops::trunc_div::TruncDivFunctionalDps,
            TruncRem => crate::ops::trunc_rem::TruncRemFunctionalDps,
            BufferAlloc => BufferAlloc,
            BufferFree => BufferFree,
        );
        if let Some(value) = op
            .as_any()
            .downcast_ref::<crate::ops::reduce_max::ReduceMaxDps>()
        {
            return Ok(Self::ReduceMax(value.axis));
        }
        if let Some(value) = op
            .as_any()
            .downcast_ref::<crate::ops::reduce_sum::ReduceSumDps>()
        {
            return Ok(Self::ReduceSum(value.axis));
        }
        if let Some(value) = op.as_any().downcast_ref::<crate::ops::gather::GatherDps>() {
            return Ok(Self::Gather(value.rank));
        }
        if let Some(value) = op
            .as_any()
            .downcast_ref::<crate::ops::scatter::ScatterFunctionalDps>()
        {
            return Ok(Self::Scatter(value.rank));
        }
        if let Some(value) = op
            .as_any()
            .downcast_ref::<crate::ops::constant::ConstantDps>()
        {
            return Ok(Self::Constant(value.value));
        }
        if let Some(value) = op.as_any().downcast_ref::<crate::ops::iota::IotaDps>() {
            return Ok(Self::Iota(value.expr.clone()));
        }
        if let Some(value) = op
            .as_any()
            .downcast_ref::<crate::ops::index_map_apply_materialize::IndexMapApplyMaterializeDps>(
        ) {
            return Ok(Self::IndexMap(value.entries.clone()));
        }
        if let Some(value) = op
            .as_any()
            .downcast_ref::<crate::ops::cublaslt::CublasLtDps>()
        {
            return Ok(Self::CublasLt(Box::new(value.clone())));
        }
        bail!("operation {:?} has no plan-artifact codec", op)
    }

    fn decode(self) -> Result<Box<dyn BufferTensorIrOp>> {
        use crate::ops;
        Ok(match self {
            Self::Add => Box::new(ops::add::AddFunctionalDps),
            Self::Cast => Box::new(ops::cast::CastDps),
            Self::Ceil => Box::new(ops::ceil::CeilFunctionalDps),
            Self::Div => Box::new(ops::div::DivFunctionalDps),
            Self::Exp => Box::new(ops::exp::ExpFunctionalDps),
            Self::Exp2 => Box::new(ops::exp2::Exp2FunctionalDps),
            Self::Floor => Box::new(ops::floor::FloorFunctionalDps),
            Self::LessThan => Box::new(ops::less_than::LessThanDps),
            Self::Log2 => Box::new(ops::log2::Log2FunctionalDps),
            Self::Copy => Box::new(ops::materialize_layout_copy::MaterializeLayoutCopyDps),
            Self::Mod => Box::new(ops::modulo::ModFunctionalDps),
            Self::Mul => Box::new(ops::mul::MulFunctionalDps),
            Self::Recip => Box::new(ops::recip::RecipFunctionalDps),
            Self::Round => Box::new(ops::round::RoundFunctionalDps),
            Self::Select => Box::new(ops::select::SelectFunctionalDps),
            Self::Sin => Box::new(ops::sin::SinFunctionalDps),
            Self::Sqrt => Box::new(ops::sqrt::SqrtFunctionalDps),
            Self::Trunc => Box::new(ops::trunc::TruncFunctionalDps),
            Self::TruncCast => Box::new(ops::trunc_cast::TruncCastDps),
            Self::TruncDiv => Box::new(ops::trunc_div::TruncDivFunctionalDps),
            Self::TruncRem => Box::new(ops::trunc_rem::TruncRemFunctionalDps),
            Self::ReduceMax(axis) => Box::new(ops::reduce_max::ReduceMaxDps { axis }),
            Self::ReduceSum(axis) => Box::new(ops::reduce_sum::ReduceSumDps { axis }),
            Self::Gather(rank) => Box::new(ops::gather::GatherDps { rank }),
            Self::Scatter(rank) => Box::new(ops::scatter::ScatterFunctionalDps { rank }),
            Self::Constant(value) => Box::new(ops::constant::ConstantDps { value }),
            Self::Iota(expr) => Box::new(ops::iota::IotaDps { expr }),
            Self::IndexMap(entries) => {
                Box::new(ops::index_map_apply_materialize::IndexMapApplyMaterializeDps { entries })
            }
            Self::CublasLt(value) => value,
            Self::BufferAlloc => Box::new(BufferAlloc),
            Self::BufferFree => Box::new(BufferFree),
        })
    }
}
