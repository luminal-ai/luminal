//! Versioned, reference-only representation of an elected buffer plan.
//! Loading this representation never runs egglog, extraction, or profiling.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use luminal::buffer_tensor_ir::{BufferAlloc, BufferFree, BufferTensorIrOp};
use luminal::bufferize::{
    Buffer, BufferEdge, BufferId, BufferIrGraph, BufferNode, EdgeKind, InputBinding, OutputBinding,
    Owner, SlotDescriptor,
};
use luminal::dtype::PlanDtype;
use luminal::index_expr::IotaExpr;
use luminal::layout_ir::{Access, FreedBy};
use luminal::layouts::{
    BitOffsetExpressionLayout, DecodedLayout, ElementOffsetExpressionLayout, Layout, LayoutFacts,
    LeftMajorContiguousElementLayout, RightMajorContiguousElementLayout, StridedElementLayout,
};
use luminal::prelude::{NodeIndex, egraph_serialize::ClassId};
use petgraph::graph::DiGraph;
use serde::{Deserialize, Serialize};

use crate::ops;

const SCHEMA_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
struct Artifact {
    version: u32,
    plans: Vec<BucketWire>,
}

#[derive(Serialize, Deserialize)]
struct BucketWire {
    ranges: BTreeMap<String, (usize, usize)>,
    input_buffers: Vec<i64>,
    output_buffers: Vec<i64>,
    plan: PlanWire,
}

/// One selected plan and its boundary slots. `ranges` is empty for a static
/// graph. The slot lists are declaration-order buffer literals, independent
/// of the compiling process's graph node indices.
pub struct CompiledBucket {
    pub ranges: BTreeMap<String, (usize, usize)>,
    pub input_buffers: Vec<i64>,
    pub output_buffers: Vec<i64>,
    pub plan: BufferIrGraph<DecodedLayout>,
}

pub fn serialize(buckets: &[CompiledBucket]) -> Result<Vec<u8>> {
    ensure!(
        !buckets.is_empty(),
        "no compiled reference plans to serialize"
    );
    let plans = buckets
        .iter()
        .map(|bucket| {
            Ok(BucketWire {
                ranges: bucket.ranges.clone(),
                input_buffers: bucket.input_buffers.clone(),
                output_buffers: bucket.output_buffers.clone(),
                plan: PlanWire::from_plan(&bucket.plan)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    serde_json::to_vec(&Artifact {
        version: SCHEMA_VERSION,
        plans,
    })
    .context("serializing reference plan")
}

pub fn deserialize(data: &[u8]) -> Result<Vec<CompiledBucket>> {
    let artifact: Artifact =
        serde_json::from_slice(data).context("parsing reference plan artifact")?;
    ensure!(
        artifact.version == SCHEMA_VERSION,
        "unsupported reference plan schema {}",
        artifact.version
    );
    ensure!(
        !artifact.plans.is_empty(),
        "reference plan artifact contains no plans"
    );
    artifact
        .plans
        .into_iter()
        .map(|bucket| {
            for (name, &(lo, hi)) in &bucket.ranges {
                luminal::shape::Symbol::try_new_dim(name)?;
                ensure!(
                    lo <= hi,
                    "invalid reference plan bucket range {name}={lo}..{hi}"
                );
            }
            Ok(CompiledBucket {
                ranges: bucket.ranges,
                input_buffers: bucket.input_buffers,
                output_buffers: bucket.output_buffers,
                plan: bucket.plan.into_plan()?,
            })
        })
        .collect()
}

#[derive(Clone, Serialize, Deserialize)]
enum SpellingWire {
    Right(RightMajorContiguousElementLayout),
    Left(LeftMajorContiguousElementLayout),
    Strided(StridedElementLayout),
    ElementOffset(ElementOffsetExpressionLayout),
    BitOffset(BitOffsetExpressionLayout),
}

#[derive(Clone, Serialize, Deserialize)]
struct LayoutWire {
    class: ClassId,
    dtype: Option<PlanDtype>,
    spellings: Vec<SpellingWire>,
}

impl LayoutWire {
    fn from_layout(layout: &DecodedLayout) -> Result<Self> {
        let mut spellings = Vec::new();
        for fact in layout.spellings.iter() {
            let any = fact as &dyn std::any::Any;
            let spelling = if let Some(v) = any.downcast_ref::<RightMajorContiguousElementLayout>()
            {
                SpellingWire::Right(v.clone())
            } else if let Some(v) = any.downcast_ref::<LeftMajorContiguousElementLayout>() {
                SpellingWire::Left(v.clone())
            } else if let Some(v) = any.downcast_ref::<StridedElementLayout>() {
                SpellingWire::Strided(v.clone())
            } else if let Some(v) = any.downcast_ref::<ElementOffsetExpressionLayout>() {
                SpellingWire::ElementOffset(v.clone())
            } else if let Some(v) = any.downcast_ref::<BitOffsetExpressionLayout>() {
                SpellingWire::BitOffset(v.clone())
            } else {
                bail!("unknown reference layout spelling in {}", layout.class);
            };
            spellings.push(spelling);
        }
        ensure!(
            !spellings.is_empty(),
            "layout {} has no spellings",
            layout.class
        );
        Ok(Self {
            class: layout.class.clone(),
            dtype: layout.dtype,
            spellings,
        })
    }

    fn into_layout(self) -> Result<DecodedLayout> {
        let decoded: Vec<Arc<dyn LayoutFacts>> = self
            .spellings
            .into_iter()
            .map(|spelling| match spelling {
                SpellingWire::Right(v) => Arc::new(v) as Arc<dyn LayoutFacts>,
                SpellingWire::Left(v) => Arc::new(v) as Arc<dyn LayoutFacts>,
                SpellingWire::Strided(v) => Arc::new(v) as Arc<dyn LayoutFacts>,
                SpellingWire::ElementOffset(v) => Arc::new(v) as Arc<dyn LayoutFacts>,
                SpellingWire::BitOffset(v) => Arc::new(v) as Arc<dyn LayoutFacts>,
            })
            .collect();
        ensure!(
            !decoded.is_empty(),
            "layout {} has no spellings",
            self.class
        );
        Ok(DecodedLayout {
            class: self.class,
            dtype: self.dtype,
            spellings: luminal::egglog_utils::eclass::Spellings::<Layout>::from_decoded(decoded),
        })
    }
}

#[derive(Serialize, Deserialize)]
struct BufferWire {
    id: BufferId,
    access: Access,
    freed_by: FreedBy,
    owner: Owner,
    label: String,
    lit: Option<i64>,
    backs: ClassId,
    layout: LayoutWire,
}

#[derive(Serialize, Deserialize)]
struct SlotWire {
    value: ClassId,
    buffer: BufferId,
    layout: LayoutWire,
}

#[derive(Serialize, Deserialize)]
struct OutputWire {
    index: usize,
    value: ClassId,
    buffer: BufferId,
    layout: LayoutWire,
}

#[derive(Serialize, Deserialize)]
struct OpWire {
    kind: String,
    payload: serde_json::Value,
}

#[derive(Serialize, Deserialize)]
enum NodeWire {
    Input(Vec<(ClassId, BufferId)>),
    Compute {
        op: OpWire,
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
    Output(Vec<OutputWire>),
}

#[derive(Serialize, Deserialize)]
struct EdgeWire {
    from: usize,
    to: usize,
    buffer: BufferId,
    port: String,
    kind: EdgeKind,
}

#[derive(Serialize, Deserialize)]
struct PlanWire {
    nodes: Vec<NodeWire>,
    edges: Vec<EdgeWire>,
    buffers: Vec<BufferWire>,
    value_buffer: Vec<(ClassId, BufferId)>,
    outputs: Vec<usize>,
}

impl PlanWire {
    fn from_plan(plan: &BufferIrGraph<DecodedLayout>) -> Result<Self> {
        use petgraph::visit::EdgeRef;
        let ids: HashMap<_, _> = plan
            .dag
            .node_indices()
            .enumerate()
            .map(|(i, id)| (id, i))
            .collect();
        let slot = |s: &SlotDescriptor<DecodedLayout>| -> Result<SlotWire> {
            Ok(SlotWire {
                value: s.value.clone(),
                buffer: s.buffer.clone(),
                layout: LayoutWire::from_layout(&s.layout)?,
            })
        };
        let nodes = plan
            .dag
            .node_indices()
            .map(|id| -> Result<NodeWire> {
                Ok(match &plan.dag[id] {
                    BufferNode::BufferInput { slots } => NodeWire::Input(
                        slots
                            .iter()
                            .map(|s| (s.value.clone(), s.buffer.clone()))
                            .collect(),
                    ),
                    BufferNode::Compute {
                        op,
                        reads,
                        writes,
                        ties,
                        operand_info,
                        result_info,
                    } => NodeWire::Compute {
                        op: OpWire::from_op(op.as_ref())?,
                        reads: reads.clone(),
                        writes: writes.clone(),
                        ties: ties.clone(),
                        operand_info: operand_info.iter().map(&slot).collect::<Result<_>>()?,
                        result_info: result_info.iter().map(&slot).collect::<Result<_>>()?,
                    },
                    BufferNode::BufferCopy { src, dst } => NodeWire::Copy {
                        src: src.clone(),
                        dst: dst.clone(),
                    },
                    BufferNode::BufferOutput { slots } => NodeWire::Output(
                        slots
                            .iter()
                            .map(|s| -> Result<_> {
                                Ok(OutputWire {
                                    index: s.index,
                                    value: s.value.clone(),
                                    buffer: s.buffer.clone(),
                                    layout: LayoutWire::from_layout(&s.layout)?,
                                })
                            })
                            .collect::<Result<_>>()?,
                    ),
                })
            })
            .collect::<Result<_>>()?;
        let edges = plan
            .dag
            .edge_references()
            .map(|e| EdgeWire {
                from: ids[&e.source()],
                to: ids[&e.target()],
                buffer: e.weight().buffer.clone(),
                port: e.weight().port.clone(),
                kind: e.weight().kind,
            })
            .collect();
        let buffers = plan
            .buffers
            .values()
            .map(|b| -> Result<_> {
                Ok(BufferWire {
                    id: b.id.clone(),
                    access: b.access,
                    freed_by: b.freed_by,
                    owner: b.owner,
                    label: b.label.clone(),
                    lit: b.lit,
                    backs: b.backs.clone(),
                    layout: LayoutWire::from_layout(&b.layout)?,
                })
            })
            .collect::<Result<_>>()?;
        let value_buffer = plan
            .value_buffer
            .iter()
            .map(|(v, b)| (v.clone(), b.clone()))
            .collect();
        let outputs = plan.outputs.iter().map(|id| ids[id]).collect();
        Ok(Self {
            nodes,
            edges,
            buffers,
            value_buffer,
            outputs,
        })
    }

    fn into_plan(self) -> Result<BufferIrGraph<DecodedLayout>> {
        let mut dag = DiGraph::new();
        for node in self.nodes {
            let slot = |s: SlotWire| -> Result<SlotDescriptor<DecodedLayout>> {
                Ok(SlotDescriptor {
                    value: s.value,
                    buffer: s.buffer,
                    layout: s.layout.into_layout()?,
                })
            };
            let node = match node {
                NodeWire::Input(slots) => BufferNode::BufferInput {
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
                    op: op.into_op()?,
                    reads,
                    writes,
                    ties,
                    operand_info: operand_info.into_iter().map(&slot).collect::<Result<_>>()?,
                    result_info: result_info.into_iter().map(&slot).collect::<Result<_>>()?,
                },
                NodeWire::Copy { src, dst } => BufferNode::BufferCopy { src, dst },
                NodeWire::Output(slots) => BufferNode::BufferOutput {
                    slots: slots
                        .into_iter()
                        .map(|s| -> Result<_> {
                            Ok(OutputBinding {
                                index: s.index,
                                value: s.value,
                                buffer: s.buffer,
                                layout: s.layout.into_layout()?,
                            })
                        })
                        .collect::<Result<_>>()?,
                },
            };
            dag.add_node(node);
        }
        for edge in self.edges {
            ensure!(
                edge.from < dag.node_count() && edge.to < dag.node_count(),
                "artifact edge refers to a missing node"
            );
            dag.add_edge(
                NodeIndex::new(edge.from),
                NodeIndex::new(edge.to),
                BufferEdge {
                    buffer: edge.buffer,
                    port: edge.port,
                    kind: edge.kind,
                },
            );
        }
        let buffers = self
            .buffers
            .into_iter()
            .map(|b| -> Result<_> {
                let id = b.id;
                Ok((
                    id.clone(),
                    Buffer {
                        id,
                        access: b.access,
                        freed_by: b.freed_by,
                        owner: b.owner,
                        label: b.label,
                        lit: b.lit,
                        backs: b.backs,
                        layout: b.layout.into_layout()?,
                    },
                ))
            })
            .collect::<Result<HashMap<_, _>>>()?;
        let value_buffer = self.value_buffer.into_iter().collect();
        let outputs = self
            .outputs
            .into_iter()
            .map(|id| {
                ensure!(
                    id < dag.node_count(),
                    "artifact output refers to a missing node"
                );
                Ok(NodeIndex::new(id))
            })
            .collect::<Result<_>>()?;
        Ok(BufferIrGraph {
            dag,
            buffers,
            value_buffer,
            outputs,
        })
    }
}

impl OpWire {
    fn from_op(op: &dyn BufferTensorIrOp) -> Result<Self> {
        let any = op.as_any();
        macro_rules! unit { ($($ty:ty),* $(,)?) => { $(if any.is::<$ty>() {
            return Ok(Self { kind: stringify!($ty).into(), payload: serde_json::Value::Null });
        })* }; }
        unit!(
            BufferAlloc,
            BufferFree,
            luminal::buffer_tensor_ir::Poison,
            ops::AddFunctionalDps,
            ops::MulFunctionalDps,
            ops::DivFunctionalDps,
            ops::TruncDivFunctionalDps,
            ops::TruncRemFunctionalDps,
            ops::ModFunctionalDps,
            ops::CastDps,
            ops::TruncCastDps,
            ops::CeilFunctionalDps,
            ops::FloorFunctionalDps,
            ops::RoundFunctionalDps,
            ops::TruncFunctionalDps,
            ops::ExpFunctionalDps,
            ops::Exp2FunctionalDps,
            ops::Log2FunctionalDps,
            ops::RecipFunctionalDps,
            ops::SinFunctionalDps,
            ops::SqrtFunctionalDps,
            ops::LessThanDps,
            ops::SelectFunctionalDps
        );
        let (kind, payload) = if let Some(v) = any.downcast_ref::<ops::ConstantDps>() {
            ("ops::ConstantDps", serde_json::json!(v.value.to_bits()))
        } else if let Some(v) = any.downcast_ref::<ops::GatherDps>() {
            ("ops::GatherDps", serde_json::json!(v.rank))
        } else if let Some(v) = any.downcast_ref::<ops::ScatterFunctionalDps>() {
            ("ops::ScatterFunctionalDps", serde_json::json!(v.rank))
        } else if let Some(v) = any.downcast_ref::<ops::ReduceSumDps>() {
            ("ops::ReduceSumDps", serde_json::json!(v.axis))
        } else if let Some(v) = any.downcast_ref::<ops::ReduceMaxDps>() {
            ("ops::ReduceMaxDps", serde_json::json!(v.axis))
        } else if let Some(v) = any.downcast_ref::<ops::IotaDps>() {
            ("ops::IotaDps", serde_json::to_value(&v.expr)?)
        } else if let Some(v) = any.downcast_ref::<ops::IndexMapApplyMaterializeDps>() {
            (
                "ops::IndexMapApplyMaterializeDps",
                serde_json::to_value(&v.entries)?,
            )
        } else {
            bail!("reference artifact cannot encode operation {}", op.label());
        };
        Ok(Self {
            kind: kind.into(),
            payload,
        })
    }

    fn into_op(self) -> Result<Box<dyn BufferTensorIrOp>> {
        macro_rules! unit { ($($ty:path),* $(,)?) => { $(if self.kind == stringify!($ty) {
            ensure!(self.payload.is_null(), "unit operation has a payload");
            return Ok(Box::new($ty));
        })* }; }
        unit!(
            BufferAlloc,
            BufferFree,
            luminal::buffer_tensor_ir::Poison,
            ops::AddFunctionalDps,
            ops::MulFunctionalDps,
            ops::DivFunctionalDps,
            ops::TruncDivFunctionalDps,
            ops::TruncRemFunctionalDps,
            ops::ModFunctionalDps,
            ops::CastDps,
            ops::TruncCastDps,
            ops::CeilFunctionalDps,
            ops::FloorFunctionalDps,
            ops::RoundFunctionalDps,
            ops::TruncFunctionalDps,
            ops::ExpFunctionalDps,
            ops::Exp2FunctionalDps,
            ops::Log2FunctionalDps,
            ops::RecipFunctionalDps,
            ops::SinFunctionalDps,
            ops::SqrtFunctionalDps,
            ops::LessThanDps,
            ops::SelectFunctionalDps
        );
        let kind = self.kind.as_str();
        let payload = self.payload;
        Ok(match kind {
            "ops::ConstantDps" => Box::new(ops::ConstantDps {
                value: f64::from_bits(serde_json::from_value(payload)?),
            }),
            "ops::GatherDps" => Box::new(ops::GatherDps {
                rank: serde_json::from_value(payload)?,
            }),
            "ops::ScatterFunctionalDps" => Box::new(ops::ScatterFunctionalDps {
                rank: serde_json::from_value(payload)?,
            }),
            "ops::ReduceSumDps" => Box::new(ops::ReduceSumDps {
                axis: serde_json::from_value(payload)?,
            }),
            "ops::ReduceMaxDps" => Box::new(ops::ReduceMaxDps {
                axis: serde_json::from_value(payload)?,
            }),
            "ops::IotaDps" => Box::new(ops::IotaDps {
                expr: serde_json::from_value::<Option<IotaExpr>>(payload)?,
            }),
            "ops::IndexMapApplyMaterializeDps" => Box::new(ops::IndexMapApplyMaterializeDps {
                entries: serde_json::from_value(payload)?,
            }),
            _ => bail!("unknown reference operation kind {kind}"),
        })
    }
}

#[cfg(test)]
mod tests {
    use luminal::graph::Graph;
    use luminal::prelude::DType;

    use crate::{ReferenceRuntime, TypedBuffer};

    #[test]
    fn selected_plan_round_trips_without_search() {
        let mut graph = Graph::new();
        let a = graph.tensor((2, 2), DType::F32);
        let b = graph.tensor((2, 2), DType::F32);
        let result = a + b;
        let mut leader = ReferenceRuntime::load(&graph).unwrap();
        let data = [
            (a.id, TypedBuffer::F32(vec![1.; 4])),
            (b.id, TypedBuffer::F32(vec![2.; 4])),
        ]
        .into_iter()
        .collect();
        leader
            .search(&data, &crate::harness_search_options())
            .unwrap();
        let output_buffer = crate::ReferenceBindings::leaves(&graph.logical).outputs()[0].buffer;
        let bytes = leader
            .serialize_compiled(&[a.id, b.id], &[output_buffer])
            .unwrap();

        let mut follower = ReferenceRuntime::default();
        let output_buffers = follower
            .deserialize_compiled(
                &bytes,
                &[a.id, b.id],
                &[result.id],
                crate::runtime::DEFAULT_MEMORY_BUDGET_BYTES,
            )
            .unwrap();
        assert_eq!(output_buffers.len(), 1);
        follower.set_data(a.id, vec![3.; 4]);
        follower.set_data(b.id, vec![4.; 4]);
        follower.execute().unwrap();
        assert_eq!(follower.get_f32(result.id).unwrap(), &vec![7.; 4]);
    }

    #[test]
    fn symbolic_bucket_plans_round_trip_and_select_by_runtime_dims() {
        use luminal::graph::DimBucket;
        use luminal::shape::Symbol;

        let mut graph = Graph::new();
        graph.set_dim('s', 3);
        let x = graph.tensor(('s', 2), DType::F32);
        let y = graph.tensor(('s', 2), DType::F32);
        let out = x + y;
        let output_buffer = crate::ReferenceBindings::leaves(&graph.logical).outputs()[0].buffer;
        let mut leader = ReferenceRuntime::load(&graph).unwrap();
        leader
            .bind_dim_buckets('s', vec![DimBucket::new(2, 4), DimBucket::new(5, 8)])
            .unwrap();
        leader
            .search_buckets(
                |dims| {
                    let n = dims[&Symbol::from('s')] * 2;
                    [
                        (x.id, TypedBuffer::F32(vec![1.; n])),
                        (y.id, TypedBuffer::F32(vec![2.; n])),
                    ]
                    .into_iter()
                    .collect()
                },
                &crate::harness_search_options(),
            )
            .unwrap();
        let bytes = leader
            .serialize_compiled(&[x.id, y.id], &[output_buffer])
            .unwrap();
        let mut follower = ReferenceRuntime::default();
        follower
            .deserialize_compiled(
                &bytes,
                &[x.id, y.id],
                &[out.id],
                crate::runtime::DEFAULT_MEMORY_BUDGET_BYTES,
            )
            .unwrap();
        for size in [4, 7] {
            follower.set_dim('s', size);
            follower.set_data(x.id, vec![3.; size * 2]);
            follower.set_data(y.id, vec![4.; size * 2]);
            follower.execute().unwrap();
            assert_eq!(follower.get_f32(out.id).unwrap(), &vec![7.; size * 2]);
        }
    }
}
