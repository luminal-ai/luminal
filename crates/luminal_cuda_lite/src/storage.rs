//! CUDA sizing adapter for the shared physical-lifetime planner. Finalist
//! budgets and installed graphs use exactly the same schedule and offsets.
use crate::{
    arena::ArenaPlan,
    layouts::CudaPlan,
    symbolic::{Bounds, capacity_bytes},
};
use anyhow::{Result, anyhow};

pub(crate) fn plan(plan: &CudaPlan, bounds: &Bounds) -> Result<ArenaPlan> {
    plan_resident(plan, bounds, &Default::default())
}

pub(crate) fn plan_resident(
    plan: &CudaPlan,
    bounds: &Bounds,
    bindings: &crate::resident::ResidentBindings,
) -> Result<ArenaPlan> {
    crate::arena::plan_resident_over(
        plan,
        |buffer| capacity_bytes(&buffer.layout, bounds),
        |node| match &plan.dag[node] {
            luminal::bufferize::BufferNode::Compute { op, .. } => {
                crate::as_host_op(op.as_ref()).map_or(Ok(0), |host| host.workspace_bytes(bounds))
            }
            _ => Ok(0),
        },
        bounds
            .len()
            .checked_mul(8)
            .ok_or_else(|| anyhow!("parameter size overflow"))?
            .max(8),
        crate::arena::issue_order(plan)?,
        &bindings.inputs,
        &bindings.feedback.keys().copied().collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::{ARENA_ALIGN, ArenaStep};
    use luminal::{
        buffer_tensor_ir::{BufferAlloc, BufferFree, BufferTensorIrOp, OpSlotNames},
        bufferize::{
            Buffer, BufferEdge, BufferId, BufferNode, EdgeKind, OutputBinding, Owner,
            SlotDescriptor,
        },
        dtype::PlanDtype,
        layout_ir::{Access, FreedBy},
        layouts::{
            BitWidthTerm, DecodedLayout, IntExprTerm, RightMajorContiguousElementLayout, ShapeTerm,
        },
    };

    #[derive(Debug, Clone)]
    struct ScratchCopy;
    impl OpSlotNames for ScratchCopy {}
    impl BufferTensorIrOp for ScratchCopy {
        fn label(&self) -> &str {
            "ScratchCopy"
        }
        fn operand_reads_memory(&self, i: usize) -> bool {
            i == 0
        }
        fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
            Some(crate::CudaOpInterface::host::<Self>())
        }
    }
    impl crate::HostOp for ScratchCopy {
        fn workspace_bytes(&self, bounds: &Bounds) -> Result<usize> {
            Ok(bounds[&'n'.into()].1 * 16)
        }
        #[cfg(feature = "device")]
        unsafe fn prepare(
            &self,
            ctx: &crate::host::HostOpContext<'_>,
        ) -> Result<Box<dyn crate::host::PreparedHostOp>> {
            struct Prepared {
                src: u64,
                dst: u64,
                scratch: u64,
                bytes: usize,
            }
            impl crate::host::PreparedHostOp for Prepared {
                unsafe fn record(&self, capture: &crate::host::CaptureCtx<'_>) -> Result<()> {
                    if self.bytes > 0 {
                        unsafe {
                            cudarc::driver::result::memcpy_dtod_async(
                                self.scratch,
                                self.src,
                                self.bytes,
                                capture.stream().cu_stream(),
                            )?;
                            cudarc::driver::result::memcpy_dtod_async(
                                self.dst,
                                self.scratch,
                                self.bytes,
                                capture.stream().cu_stream(),
                            )?;
                        }
                    }
                    Ok(())
                }
            }
            Ok(Box::new(Prepared {
                src: ctx.inputs[0].ptr,
                dst: ctx.dest.ptr,
                scratch: ctx.workspace.ptr,
                bytes: ctx.dest.bytes,
            }))
        }
    }

    /// Two independent results with an early output boundary, a donated input,
    /// a late input, and a HostOp whose scratch can overwrite the early result.
    fn copy_plan() -> (CudaPlan, Vec<BufferId>) {
        let mut plan = CudaPlan {
            dag: Default::default(),
            buffers: Default::default(),
            value_buffer: Default::default(),
            outputs: vec![],
        };
        let mut ids = vec![];
        for (i, (name, owned, donated, wide)) in [
            ("a", false, true, true),
            ("early", true, false, true),
            ("b", false, false, false),
            ("temp", true, true, false),
            ("out", true, false, false),
            ("c", false, false, true),
        ]
        .into_iter()
        .enumerate()
        {
            let id = if owned {
                BufferId::Allocated(i as u32)
            } else {
                BufferId::Boundary(name.into())
            };
            let layout = DecodedLayout::of(
                RightMajorContiguousElementLayout {
                    shape: ShapeTerm(vec![IntExprTerm::Mul(
                        Box::new(IntExprTerm::Var("n".into())),
                        Box::new(IntExprTerm::Lit(if wide { 4 } else { 1 })),
                    )]),
                    width: BitWidthTerm(32),
                },
                Some(PlanDtype::F32),
            );
            plan.buffers.insert(
                id.clone(),
                Buffer {
                    id: id.clone(),
                    access: Access::ReadWrite,
                    freed_by: if donated {
                        FreedBy::Program
                    } else {
                        FreedBy::Caller
                    },
                    owner: if owned { Owner::System } else { Owner::Caller },
                    label: name.into(),
                    lit: (!owned).then_some(i as i64),
                    backs: name.into(),
                    layout,
                },
            );
            ids.push(id);
        }
        let desc = |i: usize| SlotDescriptor {
            value: plan.buffers[&ids[i]].backs.clone(),
            buffer: ids[i].clone(),
            layout: plan.buffers[&ids[i]].layout.clone(),
        };
        let compute = |op: Box<dyn BufferTensorIrOp>, reads: Vec<usize>, writes: Vec<usize>| {
            BufferNode::Compute {
                op,
                operand_info: reads.iter().map(|&i| desc(i)).collect(),
                result_info: writes.iter().map(|&i| desc(i)).collect(),
                reads: reads.into_iter().map(|i| ids[i].clone()).collect(),
                writes: writes.into_iter().map(|i| ids[i].clone()).collect(),
                ties: vec![],
            }
        };
        let output = |slots: &[(usize, usize)]| BufferNode::BufferOutput {
            slots: slots
                .iter()
                .map(|&(index, i)| OutputBinding {
                    index,
                    value: plan.buffers[&ids[i]].backs.clone(),
                    buffer: ids[i].clone(),
                    layout: plan.buffers[&ids[i]].layout.clone(),
                })
                .collect(),
        };
        let nodes = vec![
            compute(Box::new(BufferAlloc), vec![], vec![1]),
            BufferNode::BufferCopy {
                src: ids[0].clone(),
                dst: ids[1].clone(),
            },
            compute(Box::new(BufferFree), vec![0], vec![]),
            output(&[(0, 1), (1, 1)]),
            compute(Box::new(BufferAlloc), vec![], vec![3]),
            BufferNode::BufferCopy {
                src: ids[2].clone(),
                dst: ids[3].clone(),
            },
            compute(Box::new(BufferAlloc), vec![], vec![4]),
            compute(Box::new(ScratchCopy), vec![3, 4], vec![4]),
            compute(Box::new(BufferFree), vec![3], vec![]),
            output(&[(2, 4)]),
            output(&[(3, 5)]),
        ];
        let mut previous = None;
        for node in nodes {
            let free =
                matches!(&node, BufferNode::Compute { op, .. } if op.label() == "BufferFree");
            let output = matches!(&node, BufferNode::BufferOutput { .. });
            let at = plan.dag.add_node(node);
            if let Some(prev) = previous {
                plan.dag.add_edge(
                    prev,
                    at,
                    BufferEdge {
                        buffer: ids[0].clone(),
                        port: "order".into(),
                        kind: EdgeKind::Anti,
                    },
                );
            }
            if !free {
                previous = Some(at);
            }
            if output {
                plan.outputs.push(at);
            }
        }
        (plan, ids)
    }
    fn bounds(hi: usize) -> Bounds {
        [('n'.into(), (0, hi))].into_iter().collect()
    }

    #[test]
    fn all_storage_classes_share_physical_ranges_and_transfer_lifetimes() {
        let (graph, ids) = copy_plan();
        let p = plan(&graph, &bounds(128)).unwrap();
        let overlaps = |a: crate::arena::ArenaSlice, b: crate::arena::ArenaSlice| {
            a.offset < b.offset + b.reserved() && b.offset < a.offset + a.reserved()
        };
        let scratch = *p.workspaces.values().next().unwrap();
        assert!(
            overlaps(scratch, p.slices[&ids[0]]) || overlaps(scratch, p.slices[&ids[1]]),
            "scratch recycles donation/early output"
        );
        assert!(
            overlaps(p.slices[&ids[0]], p.slices[&ids[5]])
                || overlaps(p.slices[&ids[1]], p.slices[&ids[5]]),
            "late input reuses an earlier device copy"
        );
        assert!(!overlaps(scratch, p.slices[&ids[3]]));
        assert!(!overlaps(scratch, p.slices[&ids[4]]));
        let uploads: Vec<_> = p
            .steps
            .iter()
            .enumerate()
            .filter_map(|(t, s)| match s {
                ArenaStep::Upload { buffer, staging } => Some((t, buffer, staging)),
                _ => None,
            })
            .collect();
        let downloads: Vec<_> = p
            .steps
            .iter()
            .enumerate()
            .filter_map(|(t, s)| match s {
                ArenaStep::Download {
                    buffer,
                    slots,
                    staging,
                    ..
                } => Some((t, buffer, slots, staging)),
                _ => None,
            })
            .collect();
        assert_eq!(
            downloads.len(),
            3,
            "two aliased output slots use one transfer"
        );
        assert_eq!(downloads[0].2.len(), 2);
        assert!(
            downloads[0].0 < uploads.last().unwrap().0,
            "early readback precedes late upload"
        );
        // Host inputs are all populated before launch: early downloads must
        // not overwrite an input which has yet to be uploaded.
        for (dt, _, _, dst) in &downloads {
            for (ut, _, src) in &uploads {
                if ut >= dt {
                    assert!(
                        dst.offset + dst.bytes <= src.offset
                            || src.offset + src.bytes <= dst.offset
                    );
                }
            }
        }
        let naive: usize = p.slices.values().map(|s| s.reserved()).sum::<usize>()
            + scratch.reserved()
            + ARENA_ALIGN;
        assert!(
            p.slab_bytes * 2 < naive,
            "physical packing should more than halve this fixture: {} vs {naive}",
            p.slab_bytes
        );
        assert_eq!(p.peak_live_bytes, p.slab_bytes);
    }

    #[cfg(feature = "device")]
    #[test]
    fn replay_preserves_early_outputs_and_late_inputs_across_shapes_and_buckets() {
        use crate::{device::CudaDevice, host_buffer::HostBuffer};
        use luminal::prelude::FxHashMap;
        let (graph, _) = copy_plan();
        let capacities = [32, 128];
        let expected_bytes = capacities
            .iter()
            .map(|&hi| plan(&graph, &bounds(hi)).unwrap().slab_bytes)
            .max()
            .unwrap();
        let mut device = CudaDevice::new(0).unwrap();
        device
            .install(
                capacities
                    .iter()
                    .map(|&hi| (graph.clone(), bounds(hi)))
                    .collect(),
            )
            .unwrap();
        let mut retained = None;
        for (bucket, n) in [(0, 7), (1, 128), (0, 0), (1, 13), (0, 32), (0, 7)] {
            let a: Vec<_> = (0..n * 4).map(|i| i as f32 + 1.).collect();
            let b: Vec<_> = (0..n).map(|i| i as f32 - 1000.).collect();
            let c = vec![99f32; n * 4];
            let data: FxHashMap<i64, HostBuffer> = [
                (0, a.clone().into()),
                (2, b.clone().into()),
                (5, c.clone().into()),
            ]
            .into_iter()
            .collect();
            let staged = data.iter().map(|(&k, v)| (k, v)).collect();
            let outputs = device
                .execute(bucket, &staged, &[('n'.into(), n)].into_iter().collect())
                .unwrap();
            assert_eq!(outputs[&0].0.as_f32().unwrap(), a);
            assert_eq!(outputs[&1].0.as_f32().unwrap(), a);
            assert_eq!(outputs[&2].0.as_f32().unwrap(), b);
            assert_eq!(outputs[&3].0.as_f32().unwrap(), c);
            if retained.is_none() {
                retained = Some(outputs);
            }
            assert_eq!(device.stats().arena_bytes, expected_bytes);
            assert_eq!(device.stats().arena_generation, 1);
        }
        assert_eq!(
            retained.unwrap()[&0].0.as_f32().unwrap(),
            (1..=28).map(|i| i as f32).collect::<Vec<_>>()
        );
    }
}
