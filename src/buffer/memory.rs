//! Optional application composition for programs executed serially. Sharing is
//! declared by resource identity and byte capacity; tensor contracts belong to
//! each program and the application that binds it.
use std::collections::BTreeMap;

use crate::arena::{ARENA_ALIGN, ArenaSlice, SlabAlgorithm, SlabBuffer, plan_slab};
use anyhow::{Result, ensure};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceId(pub u64);

#[derive(Clone, Debug)]
pub struct PersistentBinding {
    pub resource: ResourceId,
    pub buffer: i64,
    pub bytes: usize,
}

pub struct ProgramMemory {
    pub scratch_bytes: usize,
    pub bindings: Vec<PersistentBinding>,
}

/// A convenience over the DAG planner: all programs reuse one scratch region,
/// while application-declared resources remain live across every invocation.
/// It owns no allocation and checks no shape, dtype, or access permissions.
#[derive(Debug)]
pub struct SharedArenaPlan {
    pub scratch_bytes: usize,
    pub bytes: usize,
    pub homes: BTreeMap<ResourceId, ArenaSlice>,
    bindings: Vec<BTreeMap<i64, ArenaSlice>>,
}

impl SharedArenaPlan {
    pub fn build(programs: &[ProgramMemory], capacity: usize) -> Result<Self> {
        ensure!(!programs.is_empty(), "no programs to allocate");
        let scratch_bytes = programs
            .iter()
            .map(|p| p.scratch_bytes)
            .max()
            .unwrap()
            .max(1);
        let mut resources = BTreeMap::<ResourceId, usize>::new();
        for program in programs {
            let mut local = BTreeMap::new();
            for binding in &program.bindings {
                ensure!(
                    local.insert(binding.buffer, binding.resource).is_none(),
                    "duplicate buffer binding"
                );
                let bytes = resources.entry(binding.resource).or_default();
                *bytes = (*bytes).max(binding.bytes);
            }
        }
        let mut dag = crate::prelude::petgraph::graph::DiGraph::<(), ()>::new();
        let execution = dag.add_node(());
        let mut requests = vec![SlabBuffer {
            id: None,
            bytes: scratch_bytes,
            uses: vec![execution],
        }];
        requests.extend(resources.iter().map(|(&id, &bytes)| SlabBuffer {
            id: Some(id),
            bytes,
            uses: vec![execution],
        }));
        let offsets = plan_slab(
            &dag,
            &requests,
            capacity,
            ARENA_ALIGN,
            SlabAlgorithm::FirstFit,
        )?;
        let mut bytes = 0;
        let mut homes = BTreeMap::new();
        for (request, (id, offset)) in requests.iter().zip(offsets) {
            let home = ArenaSlice {
                offset,
                bytes: request.bytes,
            };
            bytes = bytes.max(offset + home.reserved());
            if let Some(id) = id {
                homes.insert(id, home);
            }
        }
        let bindings = programs
            .iter()
            .map(|p| {
                p.bindings
                    .iter()
                    .map(|b| (b.buffer, homes[&b.resource]))
                    .collect()
            })
            .collect();
        Ok(Self {
            scratch_bytes,
            bytes,
            homes,
            bindings,
        })
    }

    pub fn bindings(&self, program: usize) -> &BTreeMap<i64, ArenaSlice> {
        &self.bindings[program]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_resources_use_max_capacity_without_tensor_policy() {
        let programs = [
            ProgramMemory {
                scratch_bytes: 256,
                bindings: vec![PersistentBinding {
                    resource: ResourceId(0),
                    buffer: 1,
                    bytes: 64,
                }],
            },
            ProgramMemory {
                scratch_bytes: 512,
                bindings: vec![PersistentBinding {
                    resource: ResourceId(0),
                    buffer: 7,
                    bytes: 300,
                }],
            },
        ];
        let plan = SharedArenaPlan::build(&programs, 1024).unwrap();
        assert_eq!(plan.scratch_bytes, 512);
        assert_eq!(plan.bindings(0)[&1], plan.bindings(1)[&7]);
        assert_eq!(plan.homes[&ResourceId(0)].bytes, 300);
        assert_eq!(plan.bytes, 1024);
        assert!(SharedArenaPlan::build(&programs, 1023).is_err());
    }
}
