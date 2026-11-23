//! Optional application composition for programs executed serially. Sharing is
//! declared by resource identity and byte capacity; tensor contracts belong to
//! each program and the application that binds it.
use std::collections::BTreeMap;

use crate::arena::{ARENA_ALIGN, ArenaSlice, SlabAlgorithm, SlabBuffer, plan_slab};
use anyhow::{Context, Result, ensure};

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
    /// The largest per-program scratch that still COMPOSES inside
    /// `capacity` next to `persistent`.
    ///
    /// [`Self::build`] places one shared scratch region and every declared
    /// resource against a single execution node, so none of them overlap and
    /// the arena is scratch PLUS the whole persistent footprint. A search
    /// that admits plans on scratch alone against a device capacity is
    /// therefore spending memory the weights already own: it accepts plans
    /// that `build` must then refuse, and it only finds out after the search
    /// has finished. Searching against this value instead keeps admission and
    /// composition on the same budget.
    ///
    /// Resources dedup by [`ResourceId`] at their largest declared size, the
    /// same rule `build` applies. The result is rounded down to
    /// [`ARENA_ALIGN`] so a plan reporting exactly this many scratch bytes
    /// composes without the alignment pushing it over.
    pub fn scratch_budget(
        persistent: impl IntoIterator<Item = (ResourceId, usize)>,
        capacity: usize,
    ) -> Result<usize> {
        let mut resources = BTreeMap::<ResourceId, usize>::new();
        for (resource, bytes) in persistent {
            let entry = resources.entry(resource).or_default();
            *entry = (*entry).max(bytes);
        }
        let mut reserved = 0usize;
        for &bytes in resources.values() {
            let slice = ArenaSlice { offset: 0, bytes };
            reserved = reserved
                .checked_add(slice.reserved())
                .context("persistent footprint overflows usize")?;
        }
        let free = capacity.checked_sub(reserved).with_context(|| {
            format!(
                "persistent resources reserve {reserved} bytes, more than the \
                 {capacity} byte budget"
            )
        })?;
        Ok(free / ARENA_ALIGN * ARENA_ALIGN)
    }

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

    /// THE GAP `scratch_budget` CLOSES: a plan whose scratch fits the whole
    /// device budget is admissible to a search that checks scratch alone,
    /// yet composition has to seat the weights in that same arena.
    #[test]
    fn scratch_inside_the_whole_budget_can_still_fail_composition() {
        const BUDGET: usize = 8 << 20;
        let bindings = vec![PersistentBinding {
            resource: ResourceId(0),
            buffer: 1,
            bytes: 6 << 20,
        }];
        let scratch = 3 << 20;
        assert!(scratch <= BUDGET, "a scratch-only check admits this plan");
        let programs = [ProgramMemory {
            scratch_bytes: scratch,
            bindings: bindings.clone(),
        }];
        assert!(SharedArenaPlan::build(&programs, BUDGET).is_err());
        let budget = SharedArenaPlan::scratch_budget([(ResourceId(0), 6 << 20)], BUDGET).unwrap();
        assert!(
            budget < scratch,
            "the honest budget rules the plan out up front"
        );
    }

    #[test]
    fn scratch_budget_is_the_exact_composable_remainder() {
        let bindings = vec![
            PersistentBinding {
                resource: ResourceId(0),
                buffer: 1,
                bytes: 300,
            },
            PersistentBinding {
                resource: ResourceId(1),
                buffer: 2,
                bytes: 100,
            },
        ];
        // 300 reserves 512 and 100 reserves 256, so 768 of 2048 is spoken for.
        let budget =
            SharedArenaPlan::scratch_budget(bindings.iter().map(|b| (b.resource, b.bytes)), 2048)
                .unwrap();
        assert_eq!(budget, 1280);
        let fits = [ProgramMemory {
            scratch_bytes: budget,
            bindings: bindings.clone(),
        }];
        assert_eq!(SharedArenaPlan::build(&fits, 2048).unwrap().bytes, 2048);
        let over = [ProgramMemory {
            scratch_bytes: budget + 1,
            bindings,
        }];
        assert!(
            SharedArenaPlan::build(&over, 2048).is_err(),
            "one byte past the budget must not compose"
        );
    }

    #[test]
    fn scratch_budget_dedups_a_resource_bound_by_several_programs() {
        let shared = [
            (ResourceId(0), 300),
            (ResourceId(0), 100),
            (ResourceId(1), 100),
        ];
        // ResourceId(0) counts once at its largest size, so 512 + 256.
        assert_eq!(SharedArenaPlan::scratch_budget(shared, 2048).unwrap(), 1280);
    }

    #[test]
    fn scratch_budget_refuses_a_budget_the_weights_alone_exceed() {
        let err = SharedArenaPlan::scratch_budget([(ResourceId(0), 4096)], 2048)
            .expect_err("weights larger than the budget leave no scratch");
        assert!(
            format!("{err:#}").contains("more than the 2048 byte budget"),
            "the refusal must name the budget: {err:#}"
        );
    }
}
