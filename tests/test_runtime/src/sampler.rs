//! THE GENOME SAMPLER — the test runtime's copy.
//!
//! The two-phase sampler over the candidate graph's strongly connected
//! components (rulings 2026-08-07 / 2026-09-02), duplicated here with
//! the extractor it drives (ruling 2026-09-03, #420/#422 rejoin Phase 1:
//! post-saturation search is runtime-owned, and this crate's charter
//! forbids borrowing another runtime's copy). The SELECTION LOOP is not
//! here: this runtime does not execute plans, so it has nothing to price
//! them with — it exercises extraction and bufferization directly.
//!
//! Tests live with the reference copy (`luminal_reference::search`).

use std::collections::BTreeMap;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rustc_hash::FxHashSet;

use egraph_serialize::ClassId;

use crate::extractor::{Genome, ProducerChoice, SamplingSpace};

/// Producer index shorthand: class -> its candidate `(constructor
/// name, choice)` entries, in the index's own deterministic order.
pub type ProducerIndex = BTreeMap<ClassId, Vec<(String, ProducerChoice)>>;

/// Pick a position: uniformly among `allowed`, or — when nothing is
/// admissible — uniformly over the FULL candidate list.
///
/// THE FALLBACK is deliberate and unchanged from the 2026-08-07
/// sampler: a class every one of whose candidates sources from a member
/// of its own component that is not yet assigned has no acyclic option
/// at this position, which means its component holds no acyclic genome
/// at all through this class. Refusing to choose would silently shrink
/// the space; choosing anyway keeps the refusal accounting (and the
/// blockage anatomy) as the loud diagnosis for that corner.
fn choose_position(rng: &mut StdRng, allowed: &[usize], total: usize) -> usize {
    if allowed.is_empty() {
        rng.random_range(0..total)
    } else {
        allowed[rng.random_range(0..allowed.len())]
    }
}

/// GENERATION-0 SAMPLING: a FOREST inside every component.
///
/// Each component's members are assigned in a random order built one
/// step at a time. A member may elect a candidate with
/// intra-component sources only when ALL of them are already assigned,
/// so every chosen intra-component edge points BACKWARD along the
/// order — acyclic by construction.
///
/// THE ORDER IS DRAWN FROM THE ADMISSIBLE ONES: at each step the next
/// member is picked uniformly among those that still HAVE an
/// admissible candidate, so the first member always progresses and no
/// member is forced into the fallback by an unlucky shuffle. (A
/// uniformly random order over all members would sometimes lead with a
/// member that cannot progress — e.g. a layout copy whose only sources
/// are its own component — and then the fallback could weld a cycle
/// the 2026-08-07 sampler would have refused. Restricting to
/// admissible orders removes no genome: see COVERAGE.)
///
/// COVERAGE. This admits chains and forests, not just the 2026-08-07
/// star (one elected primary, everyone else copying it): for any
/// acyclic assignment of intra-component candidates there is a
/// topological order of the chosen edges, at every prefix of which the
/// next member's own chosen candidate is admissible — so that member
/// is in the pool, and the assignment is reachable. The only genomes
/// excluded are the cyclic ones. Classes outside every component are
/// sampled freely: their candidates' edges all leave the component,
/// and the condensation of the SCC decomposition is a DAG.
pub fn sample_genome(index: &ProducerIndex, space: &SamplingSpace, rng: &mut StdRng) -> Genome {
    sample_genome_reporting(index, space, rng).0
}

/// [`sample_genome`] plus the classes that had to take the full-list
/// fallback, in the order they took it — the sampler's own account of
/// where its acyclicity invariant was unenforceable. An EMPTY list is
/// the guarantee: the genome's chosen-edge graph is acyclic.
pub fn sample_genome_reporting(
    index: &ProducerIndex,
    space: &SamplingSpace,
    rng: &mut StdRng,
) -> (Genome, Vec<ClassId>) {
    let mut genome = Genome::default();
    let mut fallbacks: Vec<ClassId> = Vec::new();
    for members in &space.components {
        // Per member, per candidate: how many of its intra-component
        // sources are still unassigned. Zero = admissible now. (Sources
        // are deduplicated, so one decrement each.)
        let mut pending: Vec<Vec<usize>> = members
            .iter()
            .map(|class| {
                space.intra_sources[class]
                    .iter()
                    .map(Vec::len)
                    .collect::<Vec<usize>>()
            })
            .collect();
        // How many admissible candidates each member currently has.
        let mut admissible: Vec<usize> = pending
            .iter()
            .map(|per_candidate| per_candidate.iter().filter(|left| **left == 0).count())
            .collect();
        // source class -> the (member, candidate) counters it releases.
        let mut dependents: std::collections::BTreeMap<&ClassId, Vec<(usize, usize)>> =
            std::collections::BTreeMap::new();
        for (member, class) in members.iter().enumerate() {
            for (candidate, sources) in space.intra_sources[class].iter().enumerate() {
                for source in sources {
                    dependents
                        .entry(source)
                        .or_default()
                        .push((member, candidate));
                }
            }
        }

        let mut assigned = vec![false; members.len()];
        for _ in 0..members.len() {
            // The next member: uniform over those that can still choose
            // admissibly, or — when none can — over whoever is left,
            // which is the full-list fallback.
            let mut pool: Vec<usize> = (0..members.len())
                .filter(|member| !assigned[*member] && admissible[*member] > 0)
                .collect();
            let forced = pool.is_empty();
            if forced {
                pool = (0..members.len())
                    .filter(|member| !assigned[*member])
                    .collect();
            }
            let member = pool[rng.random_range(0..pool.len())];
            let class = &members[member];
            let candidates = &index[class];
            let allowed: Vec<usize> = (0..candidates.len())
                .filter(|position| pending[member][*position] == 0)
                .collect();
            debug_assert_eq!(allowed.is_empty(), forced);
            if forced {
                fallbacks.push(class.clone());
            }
            let position = choose_position(rng, &allowed, candidates.len());
            genome
                .choices
                .insert(class.clone(), candidates[position].1.clone());
            assigned[member] = true;
            for (other, candidate) in dependents.get(class).into_iter().flatten() {
                pending[*other][*candidate] -= 1;
                if pending[*other][*candidate] == 0 {
                    admissible[*other] += 1;
                }
            }
        }
    }
    for (class, candidates) in index {
        if genome.choices.contains_key(class) {
            continue;
        }
        let position = rng.random_range(0..candidates.len());
        genome
            .choices
            .insert(class.clone(), candidates[position].1.clone());
    }
    (genome, fallbacks)
}

/// Would routing `class` through `sources` close a cycle in the genome's
/// chosen intra-component edge graph? A DFS from each source over the
/// OTHER members' current choices; reaching `class` again — or a source
/// that IS `class` — is the cycle. Intra-component edges never leave the
/// component, so the walk is component-local and small.
fn flip_closes_cycle(
    index: &ProducerIndex,
    space: &SamplingSpace,
    genome: &Genome,
    class: &ClassId,
    sources: &[ClassId],
) -> bool {
    let mut stack: Vec<ClassId> = sources.to_vec();
    let mut seen: FxHashSet<ClassId> = FxHashSet::default();
    while let Some(node) = stack.pop() {
        if &node == class {
            return true;
        }
        if !seen.insert(node.clone()) {
            continue;
        }
        let Some(position) = space.chosen_position(index, genome, &node) else {
            continue;
        };
        if let Some(next) = space
            .intra_sources
            .get(&node)
            .and_then(|per_candidate| per_candidate.get(position))
        {
            stack.extend(next.iter().cloned());
        }
    }
    false
}

/// POINT MUTATION under the same invariant: a flip of one class to one
/// candidate is admissible exactly when the resulting chosen-edge graph
/// closes no cycle through that class (see [`flip_closes_cycle`]).
/// Progressing candidates have no intra-component sources and so are
/// always admissible; a candidate sourcing from its own class never is.
/// Mutations hit ANY producer class, dead rows included (deliberately —
/// a dead-row mutation is free now and pre-stages the choice a later
/// route flip lands on).
pub fn mutate_genome(
    parent: &Genome,
    index: &ProducerIndex,
    space: &SamplingSpace,
    classes: &[ClassId],
    rng: &mut StdRng,
    count: usize,
) -> Genome {
    mutate_genome_reporting(parent, index, space, classes, rng, count).0
}

/// [`mutate_genome`] plus the classes whose flip found NO admissible
/// candidate and took the full-list fallback (see
/// [`sample_genome_reporting`]).
pub fn mutate_genome_reporting(
    parent: &Genome,
    index: &ProducerIndex,
    space: &SamplingSpace,
    classes: &[ClassId],
    rng: &mut StdRng,
    count: usize,
) -> (Genome, Vec<ClassId>) {
    let mut child = parent.clone();
    let mut fallbacks: Vec<ClassId> = Vec::new();
    if classes.is_empty() {
        return (child, fallbacks); // one-point genome space: nothing to mutate
    }
    for _ in 0..count {
        let class = &classes[rng.random_range(0..classes.len())];
        let candidates = &index[class];
        let sources = &space.intra_sources[class];
        let allowed: Vec<usize> = (0..candidates.len())
            .filter(|&position| !flip_closes_cycle(index, space, &child, class, &sources[position]))
            .collect();
        if allowed.is_empty() {
            fallbacks.push(class.clone());
        }
        let position = choose_position(rng, &allowed, candidates.len());
        child
            .choices
            .insert(class.clone(), candidates[position].1.clone());
    }
    (child, fallbacks)
}

/// [`sample_genome_reporting`] from a seed — the entry the
/// sampler-invariant boards use, so a test crate needs no `rand` of its
/// own. Returns the genome and its fallback classes.
pub fn sample_genome_with_seed(
    index: &ProducerIndex,
    space: &SamplingSpace,
    seed: u64,
) -> (Genome, Vec<ClassId>) {
    sample_genome_reporting(index, space, &mut StdRng::seed_from_u64(seed))
}

/// [`mutate_genome_reporting`] from a seed (see
/// [`sample_genome_with_seed`]).
pub fn mutate_genome_with_seed(
    parent: &Genome,
    index: &ProducerIndex,
    space: &SamplingSpace,
    count: usize,
    seed: u64,
) -> (Genome, Vec<ClassId>) {
    let classes: Vec<ClassId> = index.keys().cloned().collect();
    mutate_genome_reporting(
        parent,
        index,
        space,
        &classes,
        &mut StdRng::seed_from_u64(seed),
        count,
    )
}
