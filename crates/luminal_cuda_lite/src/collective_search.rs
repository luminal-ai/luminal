//! Coordinator-owned implementation search for collective programs.
//!
//! Rank zero owns the compiler and sends admitted genomes to profiling
//! workers. Workers never sample, rank, or select implementations. They only
//! materialize the coordinator's exact candidate and enter the same fixed
//! warmup/trial sequence, so every authored collective is launched in the
//! same order on every rank.

#[cfg(feature = "device")]
use anyhow::anyhow;
#[cfg(any(feature = "device", test))]
use anyhow::{Context, Result, ensure};
#[cfg(any(feature = "device", test))]
use luminal::extraction::{Genome, ProducerChoice};
#[cfg(any(feature = "device", test))]
use luminal::prelude::egraph_serialize::ClassId;
#[cfg(any(feature = "device", test))]
use luminal::prelude::egraph_serialize::EGraph;
#[cfg(any(feature = "device", test))]
use luminal::search_support::ProducerIndex;
#[cfg(any(feature = "device", test))]
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProfileRun {
    Warmup,
    Trial,
}

/// Authoritative launch sequence for every rank: one warmup followed by a
/// fixed, nonzero number of timed trials. Neither timing nor early-stop state
/// can alter this sequence on an individual rank.
pub(crate) fn fixed_profile_runs(trials: usize) -> impl Iterator<Item = ProfileRun> {
    std::iter::once(ProfileRun::Warmup).chain(std::iter::repeat_n(ProfileRun::Trial, trials.max(1)))
}

/// Stable dense representation of a genome relative to one producer index.
/// The index is sorted by class and every candidate list is deterministically
/// sorted by extraction, so the same pruned snapshot gives every rank the same
/// wire vocabulary.
#[derive(Debug, Clone)]
#[cfg(any(feature = "device", test))]
pub(crate) struct GenomeCodec {
    rows: Vec<(ClassId, Vec<ProducerChoice>)>,
}

#[cfg(any(feature = "device", test))]
impl GenomeCodec {
    pub(crate) fn new(index: &ProducerIndex) -> Self {
        Self {
            rows: index
                .iter()
                .map(|(class, choices)| {
                    (
                        class.clone(),
                        choices.iter().map(|(_, choice)| choice.clone()).collect(),
                    )
                })
                .collect(),
        }
    }

    pub(crate) fn encode(&self, genome: &Genome) -> Result<Vec<u32>> {
        self.rows
            .iter()
            .map(|(class, choices)| {
                let chosen = genome.choices.get(class).with_context(|| {
                    format!("collective candidate omits producer class {class}")
                })?;
                let position = choices
                    .iter()
                    .position(|candidate| candidate == chosen)
                    .with_context(|| {
                        format!("collective candidate names an unknown producer for class {class}")
                    })?;
                u32::try_from(position).context("collective producer index exceeds u32")
            })
            .collect()
    }

    pub(crate) fn decode(&self, encoded: &[u32]) -> Result<Genome> {
        ensure!(
            encoded.len() == self.rows.len(),
            "collective candidate has {} producer rows, expected {}",
            encoded.len(),
            self.rows.len()
        );
        let mut genome = Genome::default();
        for ((class, choices), &position) in self.rows.iter().zip(encoded) {
            let choice = choices
                .get(usize::try_from(position)?)
                .with_context(|| {
                    format!("collective producer {position} is out of range for class {class}")
                })?
                .clone();
            genome.choices.insert(class.clone(), choice);
        }
        Ok(genome)
    }
}

#[derive(Serialize, Deserialize)]
#[cfg(any(feature = "device", test))]
pub(crate) struct Begin {
    pub(crate) graph: EGraph,
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
    Candidate(Vec<u32>),
    Finish(Vec<u32>),
    Abort(String),
}

#[cfg(feature = "device")]
#[doc(hidden)]
pub struct Coordinator {
    group: std::rc::Rc<crate::collective::NcclCommunicator>,
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
    codec: Option<GenomeCodec>,
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
            codec: None,
            finished: false,
        })
    }

    pub(crate) fn begin(
        &mut self,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        graph: &EGraph,
        index: &ProducerIndex,
        trials: usize,
        arena_budget_bytes: usize,
    ) -> Result<()> {
        ensure!(self.codec.is_none(), "collective search began twice");
        let payload = bincode::serialize(&Handshake::Begin(Box::new(Begin {
            graph: graph.clone(),
            trials: trials.max(1),
            arena_budget_bytes,
        })))?;
        self.group.broadcast_bytes(stream, Some(&payload))?;
        self.codec = Some(GenomeCodec::new(index));
        Ok(())
    }

    pub(crate) fn candidate(
        &self,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        genome: &Genome,
    ) -> Result<()> {
        let encoded = self
            .codec
            .as_ref()
            .context("collective search has not begun")?
            .encode(genome)?;
        self.send(stream, &Command::Candidate(encoded))
    }

    pub(crate) fn finish(
        &mut self,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        genome: &Genome,
    ) -> Result<()> {
        let encoded = self
            .codec
            .as_ref()
            .context("collective search has not begun")?
            .encode(genome)?;
        self.send(stream, &Command::Finish(encoded))?;
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
        let payload = if self.codec.is_some() {
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
    codec: Option<GenomeCodec>,
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
        Ok(Self {
            group,
            begin,
            codec: None,
        })
    }

    pub(crate) fn install_index(&mut self, index: &ProducerIndex) {
        self.codec = Some(GenomeCodec::new(index));
    }

    pub(crate) fn next(
        &self,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    ) -> Result<WorkerCommand> {
        let payload = self.group.broadcast_bytes(stream, None)?;
        let command: Command =
            bincode::deserialize(&payload).context("decode collective search command")?;
        let codec = self
            .codec
            .as_ref()
            .context("collective profiling worker has no producer index")?;
        match command {
            Command::Candidate(encoded) => Ok(WorkerCommand::Candidate(codec.decode(&encoded)?)),
            Command::Finish(encoded) => Ok(WorkerCommand::Finish(codec.decode(&encoded)?)),
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
    Candidate(Genome),
    Finish(Genome),
}

#[cfg(test)]
mod tests {
    use super::*;
    use luminal::prelude::egraph_serialize::{ClassId, NodeId};

    fn fixture() -> (ProducerIndex, Genome) {
        let c0 = ClassId::dense(3);
        let c1 = ClassId::dense(7);
        let a = ProducerChoice {
            enode: NodeId::dense(11),
            output_index: 0,
        };
        let b = ProducerChoice {
            enode: NodeId::dense(12),
            output_index: 1,
        };
        let c = ProducerChoice {
            enode: NodeId::dense(19),
            output_index: 0,
        };
        let index = [
            (
                c0.clone(),
                vec![("A".into(), a.clone()), ("B".into(), b.clone())],
            ),
            (c1.clone(), vec![("C".into(), c.clone())]),
        ]
        .into_iter()
        .collect();
        let mut genome = Genome::default();
        genome.choices.insert(c0, b);
        genome.choices.insert(c1, c);
        (index, genome)
    }

    #[test]
    fn dense_genome_wire_round_trips_against_the_shared_index() {
        let (index, genome) = fixture();
        let codec = GenomeCodec::new(&index);
        let encoded = codec.encode(&genome).unwrap();
        assert_eq!(encoded, [1, 0]);
        assert_eq!(codec.decode(&encoded).unwrap().choices, genome.choices);
    }

    #[test]
    fn dense_genome_wire_refuses_different_candidate_vocabularies() {
        let (index, genome) = fixture();
        let codec = GenomeCodec::new(&index);
        let mut encoded = codec.encode(&genome).unwrap();
        encoded[1] = 4;
        assert!(
            codec
                .decode(&encoded)
                .unwrap_err()
                .to_string()
                .contains("out of range")
        );
        assert!(
            codec
                .decode(&encoded[..1])
                .unwrap_err()
                .to_string()
                .contains("expected 2")
        );
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
            graph: EGraph::default(),
            trials: 3,
            arena_budget_bytes: 4096,
        })))
        .unwrap();
        let Handshake::Begin(begin) = bincode::deserialize(&payload).unwrap() else {
            panic!("begin handshake changed variant")
        };
        assert_eq!(begin.trials, 3);
        assert_eq!(begin.arena_budget_bytes, 4096);
        assert!(begin.graph.nodes.is_empty());

        let payload = bincode::serialize(&Command::Candidate(vec![2, 0, 4])).unwrap();
        let Command::Candidate(candidate) = bincode::deserialize(&payload).unwrap() else {
            panic!("candidate command changed variant")
        };
        assert_eq!(candidate, [2, 0, 4]);
    }
}
