//! Dense rolled LLIR between indexed extraction and loop unrolling, plus the
//! program fingerprint search uses to evaluate each distinct LLIR only once.

use std::collections::hash_map::DefaultHasher;
use std::fmt::Write as FmtWrite;
use std::hash::{Hash, Hasher};

use petgraph::stable_graph::NodeIndex;

use crate::graph::LLIRGraph;
use crate::op::LLIROp;

/// Dense rolled LLIR produced by indexed e-graph extraction, before loop
/// materialization. Runtimes receive [`LLIRGraph`]s; this avoids constructing
/// a throwaway rolled `StableGraph` for every search candidate.
pub struct PackedLLIRGraph {
    pub ops: Vec<LLIROp>,
    pub incoming_offsets: Vec<usize>,
    pub incoming_sources: Vec<usize>,
    pub outgoing_offsets: Vec<usize>,
    pub outgoing_targets: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LlirFingerprint(u64, u64);

struct LlirFingerprintHasher {
    first: DefaultHasher,
    second: DefaultHasher,
}

impl LlirFingerprintHasher {
    fn new() -> Self {
        let mut first = DefaultHasher::new();
        let mut second = DefaultHasher::new();
        0x243f_6a88_85a3_08d3_u64.hash(&mut first);
        0x1319_8a2e_0370_7344_u64.hash(&mut second);
        Self { first, second }
    }

    pub fn fingerprint(&self) -> LlirFingerprint {
        LlirFingerprint(self.first.finish(), self.second.finish())
    }
}

impl Hasher for LlirFingerprintHasher {
    fn finish(&self) -> u64 {
        self.first.finish()
    }

    fn write(&mut self, bytes: &[u8]) {
        self.first.write(bytes);
        self.second.write(bytes);
    }
}

struct HashWriter<'a>(&'a mut LlirFingerprintHasher);

impl FmtWrite for HashWriter<'_> {
    fn write_str(&mut self, value: &str) -> std::fmt::Result {
        self.0.write(value.as_bytes());
        Ok(())
    }
}

impl PackedLLIRGraph {
    /// Fingerprint the exact LLIR program represented by this packed graph.
    /// Packed extraction has deterministic node ordering, so the op sequence
    /// plus ordered incoming edges completely describes the graph; outgoing
    /// edges are derived from those inputs and do not need to be hashed again.
    pub fn fingerprint(&self) -> LlirFingerprint {
        let mut hasher = LlirFingerprintHasher::new();
        self.ops.len().hash(&mut hasher);
        for op in &self.ops {
            // Debug is the full semantic representation of a type-erased LLIR
            // op. Display is intentionally unsuitable: many ops omit their
            // shape and stride metadata there. Stream directly into the hash
            // to avoid allocating one String per op and candidate.
            write!(HashWriter(&mut hasher), "{op:?}").unwrap();
            hasher.write_u8(0xff);
        }
        self.incoming_offsets.hash(&mut hasher);
        self.incoming_sources.hash(&mut hasher);
        hasher.fingerprint()
    }

    pub fn to_stable(&self) -> LLIRGraph {
        let mut graph = LLIRGraph::with_capacity(self.ops.len(), self.incoming_sources.len());
        for op in &self.ops {
            graph.add_node(op.clone());
        }
        for destination in 0..graph.node_count() {
            for &source in &self.incoming_sources
                [self.incoming_offsets[destination]..self.incoming_offsets[destination + 1]]
            {
                graph.add_edge(NodeIndex::new(source), NodeIndex::new(destination), ());
            }
        }
        graph
    }
}
