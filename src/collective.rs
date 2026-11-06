//! Runtime bindings for rank-ordered AllGather. The graph stores only a group
//! name and call identity; applications own communicators and their lifetime.
use anyhow::{Result, anyhow, bail, ensure};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllGatherSpec {
    pub group: String,
    pub width: usize,
    pub sequence: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorSignature {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub bits: usize,
}
impl TensorSignature {
    pub fn elements(&self) -> Result<usize> {
        ensure!(
            self.bits > 0 && self.bits.is_multiple_of(8),
            "AllGather requires byte-addressable physical storage"
        );
        self.shape.iter().try_fold(1usize, |n, d| {
            n.checked_mul(*d)
                .ok_or_else(|| anyhow!("AllGather volume overflow"))
        })
    }
    pub fn bytes(&self) -> Result<usize> {
        Ok(self
            .elements()?
            .checked_mul(self.bits)
            .ok_or_else(|| anyhow!("AllGather byte size overflow"))?
            .div_ceil(8))
    }
}

/// A rank's endpoint. Return one complete input per rank, concatenated in rank
/// order, using the backend's physical storage width.
pub trait AllGatherTransport: Send + Sync {
    fn size(&self) -> usize;
    fn all_gather(
        &self,
        spec: &AllGatherSpec,
        signature: &TensorSignature,
        input: &[u8],
    ) -> Result<Vec<u8>>;
}
thread_local! {
    static GROUPS: RefCell<HashMap<String, Arc<dyn AllGatherTransport>>> = RefCell::new(HashMap::new());
}
/// Bind a group on the calling thread. Nested scopes and unwinding restore the
/// previous binding. Reference and Metal execute synchronously in this scope.
pub fn with_group<R>(
    name: impl Into<String>,
    transport: Arc<dyn AllGatherTransport>,
    run: impl FnOnce() -> R,
) -> R {
    struct Restore(String, Option<Arc<dyn AllGatherTransport>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            GROUPS.with(|groups| {
                let mut groups = groups.borrow_mut();
                if let Some(old) = self.1.take() {
                    groups.insert(self.0.clone(), old);
                } else {
                    groups.remove(&self.0);
                }
            });
        }
    }
    let name = name.into();
    let old = GROUPS.with(|groups| groups.borrow_mut().insert(name.clone(), transport));
    let _restore = Restore(name, old);
    run()
}

pub fn gather(spec: &AllGatherSpec, signature: &TensorSignature, input: &[u8]) -> Result<Vec<u8>> {
    ensure!(spec.width > 0, "AllGather group is empty");
    ensure!(
        input.len() == signature.bytes()?,
        "AllGather input byte count mismatch"
    );
    let stacked = if spec.width == 1 {
        input.to_vec()
    } else {
        let group = GROUPS
            .with(|groups| groups.borrow().get(&spec.group).cloned())
            .ok_or_else(|| anyhow!("AllGather group {:?} is not bound", spec.group))?;
        ensure!(group.size() == spec.width, "AllGather group size mismatch");
        group.all_gather(spec, signature, input)?
    };
    ensure!(
        stacked.len()
            == input
                .len()
                .checked_mul(spec.width)
                .ok_or_else(|| anyhow!("AllGather output size overflow"))?,
        "AllGather transport output byte count mismatch"
    );
    Ok(stacked)
}

#[derive(Serialize, Deserialize, PartialEq, Eq)]
struct Header {
    spec: AllGatherSpec,
    signature: TensorSignature,
    round: u64,
}
enum Peers {
    Root(Vec<TcpStream>),
    Rank(TcpStream),
}
struct TcpState {
    peers: Peers,
    round: u64,
    failed: bool,
}
/// Portable, synchronous AllGather over persistent TCP connections. Rank zero
/// coordinates each group. All participants must call in the same order;
/// metadata mismatches fail the group rather than silently mixing tensors.
pub struct TcpAllGather {
    size: usize,
    state: Mutex<TcpState>,
}
impl TcpAllGather {
    /// Connect all ranks concurrently to the same address. The root binds that
    /// address; other ranks retry until `timeout`. Groups need distinct ports.
    pub fn connect(
        address: SocketAddr,
        rank: usize,
        size: usize,
        timeout: Duration,
    ) -> Result<Self> {
        ensure!(
            size > 0 && rank < size && !timeout.is_zero(),
            "invalid AllGather rank, size or timeout"
        );
        let deadline = Instant::now() + timeout;
        let configure = |stream: &TcpStream| -> Result<()> {
            stream.set_nodelay(true)?;
            stream.set_read_timeout(Some(timeout))?;
            stream.set_write_timeout(Some(timeout))?;
            Ok(())
        };
        let peers = if rank == 0 {
            let listener = TcpListener::bind(address)?;
            listener.set_nonblocking(true)?;
            let mut peers: Vec<Option<TcpStream>> = (1..size).map(|_| None).collect();
            while peers.iter().any(Option::is_none) {
                ensure!(
                    Instant::now() < deadline,
                    "AllGather group connection timed out"
                );
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        configure(&stream)?;
                        let peer = read_u64(&mut stream)? as usize;
                        let width = read_u64(&mut stream)? as usize;
                        ensure!(
                            width == size && peer > 0 && peer < size,
                            "AllGather peer rank/size mismatch"
                        );
                        ensure!(peers[peer - 1].is_none(), "duplicate AllGather rank");
                        peers[peer - 1] = Some(stream);
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(2))
                    }
                    Err(e) => return Err(e.into()),
                }
            }
            Peers::Root(peers.into_iter().map(Option::unwrap).collect())
        } else {
            let mut stream = loop {
                let left = deadline.saturating_duration_since(Instant::now());
                ensure!(!left.is_zero(), "AllGather group connection timed out");
                match TcpStream::connect_timeout(&address, left.min(Duration::from_millis(100))) {
                    Ok(stream) => break stream,
                    Err(_) => std::thread::sleep(Duration::from_millis(2)),
                }
            };
            configure(&stream)?;
            write_u64(&mut stream, rank as u64)?;
            write_u64(&mut stream, size as u64)?;
            Peers::Rank(stream)
        };
        Ok(Self {
            size,
            state: Mutex::new(TcpState {
                peers,
                round: 0,
                failed: false,
            }),
        })
    }
}
fn write_u64(stream: &mut TcpStream, value: u64) -> Result<()> {
    Ok(stream.write_all(&value.to_be_bytes())?)
}
fn read_u64(stream: &mut TcpStream) -> Result<u64> {
    let mut bytes = [0; 8];
    stream.read_exact(&mut bytes)?;
    Ok(u64::from_be_bytes(bytes))
}
fn read_header(stream: &mut TcpStream) -> Result<Header> {
    let len = usize::try_from(read_u64(stream)?)?;
    ensure!(len <= 65_536, "AllGather header is too large");
    let mut bytes = vec![0; len];
    stream.read_exact(&mut bytes)?;
    Ok(serde_json::from_slice(&bytes)?)
}
impl AllGatherTransport for TcpAllGather {
    fn size(&self) -> usize {
        self.size
    }
    fn all_gather(
        &self,
        spec: &AllGatherSpec,
        signature: &TensorSignature,
        input: &[u8],
    ) -> Result<Vec<u8>> {
        ensure!(
            spec.width == self.size && input.len() == signature.bytes()?,
            "AllGather local contract mismatch"
        );
        let total = input
            .len()
            .checked_mul(self.size)
            .ok_or_else(|| anyhow!("AllGather message overflow"))?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("AllGather lock poisoned"))?;
        ensure!(
            !state.failed,
            "AllGather group failed during an earlier call"
        );
        let expected = Header {
            spec: spec.clone(),
            signature: signature.clone(),
            round: state.round,
        };
        let result: Result<Vec<u8>> = (|| match &mut state.peers {
            Peers::Rank(stream) => {
                let header = serde_json::to_vec(&expected)?;
                write_u64(stream, header.len() as u64)?;
                stream.write_all(&header)?;
                // Root validates metadata before accepting a tensor-sized payload.
                ensure!(
                    read_u64(stream)? == 0,
                    "AllGather metadata mismatch on another rank"
                );
                stream.write_all(input)?;
                ensure!(read_u64(stream)? == 0, "AllGather failed on another rank");
                let mut output = vec![0; total];
                stream.read_exact(&mut output)?;
                Ok(output)
            }
            Peers::Root(peers) => {
                let mut valid = true;
                for stream in peers.iter_mut() {
                    valid &= read_header(stream)? == expected;
                }
                for stream in peers.iter_mut() {
                    write_u64(stream, u64::from(!valid))?;
                }
                ensure!(
                    valid,
                    "AllGather rank metadata disagree (group, sequence, shape, dtype or round)"
                );
                let mut output = vec![0; total];
                output[..input.len()].copy_from_slice(input);
                for (rank, stream) in peers.iter_mut().enumerate() {
                    if let Err(error) = stream
                        .read_exact(&mut output[(rank + 1) * input.len()..(rank + 2) * input.len()])
                    {
                        for stream in peers.iter_mut() {
                            let _ = write_u64(stream, 1);
                        }
                        return Err(error.into());
                    }
                }
                for stream in peers.iter_mut() {
                    write_u64(stream, 0)?;
                    stream.write_all(&output)?;
                }
                Ok(output)
            }
        })();
        if result.is_err() {
            state.failed = true;
        }
        state.round = state
            .round
            .checked_add(1)
            .ok_or_else(|| anyhow!("AllGather round overflow"))?;
        result
    }
}

/// Build a deterministic viable plan without running it. Distributed programs
/// must not execute collectives in rank-local timing/autotuning loops.
pub fn compile_first(
    graph: &egraph_serialize::EGraph,
    decoders: &crate::egglog_utils::eclass::ConstructorRegistry,
    matchers: &[Box<dyn crate::layout_ir::OpMatcher>],
    allow: &[&str],
) -> Result<crate::bufferize::BufferIrGraph<crate::layouts::DecodedLayout>> {
    use rand::SeedableRng;
    let mut session =
        crate::extraction::ExtractionSession::new_with_matcher_set(graph, Some(allow), matchers);
    let index = session.producer_index();
    let space = session.sampling_space(&index);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let view = crate::egglog_utils::eclass::EGraphView::new(graph, decoders);
    let mut cache = HashMap::new();
    let mut last = None;
    for _ in 0..128 {
        let genome = crate::search_support::sample_genome(&index, &space, &mut rng);
        match session.extract_with_genome(&genome) {
            Ok(Some(extracted)) => {
                let dps = crate::dps::dps_rewrite(&extracted);
                match crate::layouts::decode_layout_table(
                    &view,
                    &dps,
                    "collective compilation",
                    &mut cache,
                )
                .and_then(|table| crate::bufferize::bufferize(&dps, &table))
                {
                    Ok(plan) => return Ok(plan),
                    Err(e) => last = Some(e),
                }
            }
            Err(e) => last = Some(e),
            Ok(None) => {}
        }
    }
    bail!(
        "no viable collective plan: {}; {}",
        last.map_or_else(|| "extraction failed".into(), |e| e.to_string()),
        session.failure_breakdown().2
    )
}

/// Refuse asynchronous rank-local profiling of collective work. Use each
/// runtime's `compile` entry point, then execute the same program on all ranks.
pub fn ensure_local_profile(
    plan: &crate::bufferize::BufferIrGraph<crate::layouts::DecodedLayout>,
) -> Result<()> {
    ensure!(
        !plan.dag.node_weights().any(|node| matches!(node,
        crate::bufferize::BufferNode::Compute { op, .. } if op.label() == "AllGather")),
        "AllGather plans require compile(), not rank-local profiling"
    );
    Ok(())
}
