//! KV-cache forms (zoo ruling 2026-08-12: CACHE STRUCTURE IS MODEL
//! DEFINITION; the one compromise is a BUILDER that slots cache forms,
//! with the variants living here and different runtime examples
//! exercising different forms).
//!
//! Graph-side, every zoo model shares ONE structure — the per-layer
//! SLOT POOL: a (slots, kv_dim) input pair per layer, coordinate-form
//! scatter append, coordinate-form gather read, windowing enforced by
//! the attention mask (never by eviction). What genuinely varies is
//! HOST-side: how logical positions map to physical slots, and whether
//! the causal/isolation mask derives in-graph from `q_pos` or arrives
//! as data (the multi-sequence page-table form). Those variants are
//! the [`PositionSlots`] / [`PageTable`] drivers below.

use luminal::graph::Graph;
use luminal::prelude::Ns;
use luminal::prelude::anyhow;
use luminal::prelude::{GraphTensor, TypedBuffer};
use luminal::reference::ReferenceRuntime;

/// The slot-pool cache: one (slots, kv_dim) K/V input pair per layer.
/// `kv_dim` is per-layer (gemma-4's sliding and full layers differ),
/// via [`KvCachePool::new_heterogeneous`].
pub struct KvCachePool {
    pub layers: Vec<(GraphTensor, GraphTensor)>,
    pub kv_dims: Vec<usize>,
    pub slots: usize,
}

impl KvCachePool {
    pub fn new(cx: &mut Graph, layers: usize, slots: usize, kv_dim: usize, ns: &Ns) -> Self {
        Self::new_heterogeneous(cx, slots, &vec![kv_dim; layers], ns)
    }

    /// Per-layer kv dims (heterogeneous pools — gemma-4's role-split
    /// layers).
    pub fn new_heterogeneous(cx: &mut Graph, slots: usize, kv_dims: &[usize], ns: &Ns) -> Self {
        let layers = kv_dims
            .iter()
            .enumerate()
            .map(|(l, kv_dim)| {
                let layer_ns = ns.index(l);
                (
                    cx.named_tensor(layer_ns.leaf("k_cache"), (slots, *kv_dim)),
                    cx.named_tensor(layer_ns.leaf("v_cache"), (slots, *kv_dim)),
                )
            })
            .collect();
        Self {
            layers,
            kv_dims: kv_dims.to_vec(),
            slots,
        }
    }

    /// The reference-runtime state carrier: host copies of every pool,
    /// zero-initialized, flowing runtime-out → runtime-in per step.
    pub fn zero_state(&self) -> CacheState {
        CacheState {
            k: self
                .kv_dims
                .iter()
                .map(|kv_dim| vec![0f32; self.slots * kv_dim])
                .collect(),
            v: self
                .kv_dims
                .iter()
                .map(|kv_dim| vec![0f32; self.slots * kv_dim])
                .collect(),
        }
    }
}

/// Host-side cache contents between steps (reference examples round-
/// trip state through get_f32/set_data; a device runtime would alias
/// the buffers instead — a BINDING decision, not model text).
pub struct CacheState {
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}

impl CacheState {
    /// Stage every layer's cache contents as this step's inputs.
    pub fn stage(&self, rt: &mut ReferenceRuntime, pool: &KvCachePool) {
        for (layer, (k, v)) in pool.layers.iter().enumerate() {
            rt.set_data(k.id, self.k[layer].clone());
            rt.set_data(v.id, self.v[layer].clone());
        }
    }

    /// Read every layer's post-step cache outputs back into the state.
    pub fn absorb(
        &mut self,
        rt: &ReferenceRuntime,
        cache_outs: &[(GraphTensor, GraphTensor)],
    ) -> anyhow::Result<()> {
        for (layer, (k, v)) in cache_outs.iter().enumerate() {
            self.k[layer] = rt.get_f32(k.id)?.clone();
            self.v[layer] = rt.get_f32(v.id)?.clone();
        }
        Ok(())
    }
}

/// Driver 1 — POSITION SLOTS: physical slot == absolute position (the
/// llama/gemma/qwen/whisper form). Single sequence, capacity = slots.
pub struct PositionSlots {
    pub pos: usize,
    pub slots: usize,
}

impl PositionSlots {
    pub fn new(slots: usize) -> Self {
        Self { pos: 0, slots }
    }

    /// The per-step index data for one incoming token: scatter at the
    /// frontier, q_pos = the frontier (the step-invariant decode form —
    /// gather_idx stays the full 0..slots arange, masked by position).
    pub fn step(&mut self) -> anyhow::Result<(TypedBuffer, TypedBuffer)> {
        anyhow::ensure!(
            self.pos < self.slots,
            "sequence exceeded {} slots",
            self.slots
        );
        let scatter: TypedBuffer = vec![self.pos as i32].into();
        let q_pos: TypedBuffer = vec![self.pos as i32].into();
        self.pos += 1;
        Ok((scatter, q_pos))
    }

    /// The step-invariant full-pool gather list (staged once).
    pub fn full_gather(&self) -> TypedBuffer {
        (0..self.slots as i32).collect::<Vec<i32>>().into()
    }
}

/// Driver 2 — PAGE TABLE: logical position → physical slot through a
/// per-sequence table with bump allocation (the paged_llama form).
/// Multi-sequence: one pool serves several sequences at once, and the
/// attention mask must arrive as DATA to isolate them (see
/// [`crate::paged_attention_masked`]).
pub struct PageTable {
    /// Per sequence: logical position → physical slot.
    pub sequences: Vec<Vec<usize>>,
    next_free_slot: usize,
    pub slots: usize,
}

impl PageTable {
    pub fn new(slots: usize) -> Self {
        Self {
            sequences: Vec::new(),
            next_free_slot: 0,
            slots,
        }
    }

    pub fn new_sequence(&mut self) -> usize {
        self.sequences.push(Vec::new());
        self.sequences.len() - 1
    }

    /// Allocate physical slots for `n` new tokens of a sequence.
    pub fn allocate(&mut self, sequence: usize, n: usize) -> anyhow::Result<Vec<usize>> {
        anyhow::ensure!(
            self.next_free_slot + n <= self.slots,
            "page table exhausted: {} + {n} > {}",
            self.next_free_slot,
            self.slots
        );
        let fresh: Vec<usize> = (self.next_free_slot..self.next_free_slot + n).collect();
        self.next_free_slot += n;
        self.sequences[sequence].extend(&fresh);
        Ok(fresh)
    }

    /// Assemble one BATCHED tick over `(sequence, n_new)` pairs:
    /// returns (scatter_idx, gather_idx, additive mask rows) where the
    /// mask encodes BOTH causality and cross-sequence isolation — a
    /// query sees exactly its own sequence's context prefix. Shapes:
    /// scatter (total_s,), gather (total_c,), mask (total_s, total_c)
    /// with 0.0 visible / -1e9 hidden.
    pub fn tick(
        &mut self,
        requests: &[(usize, usize)],
    ) -> anyhow::Result<(TypedBuffer, TypedBuffer, TypedBuffer, usize, usize)> {
        let mut scatter: Vec<i32> = Vec::new();
        // (sequence, first new logical position) per request, recorded
        // before allocation so the mask can compute visibility.
        let mut new_spans: Vec<(usize, usize)> = Vec::new();
        for (sequence, n_new) in requests {
            let start = self.sequences[*sequence].len();
            new_spans.push((*sequence, start));
            let fresh = self.allocate(*sequence, *n_new)?;
            scatter.extend(fresh.iter().map(|slot| *slot as i32));
        }
        // Gather = concatenation of every REQUESTED sequence's full slot
        // list (post-allocation, so a step's own rows are visible).
        let mut gather: Vec<i32> = Vec::new();
        let mut ctx_spans: Vec<(usize, usize, usize)> = Vec::new(); // (sequence, ctx_start, ctx_len)
        for (sequence, _) in requests {
            let list = &self.sequences[*sequence];
            ctx_spans.push((*sequence, gather.len(), list.len()));
            gather.extend(list.iter().map(|slot| *slot as i32));
        }
        let total_s = scatter.len();
        let total_c = gather.len();
        let mut mask = vec![-1e9f32; total_s * total_c];
        let mut row = 0usize;
        for (request_index, (sequence, first_pos)) in new_spans.iter().enumerate() {
            let n_new = requests[request_index].1;
            let (_, ctx_start, ctx_len) = *ctx_spans
                .iter()
                .find(|(seq, _, _)| seq == sequence)
                .expect("requested sequence has a context span");
            for local in 0..n_new {
                let abs_pos = first_pos + local;
                for ci in 0..ctx_len {
                    if ci <= abs_pos {
                        mask[row * total_c + ctx_start + ci] = 0.0;
                    }
                }
                row += 1;
            }
        }
        Ok((scatter.into(), gather.into(), mask.into(), total_s, total_c))
    }
}

impl PageTable {
    /// One batched tick for the STEP-INVARIANT graph shape: the gather
    /// range is the WHOLE pool (columns = physical slots), so the
    /// (total_s, slots) mask alone decides visibility — causality,
    /// cross-sequence isolation, and unwritten-slot exclusion in one
    /// predicate: slot j is visible to a query at logical position p of
    /// sequence q iff j is one of q's slots with logical index <= p.
    pub fn tick_dense(
        &mut self,
        requests: &[(usize, usize)],
    ) -> anyhow::Result<(TypedBuffer, TypedBuffer, usize)> {
        let mut scatter: Vec<i32> = Vec::new();
        let mut new_spans: Vec<(usize, usize)> = Vec::new();
        for (sequence, n_new) in requests {
            new_spans.push((*sequence, self.sequences[*sequence].len()));
            let fresh = self.allocate(*sequence, *n_new)?;
            scatter.extend(fresh.iter().map(|slot| *slot as i32));
        }
        let total_s = scatter.len();
        let mut mask = vec![-1e9f32; total_s * self.slots];
        let mut row = 0usize;
        for (request_index, (sequence, first_pos)) in new_spans.iter().enumerate() {
            let n_new = requests[request_index].1;
            for local in 0..n_new {
                let abs_pos = first_pos + local;
                for (logical, slot) in self.sequences[*sequence].iter().enumerate() {
                    if logical <= abs_pos {
                        mask[row * self.slots + slot] = 0.0;
                    }
                }
                row += 1;
            }
        }
        Ok((scatter.into(), mask.into(), total_s))
    }
}

#[cfg(test)]
mod tests {
    use super::PageTable;

    /// Two sequences through one pool: bump allocation, per-request
    /// spans, and a batched tick whose mask encodes causality AND
    /// cross-sequence isolation exactly.
    #[test]
    fn page_table_tick_masks_isolation() {
        let mut table = PageTable::new(16);
        let a = table.new_sequence();
        let b = table.new_sequence();
        table.allocate(a, 2).expect("prefill A"); // slots 0, 1
        table.allocate(b, 1).expect("prefill B"); // slot 2

        // Batched decode: one new token each.
        let (scatter, gather, mask, total_s, total_c) =
            table.tick(&[(a, 1), (b, 1)]).expect("tick");
        assert_eq!(scatter.as_i32().unwrap(), &vec![3i32, 4]);
        // Gather: A's slots then B's, post-allocation.
        assert_eq!(gather.as_i32().unwrap(), &vec![0i32, 1, 3, 2, 4]);
        assert_eq!((total_s, total_c), (2, 5));
        let mask = mask.as_f32().unwrap();
        // Row 0 = A at position 2: sees A's ctx columns 0..3, none of B.
        assert_eq!(&mask[0..5], &[0.0, 0.0, 0.0, -1e9, -1e9]);
        // Row 1 = B at position 1: sees only B's ctx columns 3..5.
        assert_eq!(&mask[5..10], &[-1e9, -1e9, -1e9, 0.0, 0.0]);
    }
}
