use crate::{backend::Backend, graph::LlmGraph, sampling::Sampler};
use anyhow::{Result, ensure};
use std::{
    collections::BTreeSet,
    time::{Duration, Instant},
};

/// Per-turn latency measurements. `ttft` is prompt processing plus the first
/// sampled token; `decode` spans the first to the last sampled token, so it
/// excludes both the prefill and the terminal token's cache update.
#[derive(Clone, Copy, Debug, Default)]
pub struct GenerationTimings {
    pub prompted_tokens: usize,
    pub generated_tokens: usize,
    pub prefill: Duration,
    pub ttft: Duration,
    pub decode: Duration,
    pub total: Duration,
}
impl GenerationTimings {
    /// Mean inter-token latency after the first token; `None` when fewer than
    /// two tokens were generated.
    pub fn tpot(&self) -> Option<Duration> {
        (self.generated_tokens > 1).then(|| self.decode / (self.generated_tokens as u32 - 1))
    }
}

/// How many of `prompt`'s leading tokens the KV cache already holds, or
/// `None` when the cached history is not a prefix of the prompt and the
/// cache has to be dropped. TOKEN prefixes decide this, never rendered
/// strings: the KV state is a function of the token history.
fn reusable(cached: &[u32], prompt: &[u32]) -> Option<usize> {
    prompt.starts_with(cached).then_some(cached.len())
}

pub struct Session<B: Backend> {
    pub graph: LlmGraph,
    pub backend: B,
    cached: Vec<u32>,
    last_logits: Option<Vec<f32>>,
    timings: Option<GenerationTimings>,
}
impl<B: Backend> Session<B> {
    pub fn new(graph: LlmGraph, backend: B) -> Self {
        Self {
            graph,
            backend,
            cached: vec![],
            last_logits: None,
            timings: None,
        }
    }
    /// The token history the KV state currently represents.
    pub fn cached(&self) -> &[u32] {
        &self.cached
    }
    /// Measurements from the most recent [`Session::generate`] call.
    pub fn timings(&self) -> Option<GenerationTimings> {
        self.timings
    }
    pub fn reset(&mut self) -> Result<()> {
        self.backend.reset()?;
        self.cached.clear();
        self.last_logits = None;
        Ok(())
    }
    fn ingest(&mut self, tokens: &[u32]) -> Result<()> {
        for chunk in tokens.chunks(self.graph.chunk_size) {
            let inputs = self.graph.step_inputs(chunk, self.cached.len())?;
            let logits =
                match self
                    .backend
                    .step(inputs, chunk.len(), self.cached.len() + chunk.len())
                {
                    Ok(logits) => logits,
                    Err(e) => {
                        self.reset()?;
                        return Err(e);
                    }
                };
            if logits.len() != self.graph.vocab {
                self.reset()?;
                anyhow::bail!(
                    "backend returned {} logits, expected {}",
                    logits.len(),
                    self.graph.vocab
                );
            }
            self.cached.extend_from_slice(chunk);
            self.last_logits = Some(logits);
        }
        Ok(())
    }
    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_new_tokens: usize,
        stops: &BTreeSet<u32>,
        sampler: &mut Sampler,
        mut emit: impl FnMut(u32) -> Result<()>,
    ) -> Result<Vec<u32>> {
        ensure!(!prompt.is_empty(), "empty prompt");
        ensure!(
            prompt.len() < self.graph.capacity,
            "prompt fills the context; use /reset or increase --max-context"
        );
        ensure!(max_new_tokens > 0, "max-new-tokens must be positive");
        let start = Instant::now();
        let consumed = match reusable(&self.cached, prompt) {
            Some(consumed) => consumed,
            None => {
                self.reset()?;
                0
            }
        };
        self.ingest(&prompt[consumed..])?;
        let prefill = start.elapsed();
        let limit = max_new_tokens.min(self.graph.capacity - self.cached.len());
        let mut generated = vec![];
        let mut first_sampled = None;
        let mut last_sampled = None;
        for _ in 0..limit {
            let token = sampler.sample(self.last_logits.as_ref().expect("nonempty prefill"))?;
            let sampled = Instant::now();
            first_sampled.get_or_insert(sampled);
            last_sampled = Some(sampled);
            // Consume even the terminal token so the next turn's prefix check
            // reflects the complete token history represented by the KV state.
            self.ingest(&[token])?;
            generated.push(token);
            if stops.contains(&token) {
                break;
            }
            emit(token)?;
        }
        let total = start.elapsed();
        let first = first_sampled.unwrap_or(start);
        let last = last_sampled.unwrap_or(first);
        self.timings = Some(GenerationTimings {
            prompted_tokens: prompt.len() - consumed,
            generated_tokens: generated.len(),
            prefill,
            ttft: first.duration_since(start),
            decode: last.duration_since(first),
            total,
        });
        Ok(generated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_cache_is_reusable_exactly_when_it_is_a_token_prefix() {
        assert_eq!(reusable(&[], &[1, 2]), Some(0));
        assert_eq!(reusable(&[1, 2], &[1, 2, 3]), Some(2));
        assert_eq!(reusable(&[1, 2], &[1, 2]), Some(2));
        // A shorter prompt, and a diverging one, both invalidate: the
        // state holds tokens the prompt no longer has.
        assert_eq!(reusable(&[1, 2, 3], &[1, 2]), None);
        assert_eq!(reusable(&[1, 8], &[1, 2, 3]), None);
    }
}
