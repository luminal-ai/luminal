use crate::{backend::Backend, graph::LlmGraph, sampling::Sampler};
use anyhow::{Result, ensure};
use std::collections::BTreeSet;

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
}
impl<B: Backend> Session<B> {
    pub fn new(graph: LlmGraph, backend: B) -> Self {
        Self {
            graph,
            backend,
            cached: vec![],
            last_logits: None,
        }
    }
    /// The token history the KV state currently represents.
    pub fn cached(&self) -> &[u32] {
        &self.cached
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
        let consumed = match reusable(&self.cached, prompt) {
            Some(consumed) => consumed,
            None => {
                self.reset()?;
                0
            }
        };
        self.ingest(&prompt[consumed..])?;
        let limit = max_new_tokens.min(self.graph.capacity - self.cached.len());
        let mut generated = vec![];
        for _ in 0..limit {
            let token = sampler.sample(self.last_logits.as_ref().expect("nonempty prefill"))?;
            // Consume even the terminal token so the next turn's prefix check
            // reflects the complete token history represented by the KV state.
            self.ingest(&[token])?;
            generated.push(token);
            if stops.contains(&token) {
                break;
            }
            emit(token)?;
        }
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
