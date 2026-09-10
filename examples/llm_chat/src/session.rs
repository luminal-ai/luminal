use crate::{backend::Backend, graph::LlmGraph, sampling::Sampler};
use anyhow::{Result, ensure};
use std::collections::BTreeSet;

pub struct Session<B> {
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
        // Token prefixes, not rendered-string prefixes, determine cache validity.
        if !prompt.starts_with(&self.cached) {
            self.reset()?;
        }
        let consumed = self.cached.len();
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
    use crate::{Inputs, graph::ModelConfig};
    use model_zoo::llama3::Llama3Dims;
    struct FakeBackend {
        calls: Vec<(usize, usize)>,
        resets: usize,
        vocab: usize,
        fail: bool,
    }
    impl Backend for FakeBackend {
        fn step(&mut self, _: Inputs, q: usize, c: usize) -> Result<Vec<f32>> {
            self.calls.push((q, c));
            ensure!(!self.fail, "injected execution failure");
            let mut logits = vec![0.; self.vocab];
            logits[4] = 1.;
            Ok(logits)
        }
        fn reset(&mut self) -> Result<()> {
            self.resets += 1;
            self.fail = false;
            Ok(())
        }
    }
    fn session() -> Session<FakeBackend> {
        let graph = LlmGraph::build(ModelConfig::Llama3(Llama3Dims::tiny()), 10, 2).unwrap();
        let backend = FakeBackend {
            calls: vec![],
            resets: 0,
            vocab: graph.vocab,
            fail: false,
        };
        Session::new(graph, backend)
    }
    #[test]
    fn chunked_prefill_decode_history_reuse_and_reset_share_one_path() {
        let mut s = session();
        let mut sampler = Sampler::new(0., 1., 0).unwrap();
        let stops = [4].into();
        let generated = s
            .generate(&[1, 2, 3], 5, &stops, &mut sampler, |_| {
                panic!("EOS must not be emitted")
            })
            .unwrap();
        assert_eq!(generated, vec![4]);
        assert_eq!(s.backend.calls, vec![(2, 2), (1, 3), (1, 4)]);
        s.generate(&[1, 2, 3, 4, 5], 5, &stops, &mut sampler, |_| Ok(()))
            .unwrap();
        assert_eq!(&s.backend.calls[3..], &[(1, 5), (1, 6)]);
        assert_eq!(s.backend.resets, 0);
        s.generate(&[1, 8], 1, &stops, &mut sampler, |_| Ok(()))
            .unwrap();
        assert_eq!(s.backend.resets, 1);
        assert_eq!(s.cached, vec![1, 8, 4]);
        assert!(
            s.generate(&[1; 10], 1, &stops, &mut sampler, |_| Ok(()))
                .is_err()
        );
    }
    #[test]
    fn failed_execution_invalidates_cache_before_retry() {
        let mut s = session();
        s.backend.fail = true;
        let mut sampler = Sampler::new(0., 1., 0).unwrap();
        assert!(
            s.generate(&[1], 1, &[4].into(), &mut sampler, |_| Ok(()))
                .is_err()
        );
        assert!(s.cached.is_empty());
        assert_eq!(s.backend.resets, 1);
        assert_eq!(
            s.generate(&[1], 1, &[4].into(), &mut sampler, |_| Ok(()))
                .unwrap(),
            vec![4]
        );
    }
}
