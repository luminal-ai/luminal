pub mod hf;
pub mod model;

use hf::prepare_hf_model;
pub use luminal::prelude::Runtime;
use luminal::prelude::*;
use luminal_tracing::luminal_filter;
use model::*;
use rustc_hash::FxHashSet;
use std::{error::Error, io::Write, time::Duration};
use tokenizers::Tokenizer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

const EOS_TOKEN: u32 = 151645; // <|im_end|>
const STOP_TOKEN: u32 = 151643; // <|endoftext|>

pub struct QwenRunConfig {
    pub repo_id: String,
    pub max_seq_len: usize,
    pub gen_tokens: usize,
    pub search_graphs: usize,
    pub prompt: String,
    pub repetition_penalty: f32,
}

fn qwen3_chat_prompt(user_prompt: &str) -> String {
    format!(
        "<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
}

impl Default for QwenRunConfig {
    fn default() -> Self {
        Self {
            repo_id: "Qwen/Qwen3-4B".to_string(),
            max_seq_len: 4096,
            gen_tokens: 500,
            search_graphs: 500,
            prompt: "Explain what a neural network is in a paragraph.".to_string(),
            repetition_penalty: 1.05,
        }
    }
}

pub trait QwenRuntime: Runtime<ExecReturn = ()> {
    type Buffer;

    fn load_safetensors(&mut self, cx: &Graph, file_path: &str);
    fn set_i32_data(&mut self, id: NodeIndex, data: Vec<i32>);
    fn set_zeros(&mut self, id: NodeIndex, num_bytes: usize);
    fn remove_buffer(&mut self, id: NodeIndex) -> Self::Buffer;
    fn set_buffer(&mut self, id: NodeIndex, buffer: Self::Buffer);
    fn get_f32(&self, id: NodeIndex) -> Vec<f32>;

    fn prepare_execute(&mut self, _dyn_map: &FxHashMap<char, usize>) {}
}

#[cfg(feature = "cuda")]
impl QwenRuntime for luminal_cuda_lite::runtime::CudaRuntime {
    type Buffer = luminal_cuda_lite::cudarc::driver::CudaSlice<u8>;

    fn load_safetensors(&mut self, cx: &Graph, file_path: &str) {
        luminal_cuda_lite::runtime::CudaRuntime::load_safetensors(self, cx, file_path);
    }

    fn set_i32_data(&mut self, id: NodeIndex, data: Vec<i32>) {
        luminal_cuda_lite::runtime::CudaRuntime::set_data(self, id, data);
    }

    fn set_zeros(&mut self, id: NodeIndex, num_bytes: usize) {
        luminal_cuda_lite::runtime::CudaRuntime::set_zeros(self, id, num_bytes);
    }

    fn remove_buffer(&mut self, id: NodeIndex) -> Self::Buffer {
        luminal_cuda_lite::runtime::CudaRuntime::remove_buffer(self, id)
    }

    fn set_buffer(&mut self, id: NodeIndex, buffer: Self::Buffer) {
        luminal_cuda_lite::runtime::CudaRuntime::set_buffer(self, id, buffer);
    }

    fn get_f32(&self, id: NodeIndex) -> Vec<f32> {
        luminal_cuda_lite::runtime::CudaRuntime::get_f32(self, id)
    }
}

#[cfg(feature = "metal")]
impl QwenRuntime for luminal_metal::MetalRuntime {
    type Buffer = luminal_metal::Buffer;

    fn load_safetensors(&mut self, cx: &Graph, file_path: &str) {
        luminal_metal::MetalRuntime::load_safetensors(self, cx, file_path);
    }

    fn set_i32_data(&mut self, id: NodeIndex, data: Vec<i32>) {
        luminal_metal::MetalRuntime::set_data(self, id, data);
    }

    fn set_zeros(&mut self, id: NodeIndex, num_bytes: usize) {
        luminal_metal::MetalRuntime::set_zeros(self, id, num_bytes);
    }

    fn remove_buffer(&mut self, id: NodeIndex) -> Self::Buffer {
        luminal_metal::MetalRuntime::remove_buffer(self, id)
    }

    fn set_buffer(&mut self, id: NodeIndex, buffer: Self::Buffer) {
        luminal_metal::MetalRuntime::set_buffer(self, id, buffer);
    }

    fn get_f32(&self, id: NodeIndex) -> Vec<f32> {
        luminal_metal::MetalRuntime::get_f32(self, id)
    }

    fn prepare_execute(&mut self, dyn_map: &FxHashMap<char, usize>) {
        luminal_metal::MetalRuntime::allocate_intermediate_buffers(self, dyn_map);
    }
}

pub fn run_qwen<R>(mut runtime: R, config: QwenRunConfig) -> Result<(), Box<dyn Error>>
where
    R: QwenRuntime + 'static,
{
    let _ = tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(luminal_filter())
        .try_init();

    let model_dir = prepare_hf_model(&config.repo_id)?;
    println!("Using model directory: {}", model_dir.display());

    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|err| err as Box<dyn Error>)?;
    let prompt = qwen3_chat_prompt(&config.prompt);
    let prompt_tokens = tokenizer
        .encode(prompt.as_str(), false)
        .map_err(|err| err as Box<dyn Error>)?
        .get_ids()
        .to_vec();

    let mut cx = Graph::default();
    let input = cx.named_tensor("input", 's').as_dtype(DType::Int);
    let token_ids = cx.named_tensor("token_ids", 's').as_dtype(DType::Int);
    let kv_cache = KVCache::new(&mut cx, config.max_seq_len);
    let (logits, cache_outputs) = Qwen::init(&mut cx).forward(input, token_ids, &kv_cache);
    let logits = logits.output();
    for (k_out, v_out) in &cache_outputs {
        k_out.output();
        v_out.output();
    }

    println!("Building E-Graph...");
    cx.build_search_space::<R>();

    println!("Loading weights...");
    let weights_path = model_dir.join("model_combined.safetensors");
    runtime.load_safetensors(&cx, weights_path.to_str().unwrap());

    let cache_bytes = N_KV_HEADS * config.max_seq_len * HEAD_DIM * std::mem::size_of::<f32>();
    for i in 0..LAYERS {
        runtime.set_zeros(kv_cache.k_caches[i].id, cache_bytes);
        runtime.set_zeros(kv_cache.v_caches[i].id, cache_bytes);
    }

    println!("Compiling...");
    let max_prefill = (prompt_tokens.len() + 16)
        .next_power_of_two()
        .min(config.max_seq_len);
    let search_s = 16.min(max_prefill).max(2);
    cx.set_dim_buckets(
        's',
        &[
            DimBucket::new(1, 1),
            DimBucket::new(2, max_prefill).representative(search_s),
        ],
    );
    cx.set_dim('s', search_s);
    cx.set_dim('p', 0);
    runtime.set_i32_data(input.id, vec![1; search_s]);
    runtime.set_i32_data(token_ids.id, (0..search_s as i32).collect::<Vec<_>>());
    runtime = cx.search(runtime, config.search_graphs);

    for i in 0..LAYERS {
        runtime.set_zeros(kv_cache.k_caches[i].id, cache_bytes);
        runtime.set_zeros(kv_cache.v_caches[i].id, cache_bytes);
    }

    let prompt_len = prompt_tokens.len();
    let mut prev_seq = 0usize;
    let mut fwd_durations = vec![];
    let mut seen_tokens = FxHashSet::default();

    println!(
        "Prompt: {} tokens, generating up to {} tokens",
        prompt_len, config.gen_tokens
    );

    let mut generated = 0usize;
    let mut sentence = Vec::new();

    if config.gen_tokens > 0 && prompt_len > 0 {
        let start = std::time::Instant::now();

        cx.set_dim('s', prompt_len);
        cx.set_dim('p', 0);

        runtime.set_i32_data(
            input.id,
            prompt_tokens.iter().map(|t| *t as i32).collect::<Vec<_>>(),
        );
        runtime.set_i32_data(token_ids.id, (0..prompt_len as i32).collect::<Vec<_>>());
        runtime.prepare_execute(&cx.dyn_map);

        runtime.execute(&cx.dyn_map);
        let logits_data = runtime.get_f32(logits.id);

        for (layer_idx, (k_out, v_out)) in cache_outputs.iter().enumerate() {
            let k_buf = runtime.remove_buffer(k_out.id);
            let v_buf = runtime.remove_buffer(v_out.id);
            runtime.set_buffer(kv_cache.k_caches[layer_idx].id, k_buf);
            runtime.set_buffer(kv_cache.v_caches[layer_idx].id, v_buf);
        }

        prev_seq = prompt_len;
        fwd_durations.push(start.elapsed());

        let row_start = (prompt_len - 1) * VOCAB_SIZE;
        let mut last_row = logits_data[row_start..row_start + VOCAB_SIZE].to_vec();
        for &tok in &seen_tokens {
            let logit = &mut last_row[tok as usize];
            if *logit > 0.0 {
                *logit /= config.repetition_penalty;
            } else {
                *logit *= config.repetition_penalty;
            }
        }
        let next_token = last_row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0 as u32;
        sentence = vec![next_token];
        seen_tokens.insert(next_token);
        generated = 1;

        if next_token != EOS_TOKEN && next_token != STOP_TOKEN {
            let decoded = tokenizer
                .decode(&[next_token], true)
                .map_err(|err| err as Box<dyn Error>)?;
            print!("{}", decoded);
            std::io::stdout().flush()?;
        }
    }

    while generated < config.gen_tokens && !sentence.is_empty() {
        let start = std::time::Instant::now();
        let seq_len = sentence.len();
        let current_token = sentence[0];

        if current_token == EOS_TOKEN || current_token == STOP_TOKEN {
            break;
        }

        cx.set_dim('s', seq_len);
        cx.set_dim('p', prev_seq);

        runtime.set_i32_data(
            input.id,
            sentence.iter().map(|t| *t as i32).collect::<Vec<_>>(),
        );
        runtime.set_i32_data(
            token_ids.id,
            (prev_seq as i32..(seq_len + prev_seq) as i32).collect::<Vec<_>>(),
        );
        runtime.prepare_execute(&cx.dyn_map);

        runtime.execute(&cx.dyn_map);
        let logits_data = runtime.get_f32(logits.id);

        for (layer_idx, (k_out, v_out)) in cache_outputs.iter().enumerate() {
            let k_buf = runtime.remove_buffer(k_out.id);
            let v_buf = runtime.remove_buffer(v_out.id);
            runtime.set_buffer(kv_cache.k_caches[layer_idx].id, k_buf);
            runtime.set_buffer(kv_cache.v_caches[layer_idx].id, v_buf);
        }

        prev_seq += seq_len;
        fwd_durations.push(start.elapsed());

        let mut last_row = logits_data[logits_data.len() - VOCAB_SIZE..].to_vec();
        for &tok in &seen_tokens {
            let logit = &mut last_row[tok as usize];
            if *logit > 0.0 {
                *logit /= config.repetition_penalty;
            } else {
                *logit *= config.repetition_penalty;
            }
        }
        let next_token = last_row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap()
            .0 as u32;
        sentence = vec![next_token];
        seen_tokens.insert(next_token);
        generated += 1;

        if next_token == EOS_TOKEN || next_token == STOP_TOKEN {
            break;
        }

        let decoded = tokenizer
            .decode(&[next_token], true)
            .map_err(|err| err as Box<dyn Error>)?;
        print!("{}", decoded);
        std::io::stdout().flush()?;
    }
    println!();

    let decode_durations: Vec<_> = fwd_durations.iter().skip(1).collect();
    if decode_durations.len() > 2 {
        println!(
            "  TTFT: {:.2} ms",
            fwd_durations[..1].iter().sum::<Duration>().as_secs_f64() * 1e3
        );
        println!(
            "  TPOT: {:.2} ms",
            (decode_durations.iter().skip(1).copied().sum::<Duration>()
                / (decode_durations.len() - 1) as u32)
                .as_secs_f64()
                * 1_000.
        );
    }

    Ok(())
}
