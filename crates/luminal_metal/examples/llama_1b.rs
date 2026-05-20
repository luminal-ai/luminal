use hf_hub::api::sync::Api;
use luminal::{
    dtype::DType,
    graph::{BuildSearchSpaceOptions, DimBucket, Graph},
    prelude::{F32Pow, GraphTensor, Runtime},
};
use luminal_metal::MetalRuntime;
use luminal_nn::{LayerNorm, gather_rows, scatter_rows};
use luminal_tracing::luminal_filter;
use rustc_hash::FxHashSet;
use std::{
    error::Error,
    io::Write,
    path::PathBuf,
    time::{Duration, Instant},
};
use tokenizers::Tokenizer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

const REPO_ID: &str = "unsloth/Llama-3.2-1B-Instruct";
const MAX_SEQ_LEN: usize = 2048;
const GEN_TOKENS: usize = 96;
const SEARCH_GRAPHS: usize = 100;
const SEARCH_MEMORY_MIB: usize = 1536;
const PROMPT: &str = "In one short paragraph, explain neural networks using the words layers, neurons, learning, and data.";

const LAYERS: usize = 16;
const HIDDEN: usize = 2048;
const INTERMEDIATE: usize = 8192;
const HEAD_DIM: usize = 64;
const N_HEADS: usize = 32;
const N_KV_HEADS: usize = 8;
const KV_GROUPS: usize = N_HEADS / N_KV_HEADS;
const KV_DIM: usize = N_KV_HEADS * HEAD_DIM;
const VOCAB_SIZE: usize = 128256;
const RMS_NORM_EPS: f32 = 1e-5;
const ROPE_THETA: f32 = 500_000.0;
const EOS_TOKEN: u32 = 128009;
const STOP_TOKEN: u32 = 128001;

fn prepare_hf_model() -> Result<PathBuf, Box<dyn Error>> {
    let repo = Api::new()?.model(REPO_ID.to_string());
    let tokenizer_path = repo.get("tokenizer.json")?;
    repo.get("model.safetensors")?;
    Ok(tokenizer_path.parent().unwrap().to_path_buf())
}

fn llama3_chat_prompt(user_prompt: &str) -> String {
    format!(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
}

#[derive(Default, Clone)]
struct StepProfile {
    total: Duration,
    execute: Duration,
    get_logits: Duration,
    cache_roundtrip: Duration,
}

fn avg_ms(duration: Duration, n: usize) -> f64 {
    if n == 0 {
        0.0
    } else {
        duration.as_secs_f64() * 1e3 / n as f64
    }
}

fn sample_greedy(logits_row: &[f32], seen: &FxHashSet<u32>, repetition_penalty: f32) -> u32 {
    let mut row = logits_row.to_vec();
    for &tok in seen {
        let logit = &mut row[tok as usize];
        if *logit > 0.0 {
            *logit /= repetition_penalty;
        } else {
            *logit *= repetition_penalty;
        }
    }
    row.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap()
        .0 as u32
}

fn causal_mask(q_pos: &[usize], context_len: usize) -> Vec<f32> {
    let mut mask = vec![-1e10f32; q_pos.len() * context_len];
    for (qi, &pos) in q_pos.iter().enumerate() {
        for ci in 0..context_len {
            if ci <= pos {
                mask[qi * context_len + ci] = 0.0;
            }
        }
    }
    mask
}

struct KVCache {
    k_caches: Vec<GraphTensor>,
    v_caches: Vec<GraphTensor>,
}

impl KVCache {
    fn new(cx: &mut Graph, num_slots: usize) -> Self {
        let mut k_caches = Vec::with_capacity(LAYERS);
        let mut v_caches = Vec::with_capacity(LAYERS);
        for l in 0..LAYERS {
            k_caches.push(
                cx.named_tensor(format!("kv_cache.{l}.k"), (num_slots, KV_DIM))
                    .persist(),
            );
            v_caches.push(
                cx.named_tensor(format!("kv_cache.{l}.v"), (num_slots, KV_DIM))
                    .persist(),
            );
        }
        Self { k_caches, v_caches }
    }
}

struct Llama {
    embedding: GraphTensor,
    layers: Vec<LlamaLayer>,
    lm_norm: LayerNorm,
}

impl Llama {
    fn init(cx: &mut Graph) -> Self {
        let mut layers = Vec::with_capacity(LAYERS);
        for l in 0..LAYERS {
            layers.push(LlamaLayer {
                up: cx
                    .named_tensor(
                        format!("model.layers.{l}.mlp.up_proj.weight"),
                        (INTERMEDIATE, HIDDEN),
                    )
                    .persist(),
                gate: cx
                    .named_tensor(
                        format!("model.layers.{l}.mlp.gate_proj.weight"),
                        (INTERMEDIATE, HIDDEN),
                    )
                    .persist(),
                down: cx
                    .named_tensor(
                        format!("model.layers.{l}.mlp.down_proj.weight"),
                        (HIDDEN, INTERMEDIATE),
                    )
                    .persist(),
                q_proj: cx
                    .named_tensor(
                        format!("model.layers.{l}.self_attn.q_proj.weight"),
                        (HIDDEN, HIDDEN),
                    )
                    .persist(),
                k_proj: cx
                    .named_tensor(
                        format!("model.layers.{l}.self_attn.k_proj.weight"),
                        (KV_DIM, HIDDEN),
                    )
                    .persist(),
                v_proj: cx
                    .named_tensor(
                        format!("model.layers.{l}.self_attn.v_proj.weight"),
                        (KV_DIM, HIDDEN),
                    )
                    .persist(),
                o_proj: cx
                    .named_tensor(
                        format!("model.layers.{l}.self_attn.o_proj.weight"),
                        (HIDDEN, HIDDEN),
                    )
                    .persist(),
                attn_rms: LayerNorm::new(
                    HIDDEN,
                    Some(&format!("model.layers.{l}.input_layernorm.weight")),
                    None,
                    false,
                    RMS_NORM_EPS,
                    cx,
                ),
                mlp_rms: LayerNorm::new(
                    HIDDEN,
                    Some(&format!("model.layers.{l}.post_attention_layernorm.weight")),
                    None,
                    false,
                    RMS_NORM_EPS,
                    cx,
                ),
            });
        }

        Self {
            embedding: cx
                .named_tensor("model.embed_tokens.weight", (VOCAB_SIZE, HIDDEN))
                .persist(),
            layers,
            lm_norm: LayerNorm::new(
                HIDDEN,
                Some("model.norm.weight"),
                None,
                false,
                RMS_NORM_EPS,
                cx,
            ),
        }
    }

    fn forward(
        &self,
        input: GraphTensor,
        q_pos: GraphTensor,
        scatter_idx: GraphTensor,
        gather_idx: GraphTensor,
        attn_mask: GraphTensor,
        kv_cache: &KVCache,
    ) -> (GraphTensor, Vec<(GraphTensor, GraphTensor)>) {
        let seq = input.dims1();
        let mut x = self.embedding.gather(
            (input * HIDDEN).expand_dim(1, HIDDEN)
                + input.graph().arange(HIDDEN).expand_dim(0, seq),
        );
        let mut cache_outputs = Vec::with_capacity(LAYERS);
        for (i, layer) in self.layers.iter().enumerate() {
            let (x_new, k_out, v_out) = layer.forward(
                x,
                q_pos,
                scatter_idx,
                gather_idx,
                attn_mask,
                kv_cache.k_caches[i],
                kv_cache.v_caches[i],
            );
            x = x_new;
            cache_outputs.push((k_out, v_out));
        }

        let logits = self.lm_norm.forward(x).matmul(self.embedding.t());
        (logits, cache_outputs)
    }
}

struct LlamaLayer {
    up: GraphTensor,
    gate: GraphTensor,
    down: GraphTensor,
    q_proj: GraphTensor,
    k_proj: GraphTensor,
    v_proj: GraphTensor,
    o_proj: GraphTensor,
    attn_rms: LayerNorm,
    mlp_rms: LayerNorm,
}

fn llama_rotary_embeddings(mut input: GraphTensor, pos_ids: GraphTensor) -> GraphTensor {
    input = input.split_dims(1, HEAD_DIM).transpose(0, 1);

    let freqs = input
        .graph()
        .arange_options(0, HEAD_DIM, 2)
        .cast(DType::F32)
        / HEAD_DIM as f32;
    let inv_freqs = ROPE_THETA.pow(freqs).reciprocal();
    let emb = pos_ids
        .cast(DType::F32)
        .expand_dim(1, 1)
        .matmul(inv_freqs.expand_dim(0, 1));

    let x0 = input.slice((.., .., ..HEAD_DIM / 2));
    let x1 = input.slice((.., .., HEAD_DIM / 2..));

    let cos = emb.cos().expand_dim(0, x0.dims()[0]);
    let sin = emb.sin().expand_dim(0, x0.dims()[0]);
    let x0_out = x0 * cos - x1 * sin;
    let x1_out = x1 * cos + x0 * sin;

    x0_out
        .concat_along(x1_out, 2)
        .transpose(0, 1)
        .merge_dims(1, 2)
}

#[allow(clippy::too_many_arguments)]
fn attention(
    q_rope: GraphTensor,
    k_rope: GraphTensor,
    v: GraphTensor,
    k_cache: GraphTensor,
    v_cache: GraphTensor,
    scatter_idx: GraphTensor,
    gather_idx: GraphTensor,
    attn_mask: GraphTensor,
) -> (GraphTensor, GraphTensor, GraphTensor) {
    let k_cache_out = scatter_rows(k_rope, scatter_idx, k_cache, KV_DIM);
    let v_cache_out = scatter_rows(v, scatter_idx, v_cache, KV_DIM);

    let k = gather_rows(k_cache_out, gather_idx, KV_DIM);
    let v_ctx = gather_rows(v_cache_out, gather_idx, KV_DIM);

    let q = (q_rope * 1.0).split_dims(1, HEAD_DIM).transpose(0, 1);
    let k = k.split_dims(1, HEAD_DIM).permute((1, 2, 0));
    let v_ctx = v_ctx.split_dims(1, HEAD_DIM).transpose(0, 1);

    let k = k.expand_dim(1, KV_GROUPS).merge_dims(0, 1) * 1.0;
    let v_ctx = v_ctx.expand_dim(1, KV_GROUPS).merge_dims(0, 1) * 1.0;

    let scores = q.matmul(k) / (HEAD_DIM as f32).sqrt();
    let masked_scores = scores + attn_mask.expand_dim(0, N_HEADS);
    let weights = masked_scores.softmax(2);
    let out = weights.matmul(v_ctx);
    let attn_out = out.transpose(0, 1).merge_dims(1, 2);

    (attn_out, k_cache_out, v_cache_out)
}

impl LlamaLayer {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        mut x: GraphTensor,
        q_pos: GraphTensor,
        scatter_idx: GraphTensor,
        gather_idx: GraphTensor,
        attn_mask: GraphTensor,
        k_cache: GraphTensor,
        v_cache: GraphTensor,
    ) -> (GraphTensor, GraphTensor, GraphTensor) {
        let x_attn = self.attn_rms.forward(x);
        let q = x_attn.matmul(self.q_proj.t());
        let k = x_attn.matmul(self.k_proj.t());
        let v = x_attn.matmul(self.v_proj.t());

        let q_rope = llama_rotary_embeddings(q, q_pos);
        let k_rope = llama_rotary_embeddings(k, q_pos);
        let (attn_out, k_cache_out, v_cache_out) = attention(
            q_rope,
            k_rope,
            v,
            k_cache,
            v_cache,
            scatter_idx,
            gather_idx,
            attn_mask,
        );
        x += attn_out.matmul(self.o_proj.t());

        let x_mlp = self.mlp_rms.forward(x);
        let mlp_out =
            (x_mlp.matmul(self.gate.t()).swish() * x_mlp.matmul(self.up.t())).matmul(self.down.t());
        (x + mlp_out, k_cache_out, v_cache_out)
    }
}

#[allow(clippy::too_many_arguments)]
fn run_model_step(
    cx: &mut Graph,
    runtime: &mut MetalRuntime,
    input: GraphTensor,
    q_pos_t: GraphTensor,
    scatter_idx_t: GraphTensor,
    gather_idx_t: GraphTensor,
    attn_mask_t: GraphTensor,
    logits: GraphTensor,
    kv_cache: &KVCache,
    cache_outputs: &[(GraphTensor, GraphTensor)],
    tokens: &[u32],
    q_pos: &[i32],
    scatter_idx: &[i32],
    gather_idx: &[i32],
    attn_mask: &[f32],
) -> (Vec<f32>, StepProfile) {
    let start = Instant::now();
    cx.set_dim('s', tokens.len());
    cx.set_dim('c', gather_idx.len());

    runtime.set_data(input, tokens.iter().map(|t| *t as i32).collect::<Vec<_>>());
    runtime.set_data(q_pos_t, q_pos.to_vec());
    runtime.set_data(scatter_idx_t, scatter_idx.to_vec());
    runtime.set_data(gather_idx_t, gather_idx.to_vec());
    runtime.set_data(attn_mask_t, attn_mask.to_vec());
    runtime.allocate_intermediate_buffers(&cx.dyn_map);

    let execute_start = Instant::now();
    runtime.execute(&cx.dyn_map);
    let execute = execute_start.elapsed();

    let logits_start = Instant::now();
    let logits_data = runtime.get_f32(logits);
    let get_logits = logits_start.elapsed();

    let cache_start = Instant::now();
    for (layer_idx, (k_out, v_out)) in cache_outputs.iter().enumerate() {
        let k_buf = runtime.remove_buffer(*k_out);
        let v_buf = runtime.remove_buffer(*v_out);
        runtime.set_buffer(kv_cache.k_caches[layer_idx], k_buf);
        runtime.set_buffer(kv_cache.v_caches[layer_idx], v_buf);
    }
    let cache_roundtrip = cache_start.elapsed();

    (
        logits_data,
        StepProfile {
            total: start.elapsed(),
            execute,
            get_logits,
            cache_roundtrip,
        },
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    let _ = tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(luminal_filter())
        .try_init();

    let model_dir = prepare_hf_model()?;
    println!("Using model directory: {}", model_dir.display());

    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|err| err as Box<dyn Error>)?;
    let prompt_tokens = tokenizer
        .encode(llama3_chat_prompt(PROMPT), false)
        .map_err(|err| err as Box<dyn Error>)?
        .get_ids()
        .to_vec();

    let mut cx = Graph::default();
    let input = cx.named_tensor("input", 's').as_dtype(DType::Int);
    let q_pos_t = cx.named_tensor("q_pos", 's').as_dtype(DType::Int);
    let scatter_idx_t = cx.named_tensor("scatter_idx", 's').as_dtype(DType::Int);
    let gather_idx_t = cx.named_tensor("gather_idx", 'c').as_dtype(DType::Int);
    let attn_mask_t = cx.named_tensor("attn_mask", ('s', 'c'));
    let kv_cache = KVCache::new(&mut cx, MAX_SEQ_LEN);
    let (logits, cache_outputs) = Llama::init(&mut cx).forward(
        input,
        q_pos_t,
        scatter_idx_t,
        gather_idx_t,
        attn_mask_t,
        &kv_cache,
    );
    let logits = logits.output();
    for (k_out, v_out) in &cache_outputs {
        k_out.output();
        v_out.output();
    }

    cx.set_dim('s', 1);
    cx.set_dim('c', 1);

    println!("Building E-Graph...");
    let egraph_start = Instant::now();
    cx.build_search_space_with_options::<MetalRuntime>(
        BuildSearchSpaceOptions::new().max_memory_mib(SEARCH_MEMORY_MIB),
    );
    println!(
        "  E-Graph build: {:.2} s",
        egraph_start.elapsed().as_secs_f64()
    );

    println!("Loading weights...");
    let load_start = Instant::now();
    let mut runtime = MetalRuntime::initialize(());
    runtime.load_safetensors(&cx, model_dir.join("model.safetensors").to_str().unwrap());
    println!("  Weight load: {:.2} s", load_start.elapsed().as_secs_f64());

    let cache_bytes = MAX_SEQ_LEN * KV_DIM * std::mem::size_of::<f32>();
    for i in 0..LAYERS {
        runtime.set_zeros(kv_cache.k_caches[i], cache_bytes);
        runtime.set_zeros(kv_cache.v_caches[i], cache_bytes);
    }

    println!("Compiling...");
    let compile_start = Instant::now();
    let max_prefill = (prompt_tokens.len() + 16)
        .next_power_of_two()
        .min(MAX_SEQ_LEN);
    let max_context = (prompt_tokens.len() + GEN_TOKENS + 1)
        .next_power_of_two()
        .min(MAX_SEQ_LEN);
    let search_s = 16.min(max_prefill).max(2);
    let search_c = 16.min(max_context).max(2);
    cx.set_dim_buckets(
        's',
        &[
            DimBucket::new(1, 1),
            DimBucket::new(2, max_prefill).representative(search_s),
        ],
    );
    cx.set_dim_buckets(
        'c',
        &[
            DimBucket::new(1, 1),
            DimBucket::new(2, max_context).representative(search_c),
        ],
    );
    cx.set_dim('s', search_s);
    cx.set_dim('c', search_c);
    runtime.set_data(input, vec![1; search_s]);
    runtime.set_data(q_pos_t, (0..search_s as i32).collect::<Vec<_>>());
    runtime.set_data(scatter_idx_t, (0..search_s as i32).collect::<Vec<_>>());
    runtime.set_data(gather_idx_t, (0..search_c as i32).collect::<Vec<_>>());
    runtime.set_data(attn_mask_t, vec![0.0f32; search_s * search_c]);
    runtime = cx.search(runtime, SEARCH_GRAPHS);
    println!(
        "  Search/compile: {:.2} s",
        compile_start.elapsed().as_secs_f64()
    );

    for i in 0..LAYERS {
        runtime.set_zeros(kv_cache.k_caches[i], cache_bytes);
        runtime.set_zeros(kv_cache.v_caches[i], cache_bytes);
    }

    let prompt_len = prompt_tokens.len();
    let mut context_len = 0usize;
    let mut profiles = Vec::new();
    let mut seen_tokens = FxHashSet::default();
    let repetition_penalty = 1.05;

    println!(
        "Prompt: {} tokens, generating up to {} tokens",
        prompt_len, GEN_TOKENS
    );

    let mut generated = 0usize;
    let mut next_token = None;
    if GEN_TOKENS > 0 && prompt_len > 0 {
        let positions: Vec<usize> = (0..prompt_len).collect();
        let q_pos: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
        let mask = causal_mask(&positions, prompt_len);
        let (logits_data, profile) = run_model_step(
            &mut cx,
            &mut runtime,
            input,
            q_pos_t,
            scatter_idx_t,
            gather_idx_t,
            attn_mask_t,
            logits,
            &kv_cache,
            &cache_outputs,
            &prompt_tokens,
            &q_pos,
            &q_pos,
            &q_pos,
            &mask,
        );
        context_len = prompt_len;

        let token = sample_greedy(
            &logits_data[logits_data.len() - VOCAB_SIZE..],
            &seen_tokens,
            repetition_penalty,
        );
        seen_tokens.insert(token);
        next_token = Some(token);
        generated = 1;
        profiles.push(profile);

        if token != EOS_TOKEN && token != STOP_TOKEN {
            print!(
                "{}",
                tokenizer
                    .decode(&[token], true)
                    .map_err(|err| err as Box<dyn Error>)?
            );
            std::io::stdout().flush()?;
        }
    }

    while generated < GEN_TOKENS {
        let current_token = match next_token {
            Some(token) if token != EOS_TOKEN && token != STOP_TOKEN => token,
            _ => break,
        };
        let gather_idx = (0..=context_len as i32).collect::<Vec<_>>();
        let mask = causal_mask(&[context_len], context_len + 1);
        let (logits_data, profile) = run_model_step(
            &mut cx,
            &mut runtime,
            input,
            q_pos_t,
            scatter_idx_t,
            gather_idx_t,
            attn_mask_t,
            logits,
            &kv_cache,
            &cache_outputs,
            &[current_token],
            &[context_len as i32],
            &[context_len as i32],
            &gather_idx,
            &mask,
        );
        context_len += 1;

        let token = sample_greedy(
            &logits_data[logits_data.len() - VOCAB_SIZE..],
            &seen_tokens,
            repetition_penalty,
        );
        seen_tokens.insert(token);
        next_token = Some(token);
        generated += 1;
        profiles.push(profile);

        if token == EOS_TOKEN || token == STOP_TOKEN {
            break;
        }
        print!(
            "{}",
            tokenizer
                .decode(&[token], true)
                .map_err(|err| err as Box<dyn Error>)?
        );
        std::io::stdout().flush()?;
    }
    println!();

    let ttft = profiles.first().map(|p| p.total).unwrap_or_default();
    let decode_steps = profiles.len().saturating_sub(1);
    let decode_total: Duration = profiles.iter().skip(1).map(|p| p.total).sum();
    println!("  TTFT: {:.2} ms", ttft.as_secs_f64() * 1e3);
    println!("  TPOT: {:.2} ms", avg_ms(decode_total, decode_steps));

    let execute_total: Duration = profiles.iter().map(|p| p.execute).sum();
    let logits_total: Duration = profiles.iter().map(|p| p.get_logits).sum();
    let cache_total: Duration = profiles.iter().map(|p| p.cache_roundtrip).sum();
    println!(
        "  Profile: n={}, exec={:.2} ms, logits={:.2} ms, cache={:.2} ms",
        profiles.len(),
        avg_ms(execute_total, profiles.len()),
        avg_ms(logits_total, profiles.len()),
        avg_ms(cache_total, profiles.len()),
    );

    Ok(())
}
