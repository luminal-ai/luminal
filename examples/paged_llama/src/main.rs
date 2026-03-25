mod hf;
mod model;

use hf::prepare_hf_model;
use luminal::prelude::*;
use luminal_cuda_lite::{cudarc::driver::CudaContext, runtime::CudaRuntime};
use luminal_tracing::*;
use model::*;
use rustc_hash::FxHashSet;
use std::{io::Write, time::Duration};
use tokenizers::Tokenizer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

const REPO_ID: &str = "NousResearch/Meta-Llama-3-8B-Instruct";

struct PageTable {
    tables: Vec<Vec<usize>>,
    next_free_slot: usize,
}

impl PageTable {
    fn new() -> Self {
        Self {
            tables: vec![],
            next_free_slot: 0,
        }
    }
    fn new_sequence(&mut self) -> usize {
        let id = self.tables.len();
        self.tables.push(vec![]);
        id
    }
    fn allocate(&mut self, seq_id: usize, n: usize) {
        let slots: Vec<usize> = (self.next_free_slot..self.next_free_slot + n).collect();
        self.next_free_slot += n;
        self.tables[seq_id].extend_from_slice(&slots);
    }
    fn context_slots(&self, seq_id: usize) -> &[usize] {
        &self.tables[seq_id]
    }
    fn context_len(&self, seq_id: usize) -> usize {
        self.tables[seq_id].len()
    }
}

// ─── Batch Builder ───

fn build_batch(
    entries: &[(usize, Vec<usize>)],
    page_table: &PageTable,
) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<f32>) {
    let total_s: usize = entries.iter().map(|(_, pos)| pos.len()).sum();
    let mut gather_idx: Vec<i32> = vec![];
    let mut ctx_ranges: Vec<(usize, usize)> = vec![];
    for (seq_id, _) in entries {
        let start = gather_idx.len();
        let slots = page_table.context_slots(*seq_id);
        gather_idx.extend(slots.iter().map(|&s| s as i32));
        ctx_ranges.push((start, slots.len()));
    }
    let total_c = gather_idx.len();
    let mut scatter_idx: Vec<i32> = vec![];
    let mut q_pos: Vec<i32> = vec![];
    for (seq_id, positions) in entries {
        let ctx_len = page_table.context_len(*seq_id);
        let n_new = positions.len();
        let slots = page_table.context_slots(*seq_id);
        scatter_idx.extend(slots[ctx_len - n_new..].iter().map(|&s| s as i32));
        q_pos.extend(positions.iter().map(|&p| p as i32));
    }
    let mut mask = vec![-1e10f32; total_s * total_c];
    let mut q_offset = 0;
    for (entry_idx, (_, positions)) in entries.iter().enumerate() {
        let (ctx_start, ctx_len) = ctx_ranges[entry_idx];
        for (qi, &abs_pos) in positions.iter().enumerate() {
            for ci in 0..ctx_len {
                if ci <= abs_pos {
                    mask[(q_offset + qi) * total_c + (ctx_start + ci)] = 0.0;
                }
            }
        }
        q_offset += positions.len();
    }
    (scatter_idx, gather_idx, q_pos, mask)
}

// ─── Sampling ───

fn sample_greedy(logits_row: &[f32], seen: &FxHashSet<u32>, penalty: f32) -> u32 {
    let mut row = logits_row.to_vec();
    for &tok in seen {
        let logit = &mut row[tok as usize];
        if *logit > 0.0 {
            *logit /= penalty;
        } else {
            *logit *= penalty;
        }
    }
    row.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap()
        .0 as u32
}

fn logits_row(all_logits: &[f32], row_idx: usize) -> &[f32] {
    &all_logits[row_idx * VOCAB_SIZE..(row_idx + 1) * VOCAB_SIZE]
}

fn tick(
    cx: &mut Graph,
    runtime: &mut CudaRuntime,
    s: usize,
    c: usize,
    logits: GraphTensor,
    kv_cache: &PagedKVCache,
    cache_outputs: &[(GraphTensor, GraphTensor)],
) -> Vec<f32> {
    cx.set_dim('s', s);
    cx.set_dim('c', c);
    runtime.execute(&cx.dyn_map);
    let all = runtime.get_f32(logits);

    // Round-trip KV cache: move output buffers back to input tensors
    for (i, (k_out, v_out)) in cache_outputs.iter().enumerate() {
        let k_buf = runtime.remove_buffer(*k_out);
        let v_buf = runtime.remove_buffer(*v_out);
        runtime.set_buffer(kv_cache.k_caches[i], k_buf);
        runtime.set_buffer(kv_cache.v_caches[i], v_buf);
    }

    all[..s * VOCAB_SIZE].to_vec()
}

const EOS_TOKEN: u32 = 128009;
const STOP_TOKEN: u32 = 128001;

fn main() {
    let num_slots = 8192;
    let search_graphs = 100;
    let gen_tokens = 30;
    let prompt_a = "Explain what a neural network is in a paragraph.";
    let prompt_b = "What is the capital of France?";

    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(luminal_filter())
        .init();

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let model_dir = prepare_hf_model(REPO_ID).expect("Failed to prepare model");
    println!("Using model directory: {}", model_dir.display());

    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();

    let encode = |prompt: &str| -> Vec<u32> {
        let chat = format!(
            "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
        tokenizer
            .encode(chat.as_str(), true)
            .unwrap()
            .get_ids()
            .to_vec()
    };

    let tokens_a = encode(prompt_a);
    let tokens_b = encode(prompt_b);
    println!("Prompt A: {} tokens", tokens_a.len());
    println!("Prompt B: {} tokens", tokens_b.len());

    // ─── Build Graph ───
    let mut cx = Graph::default();
    let input = cx.named_tensor("input", 's').as_dtype(DType::Int);
    let q_pos_t = cx.named_tensor("q_pos", 's').as_dtype(DType::Int);
    let scatter_idx_t = cx.named_tensor("scatter_idx", 's').as_dtype(DType::Int);
    let gather_idx_t = cx.named_tensor("gather_idx", 'c').as_dtype(DType::Int);
    let attn_mask_t = cx.named_tensor("attn_mask", ('s', 'c'));

    let kv_cache = PagedKVCache::new(&mut cx, num_slots);
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

    println!("Building E-Graph...");
    cx.build_search_space::<CudaRuntime>();

    println!("Loading weights...");
    let mut runtime = CudaRuntime::initialize(stream);
    let weights_path = model_dir.join("model_combined.safetensors");
    runtime.load_safetensors(&cx, weights_path.to_str().unwrap());

    let cache_bytes = num_slots * KV_DIM * std::mem::size_of::<f32>();
    for i in 0..LAYERS {
        runtime.set_zeros(kv_cache.k_caches[i], cache_bytes);
        runtime.set_zeros(kv_cache.v_caches[i], cache_bytes);
    }

    println!("Compiling...");
    // Bucket s=1 (decode) vs s>1 (prefill/mixed). Each bucket gets its own
    // optimized compilation — decode can select warp-parallel kernels while
    // prefill can select tiled matmul / cuBLAS.
    let max_prefill = (tokens_a.len().max(tokens_b.len()) + 16).next_power_of_two();
    cx.set_dim_buckets(
        's',
        &[
            DimBucket::new(1, 1),
            DimBucket::new(2, max_prefill).representative(16),
        ],
    );

    // Dummy data sized for the largest representative (s=16, c=16)
    let search_s = 16;
    let search_c = 16;
    cx.set_dim('s', search_s);
    cx.set_dim('c', search_c);
    runtime.set_data(input, vec![1i32; search_s]);
    runtime.set_data(q_pos_t, vec![0i32; search_s]);
    runtime.set_data(scatter_idx_t, vec![0i32; search_s]);
    runtime.set_data(gather_idx_t, vec![0i32; search_c]);
    runtime.set_data(attn_mask_t, vec![0.0f32; search_s * search_c]);
    runtime = cx.search(runtime, search_graphs);

    // Re-initialize KV cache after search (search consumes buffers)
    let cache_bytes = num_slots * KV_DIM * std::mem::size_of::<f32>();
    for i in 0..LAYERS {
        runtime.set_zeros(kv_cache.k_caches[i], cache_bytes);
        runtime.set_zeros(kv_cache.v_caches[i], cache_bytes);
    }

    let mut page_table = PageTable::new();
    let penalty: f32 = 1.05;

    // Free search intermediates before inference
    runtime.free_intermediate_buffers();

    // ════════════════════════════════════════════════════════════
    // Phase 1: Prefill sequence A (parallel)
    // ════════════════════════════════════════════════════════════
    println!(
        "\n══ Phase 1: Prefill Sequence A ({} tokens) ══",
        tokens_a.len()
    );
    let seq_a = page_table.new_sequence();
    let n_a = tokens_a.len();
    page_table.allocate(seq_a, n_a);
    let positions_a: Vec<usize> = (0..n_a).collect();
    let (scatter, gather, qpos, mask) = build_batch(&[(seq_a, positions_a)], &page_table);
    runtime.set_data(
        input,
        tokens_a.iter().map(|&t| t as i32).collect::<Vec<_>>(),
    );
    runtime.set_data(q_pos_t, qpos);
    runtime.set_data(scatter_idx_t, scatter);
    runtime.set_data(gather_idx_t, gather.to_vec());
    runtime.set_data(attn_mask_t, mask);
    let prefill_start = std::time::Instant::now();
    let logits_data = tick(
        &mut cx,
        &mut runtime,
        n_a,
        gather.len(),
        logits,
        &kv_cache,
        &cache_outputs,
    );
    let prefill_dur = prefill_start.elapsed();
    let mut seen_a = FxHashSet::default();
    let mut next_a = sample_greedy(logits_row(&logits_data, n_a - 1), &seen_a, penalty);
    seen_a.insert(next_a);
    let decoded = tokenizer.decode(&[next_a], true).unwrap();
    println!(
        "  Prefill: {:.2} ms ({} tokens, {:.1} ms/token)",
        prefill_dur.as_secs_f64() * 1e3,
        n_a,
        prefill_dur.as_secs_f64() * 1e3 / n_a as f64,
    );
    print!("[A] {decoded}");
    std::io::stdout().flush().unwrap();

    // ════════════════════════════════════════════════════════════
    // Phase 2: Decode sequence A (single-token steps)
    // ════════════════════════════════════════════════════════════
    println!("\n\n══ Phase 2: Decode Sequence A ({gen_tokens} tokens) ══");
    let mut decode_times = vec![];
    print!("[A] ");
    for _ in 0..gen_tokens {
        if next_a == EOS_TOKEN || next_a == STOP_TOKEN {
            break;
        }
        let start = std::time::Instant::now();
        let pos = page_table.context_len(seq_a);
        page_table.allocate(seq_a, 1);
        let (scatter, gather, qpos, mask) = build_batch(&[(seq_a, vec![pos])], &page_table);
        runtime.set_data(q_pos_t, qpos);
        runtime.set_data(attn_mask_t, mask);
        runtime.set_data(scatter_idx_t, scatter.to_vec());
        runtime.set_data(gather_idx_t, gather.to_vec());
        runtime.set_data(input, vec![next_a as i32]);
        let logits_data = tick(
            &mut cx,
            &mut runtime,
            1,
            gather.len(),
            logits,
            &kv_cache,
            &cache_outputs,
        );
        decode_times.push(start.elapsed());
        next_a = sample_greedy(logits_row(&logits_data, 0), &seen_a, penalty);
        seen_a.insert(next_a);
        print!("{}", tokenizer.decode(&[next_a], true).unwrap());
        std::io::stdout().flush().unwrap();
    }
    println!();
    if decode_times.len() > 1 {
        let avg = decode_times.iter().skip(1).sum::<Duration>() / (decode_times.len() - 1) as u32;
        println!("  Avg TPOT: {:.2} ms", avg.as_secs_f64() * 1e3);
    }

    // ════════════════════════════════════════════════════════════
    // Phase 3: Mixed prefill+decode tick (A decodes, B prefills)
    // ════════════════════════════════════════════════════════════
    let seq_b = page_table.new_sequence();
    let n_b = tokens_b.len();
    page_table.allocate(seq_b, n_b);
    let pos_a_mixed = page_table.context_len(seq_a);
    page_table.allocate(seq_a, 1); // 1 new slot for A's decode
    let positions_b: Vec<usize> = (0..n_b).collect();
    let total_mixed = 1 + n_b; // A: 1 decode token + B: n_b prefill tokens
    println!(
        "\n══ Phase 3: Mixed Prefill+Decode (A decode 1 + B prefill {}, s={}) ══",
        n_b, total_mixed
    );
    let (scatter, gather, qpos, mask) = build_batch(
        &[(seq_a, vec![pos_a_mixed]), (seq_b, positions_b)],
        &page_table,
    );
    let mut mixed_input = vec![next_a as i32];
    mixed_input.extend(tokens_b.iter().map(|&t| t as i32));
    runtime.set_data(input, mixed_input);
    runtime.set_data(q_pos_t, qpos);
    runtime.set_data(scatter_idx_t, scatter);
    runtime.set_data(gather_idx_t, gather.to_vec());
    runtime.set_data(attn_mask_t, mask);
    let mixed_start = std::time::Instant::now();
    let logits_data_mixed = tick(
        &mut cx,
        &mut runtime,
        total_mixed,
        gather.len(),
        logits,
        &kv_cache,
        &cache_outputs,
    );
    let mixed_dur = mixed_start.elapsed();
    // Row 0 = A's decode logits, row n_b = B's last prefill logits
    next_a = sample_greedy(logits_row(&logits_data_mixed, 0), &seen_a, penalty);
    seen_a.insert(next_a);
    let mut seen_b = FxHashSet::default();
    let mut next_b = sample_greedy(
        logits_row(&logits_data_mixed, total_mixed - 1),
        &seen_b,
        penalty,
    );
    seen_b.insert(next_b);
    println!(
        "  Mixed tick: {:.2} ms (s={}, c={})",
        mixed_dur.as_secs_f64() * 1e3,
        total_mixed,
        gather.len()
    );
    println!("[A] next: {}", tokenizer.decode(&[next_a], true).unwrap());
    println!("[B] first: {}", tokenizer.decode(&[next_b], true).unwrap());

    // ════════════════════════════════════════════════════════════
    // Phase 4: Supersequence — decode A and B together (s=2)
    // ════════════════════════════════════════════════════════════
    runtime.free_intermediate_buffers();
    println!("\n══ Phase 4: Supersequence Decode (A + B, {gen_tokens} tokens each) ══");
    let mut text_a = String::new();
    let mut text_b = String::new();
    let mut super_times = vec![];
    for _ in 0..gen_tokens {
        let a_done = next_a == EOS_TOKEN || next_a == STOP_TOKEN;
        let b_done = next_b == EOS_TOKEN || next_b == STOP_TOKEN;
        if a_done && b_done {
            break;
        }
        let start = std::time::Instant::now();
        let pos_a = page_table.context_len(seq_a);
        let pos_b = page_table.context_len(seq_b);
        page_table.allocate(seq_a, 1);
        page_table.allocate(seq_b, 1);
        let (scatter, gather, qpos, mask) =
            build_batch(&[(seq_a, vec![pos_a]), (seq_b, vec![pos_b])], &page_table);
        runtime.set_data(q_pos_t, qpos);
        runtime.set_data(attn_mask_t, mask);
        runtime.set_data(scatter_idx_t, scatter.to_vec());
        runtime.set_data(gather_idx_t, gather.to_vec());
        runtime.set_data(input, vec![next_a as i32, next_b as i32]);
        let logits_data = tick(
            &mut cx,
            &mut runtime,
            2,
            gather.len(),
            logits,
            &kv_cache,
            &cache_outputs,
        );
        super_times.push(start.elapsed());
        next_a = sample_greedy(logits_row(&logits_data, 0), &seen_a, penalty);
        next_b = sample_greedy(logits_row(&logits_data, 1), &seen_b, penalty);
        seen_a.insert(next_a);
        seen_b.insert(next_b);
        if !a_done {
            text_a += &tokenizer.decode(&[next_a], true).unwrap();
        }
        if !b_done {
            text_b += &tokenizer.decode(&[next_b], true).unwrap();
        }
    }
    println!("[A] ...{text_a}");
    println!("[B] ...{text_b}");
    if super_times.len() > 1 {
        let avg = super_times.iter().skip(1).sum::<Duration>() / (super_times.len() - 1) as u32;
        println!(
            "  Avg supersequence TPOT: {:.2} ms (2 tokens/step)",
            avg.as_secs_f64() * 1e3
        );
    }

    println!(
        "\nPage table: {} slots used / {num_slots} total",
        page_table.next_free_slot
    );
    println!("Done.");
}
