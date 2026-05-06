//! Flux 2 (`black-forest-labs/FLUX.2-dev`) text-to-image example.
//!
//! End-to-end pipeline:
//! ```text
//! prompt → tokenize → Mistral3 text encoder ─►  text features (S_txt, 15360)
//! noise latent (S_img, 128) ─► transformer (28× denoising) ─► clean latent
//! latent ─► VAE decoder ─► (3, H, W) image ─► PNG
//! ```
//!
//! ## Modes
//! Defaults to a status-only printout. Set one of:
//!  * `VAE_TEST=1`  — decode a Gaussian latent through the VAE only
//!  * `TEXT_TEST=1` — encode the prompt through the Mistral 3 text encoder only
//!  * `FULL=1`      — full pipeline: text → transformer → VAE
//!
//! ## Required env for `FULL=1`
//!  * `LUMINAL_DISABLE_LOOP_ROLLING=1` — auto-loop-rolling currently
//!    interacts badly with our `CustomOpKind`-wrapped CUDA kernels. The
//!    rolling pass succeeds and produces seemingly-valid loop bodies,
//!    but the resulting graph causes `CUDA_ERROR_ILLEGAL_ADDRESS` on the
//!    first profile execute. Disabling the rolling makes the pipeline
//!    run cleanly. Compile time grows with layer count when rolling is
//!    off, so this is a workaround, not a fix.
//!  * `FLUX2_NUM_LAYERS` / `FLUX2_NUM_SINGLE_LAYERS` (optional) — override
//!    the default 8 + 48 transformer block counts. The default count
//!    overflows the 96 GB GPU because there's no live-range buffer
//!    reuse in `CudaRuntime::allocate_intermediate_buffers` — every
//!    intermediate is alive for the whole forward pass. `1 + 1` runs
//!    the full pipeline end-to-end at 1024² in well under a minute and
//!    is the right setting for plumbing-validation. Higher counts
//!    (e.g. `8 + 16`) work but use proportionally more memory.
//!
//! ## Memory plan
//! GPU is 96 GB; transformer (60 GB BF16) + text encoder (33 GB BF16) +
//! VAE (336 MB) won't all fit. The full pipeline keeps **at most one** large
//! model resident at a time:
//!   1. Load text encoder, encode prompt, **drop the runtime** to free 33 GB.
//!   2. Load transformer, run the diffusion loop, **drop the runtime**.
//!   3. Load VAE, decode, dump PNG.

mod hf;
#[allow(dead_code)]
mod quant;
mod scheduler;
mod text_encoder;
mod transformer;
mod vae;

use std::fs::File;
use std::io::BufWriter;
use std::time::Instant;

use luminal::graph::BuildSearchSpaceOptions;
use luminal::prelude::*;
use luminal_cuda_lite::{cudarc::driver::CudaContext, runtime::CudaRuntime};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::StandardNormal;
use scheduler::{compute_mu, euler_step, make_schedule, SchedulerConfig};
use tokenizers::Tokenizer;
use vae::{VaeDecoder, LATENT_CHANNELS, VAE_DOWNSAMPLE};

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Override-able via `TEXT_LEN=N` for testing. Diffusers' Flux 2 pipeline
/// pads to 512; smaller works for the text encoder's transformer compile to
/// fit in fewer GPU temp buffers during search.
const DEFAULT_TEXT_LEN: usize = 512;

fn text_len() -> usize {
    env_usize("TEXT_LEN", DEFAULT_TEXT_LEN)
}
const DEFAULT_GUIDANCE: f32 = 2.5;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "a cat in a hat".to_string());
    let width = env_usize("WIDTH", 1024);
    let height = env_usize("HEIGHT", 1024);
    let steps = env_usize("STEPS", 28);
    let guidance = env_f32("GUIDANCE", DEFAULT_GUIDANCE);

    println!("Prompt: {prompt}");
    println!("Resolution: {width}x{height}, steps={steps}, guidance={guidance}");

    if std::env::var("VAE_TEST").is_ok() {
        run_vae_only(width, height)?;
    } else if std::env::var("TEXT_TEST").is_ok() {
        let _ = run_text_encoder(&prompt)?;
    } else if std::env::var("FULL").is_ok() {
        run_full_pipeline(&prompt, width, height, steps, guidance)?;
    } else {
        print_status(width, height, steps);
    }
    Ok(())
}

fn print_status(width: usize, height: usize, steps: usize) {
    let image_seq_len = (height / 16) * (width / 16);
    let cfg = SchedulerConfig::default();
    let mu = compute_mu(&cfg, image_seq_len);
    let (sigmas, timesteps) = make_schedule(&cfg, steps, mu);
    println!("image_seq_len={image_seq_len}, mu={mu:.4}");
    println!("sigmas:    {:?}", round(&sigmas, 4));
    println!("timesteps: {:?}", round(&timesteps, 2));
    println!("\nStatus:");
    println!("  scheduler:        ✅ tested vs diffusers");
    println!("  hf loader:        ✅ F8/F4/F6 dtypes (transformer is plain BF16, no NVFP4 path)");
    println!("  VAE decoder:      ✅ runs end-to-end up to ~128² (set VAE_TEST=1)");
    println!("  text encoder:     ⚠ graph builds, not validated (set TEXT_TEST=1, needs 48 GB)");
    println!("  transformer:      ⚠ graph builds, never run (needs 64 GB shards)");
    println!("  full pipeline:    ⚠ requires ≈110 GB downloads (set FULL=1)");
}

// =============================================================================
// VAE-only test path (already validated)
// =============================================================================

fn run_vae_only(width: usize, height: usize) -> Result<(), Box<dyn std::error::Error>> {
    let h_lat = height / VAE_DOWNSAMPLE;
    let w_lat = width / VAE_DOWNSAMPLE;
    println!(
        "\nVAE-only test: latent ({LATENT_CHANNELS}, {h_lat}, {w_lat}) -> image (3, {height}, {width})",
    );

    let vae_path = hf::fetch_vae()?;
    let mut cx = Graph::default();
    let latent = cx.named_tensor("latent", (LATENT_CHANNELS, h_lat, w_lat));
    let decoder = VaeDecoder::new(&mut cx);
    let out = decoder.forward(latent).output();
    // Memory-budget enforcement is opt-in here. The estimator
    // (`memory_analysis::estimate_graph_memory_bytes`) sums every node's
    // output bytes — including views — across the whole graph rather
    // than computing peak live memory. For the conv-heavy decoder this
    // sum runs into the hundreds of GiB even at 256² where real peak
    // memory is only a few GiB, so any reasonable budget rejects all
    // candidates. Set `VAE_MEM_GIB` to opt in if you need it.
    if let Ok(g) = std::env::var("VAE_MEM_GIB").and_then(|s| s.parse::<usize>().map_err(|_| std::env::VarError::NotPresent)) {
        cx.build_search_space_with_options::<CudaRuntime>(
            BuildSearchSpaceOptions::new().max_memory_gib(g),
        );
    } else {
        cx.build_search_space::<CudaRuntime>();
    }

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let mut runtime = CudaRuntime::initialize(stream);
    runtime.load_safetensors(&cx, vae_path.to_str().unwrap());

    let mut rng = StdRng::seed_from_u64(0);
    let n_latent = LATENT_CHANNELS * h_lat * w_lat;
    let latent_data: Vec<f32> = (0..n_latent)
        .map(|_| rng.sample::<f32, _>(StandardNormal))
        .collect();
    runtime.set_data(latent, latent_data);
    runtime = cx.search(runtime, env_usize("SEARCH_ITERS", 5));
    runtime.execute(&cx.dyn_map);
    let out_data = runtime.get_f32(out);
    save_png("out.png", &out_data, width, height)?;
    println!("Wrote out.png");
    Ok(())
}

// =============================================================================
// Text encoder path (compute the (S_txt, 15360) prompt features)
// =============================================================================

fn tokenize_prompt(
    tokenizer: &Tokenizer,
    prompt: &str,
    text_len: usize,
) -> Result<(Vec<i32>, usize), Box<dyn std::error::Error>> {
    // Format the chat template (system + user) the way Flux 2's pipeline does,
    // then tokenize. The Mistral 3 tokenizer treats `[SYSTEM_PROMPT]`,
    // `[/SYSTEM_PROMPT]`, `[INST]`, `[/INST]` as added tokens, so they
    // round-trip as single ids; the `<s>` BOS is added by the tokenizer
    // (`add_bos_token = true`).
    let formatted = text_encoder::format_chat(text_encoder::SYSTEM_MESSAGE, prompt);
    let encoded = tokenizer
        .encode(formatted, true)
        .map_err(|e| format!("tokenize failed: {e}"))?;
    let mut ids: Vec<i32> = encoded.get_ids().iter().map(|&i| i as i32).collect();
    let real_len = ids.len();
    if real_len > text_len {
        ids.truncate(text_len);
    } else {
        // Right-pad to text_len with the pad token (= 0 for Mistral).
        ids.resize(text_len, 0);
    }
    Ok((ids, real_len.min(text_len)))
}

fn run_text_encoder(prompt: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    println!("\n[1/3] Resolving text encoder weights...");
    let tok_path = hf::fetch_tokenizer()?;
    let te_paths = hf::fetch_sharded("text_encoder")?;

    println!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| format!("tokenizer: {e}"))?;
    let text_len = text_len();
    let (ids, real_len) = tokenize_prompt(&tokenizer, prompt, text_len)?;
    println!(
        "  prompt → {} ids ({} real, padded to {})",
        ids.len(),
        real_len,
        text_len,
    );

    println!("Building text encoder graph...");
    let mut cx = Graph::default();
    let input_ids = cx
        .named_tensor("__input_ids", text_len)
        .as_dtype(DType::Int);
    let pos_ids = cx.named_tensor("__pos_ids", text_len).as_dtype(DType::Int);
    let encoder = text_encoder::Mistral3TextEncoder::init(&mut cx);
    let features = encoder.forward(input_ids, pos_ids).output();
    // Memory-budget enforcement is opt-in (the estimator over-counts; see
    // the matching comment in `run_vae_only`). Set `TEXT_MEM_GIB` to opt in.
    if let Ok(g) = std::env::var("TEXT_MEM_GIB").and_then(|s| s.parse::<usize>().map_err(|_| std::env::VarError::NotPresent)) {
        cx.build_search_space_with_options::<CudaRuntime>(
            BuildSearchSpaceOptions::new().max_memory_gib(g),
        );
    } else {
        cx.build_search_space::<CudaRuntime>();
    }

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let mut runtime = CudaRuntime::initialize(stream);

    println!(
        "Loading {} text encoder shards (~48 GB BF16)...",
        te_paths.len()
    );
    let t0 = Instant::now();
    for p in &te_paths {
        runtime.load_safetensors(&cx, p.to_str().unwrap());
    }
    println!("  loaded in {:.1}s", t0.elapsed().as_secs_f64());

    runtime.set_data(input_ids, ids);
    runtime.set_data(pos_ids, (0..text_len as i32).collect::<Vec<_>>());

    println!("Compiling text encoder...");
    let t0 = Instant::now();
    runtime = cx.search(runtime, env_usize("SEARCH_ITERS", 5));
    println!("  compile done in {:.1}s", t0.elapsed().as_secs_f64());

    println!("Encoding prompt...");
    let t0 = Instant::now();
    runtime.execute(&cx.dyn_map);
    let out = runtime.get_f32(features);
    println!("  encode done in {:.1}s", t0.elapsed().as_secs_f64());
    println!(
        "  features: len={} (= {} × {})",
        out.len(),
        text_len,
        text_encoder::OUTPUT_DIM,
    );
    Ok(out)
}

// =============================================================================
// Full pipeline: text → diffusion → VAE → PNG
// =============================================================================

fn run_full_pipeline(
    prompt: &str,
    width: usize,
    height: usize,
    steps: usize,
    guidance: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    // VAE latent grid: (LATENT_CHANNELS=32, h_lat, w_lat)
    let h_lat = height / VAE_DOWNSAMPLE;
    let w_lat = width / VAE_DOWNSAMPLE;
    // Transformer "pack" grid: (IN_CHANNELS=128, h_pack, w_pack) = (32*4, h_lat/2, w_lat/2).
    // The diffusers pipeline folds 2×2 spatial pixels into the channel axis
    // before the transformer (`_patchify_latents`) and undoes it after
    // (`_unpatchify_latents`), so the transformer sees `(S_img, 128)` tokens
    // where `S_img = (H/16) * (W/16)`.
    assert!(
        h_lat.is_multiple_of(2) && w_lat.is_multiple_of(2),
        "WIDTH and HEIGHT must be multiples of 16 (got {width}x{height})",
    );
    let h_pack = h_lat / 2;
    let w_pack = w_lat / 2;
    let s_img = h_pack * w_pack;
    let s_txt = text_len();

    // ── 1. TEXT ENCODE ─────────────────────────────────────────────────────────
    let text_features = run_text_encoder(prompt)?;
    assert_eq!(text_features.len(), s_txt * text_encoder::OUTPUT_DIM);

    // ── 2. DIFFUSION LOOP ──────────────────────────────────────────────────────
    println!("\n[2/3] Resolving transformer weights...");
    let tx_paths = hf::fetch_sharded("transformer")?;
    println!(
        "  {} transformer shards downloaded ({:.1} GB total)",
        tx_paths.len(),
        tx_paths
            .iter()
            .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
            .sum::<u64>() as f64
            / 1e9,
    );

    let cfg = SchedulerConfig::default();
    let image_seq_len = s_img;
    let mu = compute_mu(&cfg, image_seq_len);
    let (sigmas, timesteps) = make_schedule(&cfg, steps, mu);
    println!("  scheduler: mu={mu:.4}, {} steps", timesteps.len());

    // Pre-compute RoPE tables (host-side; these are constant per resolution).
    // Grid is the post-pack `(h_pack, w_pack)`, matching what the transformer
    // and diffusers' `_prepare_latent_ids` see.
    let (rope_cos, rope_sin) = transformer::build_rope_tables(s_txt, h_pack, w_pack);
    let s_total = s_txt + s_img;
    assert_eq!(rope_cos.len(), s_total * transformer::HEAD_DIM);

    // Initial noise latent in (S_img, IN_CHANNELS) layout.
    let mut rng = StdRng::seed_from_u64(env_usize("SEED", 0) as u64);
    let mut latent: Vec<f32> = (0..s_img * transformer::IN_CHANNELS)
        .map(|_| rng.sample::<f32, _>(StandardNormal))
        .collect();

    println!("Building transformer graph...");
    let mut cx = Graph::default();
    // Inputs that change per diffusion step.
    let latent_in = cx.named_tensor("__latent", (s_img, transformer::IN_CHANNELS));
    let timestep_in = cx.named_tensor("__timestep", 1);
    // Inputs that are constant across the whole diffusion loop. `.persist()`
    // marks them as outputs so their buffers survive between successive
    // `runtime.execute()` calls; without this the runtime treats them as
    // transient intermediates and a second `execute()` reads freed memory
    // (manifests as `CUDA_ERROR_ILLEGAL_ADDRESS` on the post-kernel sync).
    let text_in = cx
        .named_tensor("__text", (s_txt, text_encoder::OUTPUT_DIM))
        .persist();
    let cos_in = cx
        .named_tensor("__rope_cos", (s_total, transformer::HEAD_DIM))
        .persist();
    let sin_in = cx
        .named_tensor("__rope_sin", (s_total, transformer::HEAD_DIM))
        .persist();
    let guidance_in = cx.named_tensor("__guidance", 1).persist();

    let model = transformer::Flux2Transformer::init(&mut cx);
    let velocity = model
        .forward(latent_in, text_in, cos_in, sin_in, timestep_in, guidance_in)
        .output();

    println!("Building search space (this is the long step — many minutes for the full DiT)...");
    if let Ok(g) = std::env::var("TX_MEM_GIB").and_then(|s| s.parse::<usize>().map_err(|_| std::env::VarError::NotPresent)) {
        cx.build_search_space_with_options::<CudaRuntime>(
            BuildSearchSpaceOptions::new().max_memory_gib(g),
        );
    } else {
        cx.build_search_space::<CudaRuntime>();
    }

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let mut runtime = CudaRuntime::initialize(stream);

    println!(
        "Loading {} transformer shards (~{:.1} GB BF16)...",
        tx_paths.len(),
        tx_paths
            .iter()
            .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
            .sum::<u64>() as f64
            / 1e9,
    );
    let t0 = Instant::now();
    for p in &tx_paths {
        runtime.load_safetensors(&cx, p.to_str().unwrap());
    }
    println!("  loaded in {:.1}s", t0.elapsed().as_secs_f64());

    // Set the inputs that don't change across steps.
    runtime.set_data(text_in, text_features);
    runtime.set_data(cos_in, rope_cos);
    runtime.set_data(sin_in, rope_sin);
    runtime.set_data(guidance_in, vec![guidance * 1000.0]);

    // First-step dummy values so search() has shapes/data to profile against.
    runtime.set_data(latent_in, latent.clone());
    runtime.set_data(timestep_in, vec![timesteps[0] * 1000.0]);

    println!("Compiling transformer (search)...");
    let t0 = Instant::now();
    runtime = cx.search(runtime, env_usize("SEARCH_ITERS", 5));
    println!("  compile done in {:.1}s", t0.elapsed().as_secs_f64());

    println!("Running diffusion loop ({} steps)...", timesteps.len());
    for (i, &t) in timesteps.iter().enumerate() {
        let step_start = Instant::now();
        runtime.set_data(latent_in, latent.clone());
        runtime.set_data(timestep_in, vec![t * 1000.0]);
        runtime.execute(&cx.dyn_map);
        let v = runtime.get_f32(velocity);
        // Euler integrate: latent += (sigma_next - sigma) * v
        euler_step(&mut latent, &v, sigmas[i], sigmas[i + 1]);
        println!(
            "  step {:>2}/{}: t={:>8.2}, σ {:.4} → {:.4} ({:.1}s)",
            i + 1,
            timesteps.len(),
            t,
            sigmas[i],
            sigmas[i + 1],
            step_start.elapsed().as_secs_f64(),
        );
    }

    // Drop the transformer runtime to free its weights before loading the VAE.
    drop(runtime);
    drop(cx);

    // ── 3. VAE DECODE ──────────────────────────────────────────────────────────
    println!("\n[3/3] Decoding latent through VAE...");
    let vae_path = hf::fetch_vae()?;

    // Convert the diffusion output `(S_img, 128)` to the VAE's input shape
    // `(32, h_lat, w_lat)` on the host. Mirrors the diffusers pipeline:
    //   1. _unpack_latents_with_ids: (S_img, 128) -> (128, h_pack, w_pack)
    //   2. BN inverse:               x = x * bn_std + bn_mean   (per-channel)
    //   3. _unpatchify_latents:      (128, h_pack, w_pack) -> (32, h_lat, w_lat)
    let bn_mean = read_safetensors_f32(&vae_path, "bn.running_mean")?;
    let bn_var = read_safetensors_f32(&vae_path, "bn.running_var")?;
    assert_eq!(bn_mean.len(), transformer::IN_CHANNELS);
    assert_eq!(bn_var.len(), transformer::IN_CHANNELS);
    const BN_EPS: f32 = 1e-4; // matches vae/config.json batch_norm_eps=0.0001
    let bn_std: Vec<f32> = bn_var.iter().map(|v| (v + BN_EPS).sqrt()).collect();

    let unpacked = unpack_packed_host(
        &latent,
        transformer::IN_CHANNELS,
        h_pack,
        w_pack,
    );
    let denormed = bn_inverse_host(&unpacked, &bn_mean, &bn_std, transformer::IN_CHANNELS);
    let vae_input = unpatchify_host(&denormed, LATENT_CHANNELS, h_pack, w_pack);
    assert_eq!(vae_input.len(), LATENT_CHANNELS * h_lat * w_lat);

    let mut cx = Graph::default();
    let latent_in = cx.named_tensor("latent", (LATENT_CHANNELS, h_lat, w_lat));
    let decoder = VaeDecoder::new(&mut cx);
    let out = decoder.forward(latent_in).output();
    if let Ok(g) = std::env::var("VAE_MEM_GIB").and_then(|s| s.parse::<usize>().map_err(|_| std::env::VarError::NotPresent)) {
        cx.build_search_space_with_options::<CudaRuntime>(
            BuildSearchSpaceOptions::new().max_memory_gib(g),
        );
    } else {
        cx.build_search_space::<CudaRuntime>();
    }

    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let mut runtime = CudaRuntime::initialize(stream);
    runtime.load_safetensors(&cx, vae_path.to_str().unwrap());
    runtime.set_data(latent_in, vae_input);
    runtime = cx.search(runtime, env_usize("SEARCH_ITERS", 5));
    runtime.execute(&cx.dyn_map);
    let img = runtime.get_f32(out);
    save_png("out.png", &img, width, height)?;
    println!("Wrote out.png");
    Ok(())
}

// =============================================================================
// Host-side pipeline glue: pack/unpack/BN/unpatchify between the transformer
// and the VAE. These mirror diffusers' Flux2Pipeline static methods exactly.
// =============================================================================

/// Inverse of `_pack_latents`: `(S_img, C) -> (C, h_pack, w_pack)` row-major.
fn unpack_packed_host(packed: &[f32], c: usize, h_pack: usize, w_pack: usize) -> Vec<f32> {
    let s_img = h_pack * w_pack;
    assert_eq!(packed.len(), s_img * c);
    let mut out = vec![0.0_f32; c * s_img];
    for hi in 0..h_pack {
        for wi in 0..w_pack {
            let token = hi * w_pack + wi;
            for ci in 0..c {
                out[ci * s_img + token] = packed[token * c + ci];
            }
        }
    }
    out
}

/// `latent[c, *] = latent[c, *] * std[c] + mean[c]`. In-place by-copy.
fn bn_inverse_host(latent: &[f32], mean: &[f32], std: &[f32], c: usize) -> Vec<f32> {
    let hw = latent.len() / c;
    assert_eq!(mean.len(), c);
    assert_eq!(std.len(), c);
    let mut out = vec![0.0_f32; latent.len()];
    for ci in 0..c {
        let m = mean[ci];
        let s = std[ci];
        for i in 0..hw {
            out[ci * hw + i] = latent[ci * hw + i] * s + m;
        }
    }
    out
}

/// `_unpatchify_latents`: `(C*4, h_pack, w_pack) -> (C, 2*h_pack, 2*w_pack)`.
///
/// Diffusers does:
/// ```python
/// latents.reshape(B, C, 2, 2, H, W).permute(0, 1, 4, 2, 5, 3).reshape(B, C, 2H, 2W)
/// ```
/// So input channel `c*4 + ph*2 + pw` (with ph, pw in {0,1}) maps to output
/// position `(c, hi*2 + ph, wi*2 + pw)`.
fn unpatchify_host(packed: &[f32], c_out: usize, h_pack: usize, w_pack: usize) -> Vec<f32> {
    assert_eq!(packed.len(), c_out * 4 * h_pack * w_pack);
    let h_lat = h_pack * 2;
    let w_lat = w_pack * 2;
    let mut out = vec![0.0_f32; c_out * h_lat * w_lat];
    for c in 0..c_out {
        for ph in 0..2 {
            for pw in 0..2 {
                let in_c = c * 4 + ph * 2 + pw;
                for hi in 0..h_pack {
                    for wi in 0..w_pack {
                        let in_idx = in_c * h_pack * w_pack + hi * w_pack + wi;
                        let out_h = hi * 2 + ph;
                        let out_w = wi * 2 + pw;
                        let out_idx = c * h_lat * w_lat + out_h * w_lat + out_w;
                        out[out_idx] = packed[in_idx];
                    }
                }
            }
        }
    }
    out
}

/// Read one F32 tensor by name from a single-file safetensors archive.
fn read_safetensors_f32(
    path: &std::path::Path,
    name: &str,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path)?;
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)?;
    let header_len = u64::from_le_bytes(header_len_bytes) as usize;
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)?;
    let header: serde_json::Value = serde_json::from_slice(&header_bytes)?;
    let info = header
        .get(name)
        .ok_or_else(|| format!("safetensors: tensor '{name}' not found in {path:?}"))?;
    let dtype = info["dtype"].as_str().unwrap_or("");
    if dtype != "F32" {
        return Err(format!("safetensors: tensor '{name}' has dtype {dtype}, want F32").into());
    }
    let offsets = &info["data_offsets"];
    let start = offsets[0].as_u64().unwrap();
    let end = offsets[1].as_u64().unwrap();
    let n_bytes = (end - start) as usize;
    file.seek(SeekFrom::Start(8 + header_len as u64 + start))?;
    let mut buf = vec![0u8; n_bytes];
    file.read_exact(&mut buf)?;
    Ok(buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn save_png(path: &str, chw: &[f32], w: usize, h: usize) -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(chw.len(), 3 * h * w, "save_png: shape mismatch");
    let mut bytes = vec![0u8; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let v = chw[c * h * w + y * w + x];
                let v = ((v + 1.0) * 0.5 * 255.0).clamp(0.0, 255.0);
                bytes[(y * w + x) * 3 + c] = v as u8;
            }
        }
    }
    let file = File::create(path)?;
    let bw = BufWriter::new(file);
    let mut encoder = png::Encoder::new(bw, w as u32, h as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(&bytes)?;
    Ok(())
}

fn round(v: &[f32], digits: u32) -> Vec<f32> {
    let s = 10f32.powi(digits as i32);
    v.iter().map(|x| (x * s).round() / s).collect()
}
