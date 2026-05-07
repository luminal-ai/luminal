"""Dump reference tensors from diffusers FLUX.2-dev for numerics comparison.

Saves to `examples/flux2/reference/`:
  - prompt_embeds.bin   — text encoder output, (1, S_txt, 15360) F32
  - noise_latent.bin    — initial noise after pack, (1, S_img, 128) F32
  - velocity_step0.bin  — transformer output at step 0, (1, S_img, 128) F32
  - vae_input.bin       — final clean latent before VAE, (1, S_img, 128) F32
  - vae_output.bin      — final image, (H, W, 3) F32 in [0,1]
  - config.json         — height, width, steps, seed, prompt, image_seq_len, mu

Pass HEIGHT, WIDTH, STEPS, SEED via env. Default HEIGHT=WIDTH=128, STEPS=1.
"""

import os, json, struct, pathlib, sys
import torch
import numpy as np

HEIGHT = int(os.environ.get("HEIGHT", "128"))
WIDTH = int(os.environ.get("WIDTH", "128"))
STEPS = int(os.environ.get("STEPS", "1"))
SEED = int(os.environ.get("SEED", "0"))
PROMPT = os.environ.get("PROMPT", "a cat in a hat")
GUIDANCE = float(os.environ.get("GUIDANCE", "2.5"))

OUT_DIR = pathlib.Path(__file__).parent.parent / "reference"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def save_tensor(name, t):
    """Save a tensor as raw F32 little-endian + a .shape sidecar."""
    a = t.detach().to(torch.float32).cpu().numpy()
    a.tofile(OUT_DIR / f"{name}.bin")
    (OUT_DIR / f"{name}.shape").write_text(",".join(str(d) for d in a.shape))
    print(f"  {name}: shape={list(a.shape)} dtype={a.dtype} "
          f"min={a.min():+.4f} max={a.max():+.4f} mean={a.mean():+.4f}")

print(f"Loading FLUX.2-dev pipeline ({HEIGHT}x{WIDTH}, {STEPS} steps, seed={SEED})")
from diffusers import Flux2Pipeline
pipe = Flux2Pipeline.from_pretrained(
    "black-forest-labs/FLUX.2-dev",
    torch_dtype=torch.bfloat16,
)
# Full transformer + text encoder + VAE doesn't fit in 96 GB GPU when
# loaded eagerly (everything goes through .to("cuda") sequentially and
# peaks while moving). CPU offload swaps modules in as needed.
pipe.enable_model_cpu_offload()

# --- 1. Text encoder ---
print("Encoding prompt...")
with torch.no_grad():
    # encode_prompt returns (prompt_embeds, prompt_embeds_seq_lens)
    prompt_embeds, _ = pipe.encode_prompt(
        prompt=PROMPT,
        max_sequence_length=512,
        device="cuda",
    )
    save_tensor("prompt_embeds", prompt_embeds)

# --- 2. Set up diffusion loop with fixed seed and capture per-step ---
gen = torch.Generator(device="cuda").manual_seed(SEED)

# Hook the transformer to capture velocity at step 0
captured = {}
orig_forward = pipe.transformer.forward
step_counter = [0]

def hooked_forward(*args, **kwargs):
    out = orig_forward(*args, **kwargs)
    if step_counter[0] == 0:
        # First arg is hidden_states (the noisy latent in pack form)
        captured["transformer_in_step0"] = args[0] if args else kwargs.get("hidden_states")
        captured["velocity_step0"] = out[0] if isinstance(out, tuple) else out.sample if hasattr(out, "sample") else out
    step_counter[0] += 1
    return out

pipe.transformer.forward = hooked_forward

print(f"Running diffusion loop ({STEPS} steps)...")
with torch.no_grad():
    image = pipe(
        prompt=PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        generator=gen,
        output_type="pt",  # tensor instead of PIL
    ).images[0]

# --- 3. Save captured tensors ---
if "transformer_in_step0" in captured:
    save_tensor("transformer_in_step0", captured["transformer_in_step0"])
if "velocity_step0" in captured:
    save_tensor("velocity_step0", captured["velocity_step0"])
save_tensor("final_image", image)

# --- 4. Save config ---
image_seq_len = (HEIGHT // 16) * (WIDTH // 16)
cfg = {
    "height": HEIGHT,
    "width": WIDTH,
    "steps": STEPS,
    "seed": SEED,
    "prompt": PROMPT,
    "guidance_scale": GUIDANCE,
    "image_seq_len": image_seq_len,
}
(OUT_DIR / "config.json").write_text(json.dumps(cfg, indent=2))
print(f"\nWrote reference tensors to {OUT_DIR}")
print(f"config: {cfg}")
