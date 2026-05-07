"""Dump diffusers' VAE pipeline stages so we can compare each step.

Captures (for STEPS=1 by default):
  - vae_packed_latent.bin     — after diffusion, before unpack: (B, S, 128)
  - vae_unpacked.bin          — after _unpack_latents_with_ids: (B, 128, h_pack, w_pack)
  - vae_bn_inversed.bin       — after BN inverse: same shape as above
  - vae_unpatchified.bin      — after _unpatchify_latents: (B, 32, h_lat, w_lat)
  - vae_raw_decoded.bin       — vae.decode output before postprocess: (B, 3, H, W)
  - vae_final_image.bin       — after image_processor.postprocess: (B, 3, H, W) in [0,1]

Run after `dump_reference.py` so noise/text are consistent.
"""
import os, pathlib, torch
import numpy as np

HEIGHT = int(os.environ.get("HEIGHT", "128"))
WIDTH = int(os.environ.get("WIDTH", "128"))
STEPS = int(os.environ.get("STEPS", "1"))
SEED = int(os.environ.get("SEED", "0"))
PROMPT = os.environ.get("PROMPT", "a cat in a hat")

OUT_DIR = pathlib.Path(__file__).parent.parent / "reference"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def save(name, t):
    a = t.detach().to(torch.float32).cpu().numpy()
    a.tofile(OUT_DIR / f"{name}.bin")
    (OUT_DIR / f"{name}.shape").write_text(",".join(str(d) for d in a.shape))
    print(f"  {name}: shape={list(a.shape)} min={a.min():+.4f} max={a.max():+.4f} mean={a.mean():+.4f}")

print(f"Loading FLUX.2-dev ({HEIGHT}x{WIDTH}, {STEPS} steps, seed={SEED})")
from diffusers import Flux2Pipeline
pipe = Flux2Pipeline.from_pretrained("black-forest-labs/FLUX.2-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

# Hook decode to capture raw output
caps = {}
orig_decode = pipe.vae.decode
def hooked_decode(latents, return_dict=False):
    caps["vae_input"] = latents.clone()
    result = orig_decode(latents, return_dict=return_dict)
    if isinstance(result, tuple):
        caps["vae_raw_decoded"] = result[0].clone()
    else:
        caps["vae_raw_decoded"] = result.sample.clone() if hasattr(result, "sample") else result.clone()
    return result
pipe.vae.decode = hooked_decode

# Patch the pipeline's __call__ to grab packed_latent + intermediates.
# Easier: patch _unpack_latents_with_ids and _unpatchify_latents (static methods)
# by hooking __call__ to grab `latents` after the diffusion loop and recompute.
# Simpler: just intercept `image = self.vae.decode(...)` at the call.

# Run the pipeline once
gen = torch.Generator(device="cuda").manual_seed(SEED)
print("Running pipeline...")
with torch.no_grad():
    image = pipe(
        prompt=PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=STEPS,
        guidance_scale=2.5,
        generator=gen,
        output_type="pt",
    ).images[0]

# Save final outputs
save("vae_input", caps["vae_input"])  # (B, 32, h_lat, w_lat) — input to vae.decode
save("vae_raw_decoded", caps["vae_raw_decoded"])
# `image` is (3, H, W) tensor in [0,1] (postprocessed)
save("vae_final_image", image.unsqueeze(0))
print(f"\nfinal image stats: min={image.min().item():.4f} max={image.max().item():.4f} mean={image.mean().item():.4f}")
