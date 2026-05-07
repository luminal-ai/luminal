"""Hook into diffusers' Flux2 transformer and dump every intermediate.

Saves to `examples/flux2/reference/`:
  - tx_x_embedded.bin       — latent after x_embedder (1, S_img, HIDDEN)
  - tx_context_embedded.bin — text after context_embedder (1, S_txt, HIDDEN)
  - tx_temb.bin             — time+guidance embedding (1, HIDDEN)
  - tx_mod_img.bin          — double_stream_modulation_img output
  - tx_mod_txt.bin          — double_stream_modulation_txt output
  - tx_mod_single.bin       — single_stream_modulation output
  - tx_after_double_K.bin   — outputs after each dual-stream block (img and txt halves)
  - tx_after_single_K.bin   — outputs after each single-stream block
  - tx_velocity.bin         — final velocity output

Configurable: HEIGHT, WIDTH, STEPS, SEED, MAX_DOUBLE_DUMPS, MAX_SINGLE_DUMPS.
"""
import os, json, pathlib, torch, numpy as np

HEIGHT = int(os.environ.get("HEIGHT", "128"))
WIDTH = int(os.environ.get("WIDTH", "128"))
STEPS = int(os.environ.get("STEPS", "1"))
SEED = int(os.environ.get("SEED", "0"))
PROMPT = os.environ.get("PROMPT", "a cat in a hat")
MAX_DOUBLE = int(os.environ.get("MAX_DOUBLE_DUMPS", "3"))  # dump first N dual blocks
MAX_SINGLE = int(os.environ.get("MAX_SINGLE_DUMPS", "3"))  # dump first N single blocks

OUT_DIR = pathlib.Path(__file__).parent.parent / "reference"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def save(name, t):
    a = t.detach().to(torch.float32).cpu().numpy()
    a.tofile(OUT_DIR / f"{name}.bin")
    (OUT_DIR / f"{name}.shape").write_text(",".join(str(d) for d in a.shape))
    print(f"  {name}: shape={list(a.shape)} min={a.min():+.4f} max={a.max():+.4f} mean={a.mean():+.4f} std={a.std():.4f}")

print(f"Loading FLUX.2-dev ({HEIGHT}x{WIDTH}, {STEPS} steps, seed={SEED})")
from diffusers import Flux2Pipeline
pipe = Flux2Pipeline.from_pretrained("black-forest-labs/FLUX.2-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

# Patch forward so we can capture intermediates without rewriting the model.
tx = pipe.transformer
orig_forward = tx.forward

# Capture from x_embedder, context_embedder, time_guidance_embed, modulation modules, every block.
caps = {}

# Forward pre-hooks on input projections
def cap_after(name):
    def hook(module, inputs, output):
        caps[name] = output
    return hook

tx.x_embedder.register_forward_hook(cap_after("x_embedded"))
tx.context_embedder.register_forward_hook(cap_after("context_embedded"))
tx.time_guidance_embed.register_forward_hook(cap_after("temb"))
tx.double_stream_modulation_img.register_forward_hook(cap_after("mod_img"))
tx.double_stream_modulation_txt.register_forward_hook(cap_after("mod_txt"))
tx.single_stream_modulation.register_forward_hook(cap_after("mod_single"))

# Per-block dumps: block(args) returns (encoder_hidden_states, hidden_states) for double,
# or hidden_states for single.
for i, blk in enumerate(tx.transformer_blocks[:MAX_DOUBLE]):
    def make_hook(idx):
        def hook(module, inputs, output):
            # output is (encoder_hidden_states, hidden_states)
            ehs, hs = output
            caps[f"after_double_{idx}_txt"] = ehs
            caps[f"after_double_{idx}_img"] = hs
        return hook
    blk.register_forward_hook(make_hook(i))

for i, blk in enumerate(tx.single_transformer_blocks[:MAX_SINGLE]):
    def make_hook(idx):
        def hook(module, inputs, output):
            caps[f"after_single_{idx}"] = output
        return hook
    blk.register_forward_hook(make_hook(i))

# Run pipe with seed
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

print(f"\nDumping {len(caps)} captured tensors:")
for name, val in caps.items():
    save(f"tx_{name}", val)

print("\nDone.")
