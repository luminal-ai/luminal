"""Compare flux2's `ours_*.bin` dumps against diffusers' `*.bin` references.

Run after:
  1. `python3 scripts/dump_reference.py`            (writes diffusers tensors)
  2. `DUMP_REFS=1 LOAD_REF_NOISE=1 ... cargo run --release -p flux2`
                                                    (writes ours_* tensors)

Prints per-tensor stats: max abs diff, mean abs diff, cosine similarity,
and whether shapes match. Tells you exactly which stage diverges.
"""
import os, pathlib, sys
import numpy as np

REF_DIR = pathlib.Path(__file__).parent.parent / "reference"

def load(name):
    bin_path = REF_DIR / f"{name}.bin"
    shape_path = REF_DIR / f"{name}.shape"
    if not bin_path.exists() or not shape_path.exists():
        return None
    shape = tuple(int(d) for d in shape_path.read_text().split(","))
    arr = np.fromfile(bin_path, dtype=np.float32).reshape(shape)
    return arr

PAIRS = [
    ("prompt_embeds",         "ours_prompt_embeds"),
    ("transformer_in_step0",  "ours_noise_latent"),
    ("tx_temb",               "ours_tx_temb"),
    ("tx_mod_img",            "ours_tx_mod_img"),
    ("tx_mod_txt",            "ours_tx_mod_txt"),
    ("tx_mod_single",         "ours_tx_mod_single"),
    ("tx_x_embedded",         "ours_tx_x_embedded"),
    ("tx_context_embedded",   "ours_tx_context_embedded"),
    ("tx_after_double_0_img", "ours_tx_after_double_0_img"),
    ("tx_after_double_0_txt", "ours_tx_after_double_0_txt"),
    ("tx_after_double_1_img", "ours_tx_after_double_1_img"),
    ("tx_after_double_1_txt", "ours_tx_after_double_1_txt"),
    ("tx_after_double_2_img", "ours_tx_after_double_2_img"),
    ("tx_after_double_2_txt", "ours_tx_after_double_2_txt"),
    ("tx_after_single_0",     "ours_tx_after_single_0"),
    ("tx_after_single_1",     "ours_tx_after_single_1"),
    ("tx_after_single_2",     "ours_tx_after_single_2"),
    ("velocity_step0",        "ours_velocity_step0"),
    ("vae_input",             "ours_vae_input"),
    ("vae_raw_decoded",       "ours_vae_raw_decoded"),
    ("vae_final_image",       "ours_vae_final_image"),
    ("final_image",           "ours_final_image"),
]

print(f"Comparing tensors in {REF_DIR}\n")
print(f"{'tensor':<28} {'shape':<22} {'max|Δ|':>10} {'mean|Δ|':>10} {'cos_sim':>10}")
print("-" * 92)
any_failed = False
for ref_name, ours_name in PAIRS:
    ref = load(ref_name)
    ours = load(ours_name)
    if ref is None and ours is None:
        print(f"{ref_name:<28} (neither present)")
        continue
    if ref is None:
        print(f"{ref_name:<28} (no diffusers ref)")
        continue
    if ours is None:
        print(f"{ref_name:<28} (no flux2 dump — set DUMP_REFS=1)")
        continue
    # Reshape ours to match ref where it makes sense (batch dim, etc).
    if ref.shape != ours.shape:
        if ref.size == ours.size:
            ours_r = ours.reshape(ref.shape)
        else:
            print(f"{ref_name:<28} {str(ref.shape)+' vs '+str(ours.shape):<22}  size mismatch")
            any_failed = True
            continue
    else:
        ours_r = ours
    diff = np.abs(ref - ours_r)
    max_d = float(diff.max())
    mean_d = float(diff.mean())
    rf = ref.flatten().astype(np.float64)
    of = ours_r.flatten().astype(np.float64)
    cos = float((rf @ of) / (np.linalg.norm(rf) * np.linalg.norm(of) + 1e-12))
    flag = " ✓" if cos > 0.99 and max_d < 0.5 else " ✗"
    if cos <= 0.99 or max_d >= 0.5:
        any_failed = True
    print(f"{ref_name:<28} {str(ref.shape):<22} {max_d:>10.4f} {mean_d:>10.4f} {cos:>10.4f}{flag}")

print("\n" + ("Some divergences." if any_failed else "All within tolerance."))
sys.exit(1 if any_failed else 0)
