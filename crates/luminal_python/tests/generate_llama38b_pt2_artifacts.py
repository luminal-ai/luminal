"""Generate pre-computed PT2 artifacts for test_hf_llama38b_cached.

Run once:
    uv run python tests/generate_llama38b_pt2_artifacts.py

Produces:
    tests/llama38b.pt2                  — torch.export of Llama 3.1-8B
    tests/llama38b_weights.safetensors  — model weights
    tests/llama38b_ref_logits.pt        — reference logits for input_ids=[1,2,3,4]
                                          (shared with ONNX artifact script)
"""

from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, LlamaForCausalLM

SCRIPT_DIR = Path(__file__).resolve().parent
PT2_PATH = SCRIPT_DIR / "llama38b.pt2"
WEIGHTS_PATH = SCRIPT_DIR / "llama38b_weights.safetensors"
LOGITS_PATH = SCRIPT_DIR / "llama38b_ref_logits.pt"

INPUT_IDS = torch.tensor([[1, 2, 3, 4]])


def main():
    config = AutoConfig.from_pretrained("NousResearch/Meta-Llama-3.1-8B-Instruct")
    config.use_cache = False
    config._attn_implementation = "eager"

    print("Loading model on CPU...")
    model = LlamaForCausalLM.from_pretrained(
        "NousResearch/Meta-Llama-3.1-8B-Instruct",
        config=config,
        torch_dtype=torch.float32,
    ).eval()

    # Generate reference logits (shared with ONNX artifact script)
    if not LOGITS_PATH.exists():
        print("Computing reference logits...")
        with torch.no_grad():
            ref_logits = model(INPUT_IDS).logits.clone()
        print(f"Reference logits shape: {ref_logits.shape}")
        print(f"Saving reference logits to {LOGITS_PATH}")
        torch.save(ref_logits, LOGITS_PATH)
    else:
        print(f"Reference logits already exist at {LOGITS_PATH}, skipping")

    print(f"Exporting PT2 to {PT2_PATH}")
    ep = torch.export.export(model, (INPUT_IDS,), strict=False)
    torch.export.save(ep, str(PT2_PATH))

    print(f"Saving weights to {WEIGHTS_PATH}")
    state_dict = {k: v.float().clone() for k, v in ep.state_dict.items()}
    save_file(state_dict, str(WEIGHTS_PATH))

    print("Done.")


if __name__ == "__main__":
    main()
