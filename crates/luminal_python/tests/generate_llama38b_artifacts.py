"""Generate pre-computed artifacts for test_hf_llama38b_cached_onnx.

Run once:
    uv run python tests/generate_llama38b_artifacts.py

Produces:
    tests/llama38b.onnx          — ONNX export of Llama 3.1-8B
    tests/llama38b_ref_logits.pt — reference logits for input_ids=[1,2,3,4]
"""

from pathlib import Path

import torch
from transformers import AutoConfig, LlamaForCausalLM

SCRIPT_DIR = Path(__file__).resolve().parent
ONNX_PATH = SCRIPT_DIR / "llama38b.onnx"
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

    print("Computing reference logits...")
    with torch.no_grad():
        ref_logits = model(INPUT_IDS).logits.clone()
    print(f"Reference logits shape: {ref_logits.shape}")

    print(f"Saving reference logits to {LOGITS_PATH}")
    torch.save(ref_logits, LOGITS_PATH)

    print(f"Exporting ONNX to {ONNX_PATH}")
    torch.onnx.export(
        model,
        (INPUT_IDS,),
        str(ONNX_PATH),
        opset_version=20,
        input_names=["input_ids"],
        output_names=["logits"],
    )

    print("Done.")


if __name__ == "__main__":
    main()
