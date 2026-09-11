"""Generate independent, real-checkpoint chat/logit fixtures with Transformers.

Run this before the Rust validator so the two runtimes need not share GPU memory.
The output records dependency versions and checkpoint revision for reproducibility.
"""

import argparse
import json
from pathlib import Path

import safetensors
import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer


CONVERSATIONS = [
    (
        "arithmetic",
        [
            "What is 2 + 2? Answer with the number only.",
            "Now multiply that result by 3. Answer with the number only.",
        ],
    ),
    (
        "memory",
        [
            "My dog's name is Pixel. Remember it. Reply briefly.",
            "What is my dog's name? Reply with the name only.",
        ],
    ),
    (
        "translation",
        [
            "Translate 'good morning' into French. Reply briefly.",
            "Now translate 'thank you' into French. Reply briefly.",
        ],
    ),
    ("reset_replay", ["What is 2 + 2? Answer with the number only."]),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model", choices=["llama3", "qwen3", "gemma3", "qwen3-moe"], required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    factory = (
        AutoModelForImageTextToText if args.model == "gemma3" else AutoModelForCausalLM
    )
    print(f"Loading {args.checkpoint} in F32 with eager attention", flush=True)
    model = factory.from_pretrained(
        args.checkpoint,
        local_files_only=True,
        dtype=torch.float32,
        device_map=args.device,
        attn_implementation="eager",
    ).eval()
    config = json.loads((args.checkpoint / "config.json").read_text())
    stops = set()
    for value in [
        config.get("eos_token_id"),
        config.get("text_config", {}).get("eos_token_id"),
        model.generation_config.eos_token_id,
        tokenizer.eos_token_id,
    ]:
        stops.update(value if isinstance(value, list) else [] if value is None else [value])
    revision = args.checkpoint / "REVISION"
    suite = {
        "model": args.model,
        "checkpoint_revision": revision.read_text().strip() if revision.exists() else None,
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "safetensors": safetensors.__version__,
        },
        "dtype": "float32",
        "attention": "eager",
        "tf32": False,
        "max_new_tokens": args.max_new_tokens,
        "stop_tokens": sorted(stops),
        "cases": [],
    }
    with torch.inference_mode():
        for conversation, prompts in CONVERSATIONS:
            messages = []
            for turn, prompt in enumerate(prompts):
                messages.append({"role": "user", "content": prompt})
                ids = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                inputs = torch.tensor([ids], dtype=torch.long, device=args.device)
                generated = model.generate(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=sorted(stops),
                    pad_token_id=(
                        tokenizer.pad_token_id
                        if tokenizer.pad_token_id is not None
                        else tokenizer.eos_token_id
                    ),
                    use_cache=True,
                    disable_compile=True,
                )
                tokens = generated[0, len(ids):].tolist()
                # A full causal pass independently supplies the expected row at
                # every possible chunk boundary, including the consumed EOS.
                output = model(
                    input_ids=generated,
                    attention_mask=torch.ones_like(generated),
                    use_cache=False,
                )
                logits = output.logits[0].float().cpu().contiguous()
                assert torch.isfinite(logits).all(), "non-finite reference logits"
                expected_greedy = (
                    logits[len(ids) - 1 : len(ids) + len(tokens) - 1].argmax(-1).tolist()
                )
                if expected_greedy != tokens:
                    raise AssertionError(
                        "Transformers cached generation disagrees with full causal replay"
                    )
                name = f"{conversation}_{turn + 1}"
                filename = f"{name}.safetensors"
                save_file({"logits": logits}, args.output / filename)
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                suite["cases"].append(
                    {
                        "name": name,
                        "reset": turn == 0,
                        "messages": list(messages),
                        "prompt_tokens": ids,
                        "generated_tokens": tokens,
                        "text": text,
                        "logits_file": filename,
                    }
                )
                print(
                    f"{name}: {text!r} ({len(ids)} prompt / {len(tokens)} generated)",
                    flush=True,
                )
                messages.append({"role": "assistant", "content": text})
                del output, logits, generated
    suite["max_context"] = (
        max(len(c["prompt_tokens"]) + len(c["generated_tokens"]) for c in suite["cases"])
        + 8
    )
    (args.output / "suite.json").write_text(
        json.dumps(suite, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"Saved {len(suite['cases'])} cases to {args.output}", flush=True)


if __name__ == "__main__":
    main()
