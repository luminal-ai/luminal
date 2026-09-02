import re

ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

# Performance gates for the Modal examples CI (A100-80GB).
#
# Thresholds are derived from the four green main/PR runs preceding PR #373
# (2026-07-16 through 2026-07-19): the worst observed value across those runs,
# plus headroom for Modal variance. TTFT is noisy run-to-run (~1.7x worst
# observed); TPOT is stable (~1.3x worst observed). Whisper reports only
# decode throughput, so it gates on a minimum tok/s.
#
# Baselines (min-max over those runs):
#   llama       TTFT  34-62 ms   TPOT 10.8-11.6 ms
#   gemma       TTFT 127-174 ms  TPOT 15.4-17.2 ms
#   qwen        TTFT  77-106 ms  TPOT 12.4-16.8 ms
#   qwen3_moe   TTFT 109-151 ms  TPOT  9.0-18.3 ms
#   gemma4_moe  TTFT 122-137 ms  TPOT 27.4-34.2 ms
#   whisper     7.1 tok/s
PERF_GATES = {
    # Calibrated to fully-unrolled candidate profiling (search ranks
    # deployment graphs directly). GH200 references: llama 71/6.4,
    # gemma 74/10.0, qwen 99/6.7, qwen3_moe 74/7.4, gemma4_moe 149/10.8.
    "llama": {"max_ttft_ms": 150.0, "max_tpot_ms": 15.0},
    "gemma": {"max_ttft_ms": 300.0, "max_tpot_ms": 30.0},
    "qwen": {"max_ttft_ms": 180.0, "max_tpot_ms": 22.0},
    "qwen3_moe": {"max_ttft_ms": 1000.0, "max_tpot_ms": 50.0},
    # gemma4_moe's decode search still has run-to-run family variance
    # (A100 draws observed 25-52 ms TPOT vs 10.8 best; exploration work
    # tracked separately) — gate above the draw spread, below the
    # 100+ ms failure modes.
    "gemma4_moe": {"max_ttft_ms": 450.0, "max_tpot_ms": 60.0},
    "whisper": {"min_tps": 5.0},
}

EXPECTED_OUTPUT = {
    "whisper": [
        "ask not what your country can do for you",
    ],
}

EXPECTED_CONCEPTS = {
    "llama": [
        ["layers"],
        ["neurons", "nodes"],
        ["learn", "learns", "learning", "learned", "adapt", "adapts", "adaptation"],
        ["data", "patterns", "features"],
    ],
    "gemma": [
        ["neural network", "neural networks"],
        ["nodes", "neurons"],
        ["layers"],
        ["weights"],
        ["training", "learn", "learns", "learning", "learned"],
    ],
    "qwen": [
        ["neural network", "neural networks"],
        ["computational model", "computational system"],
        ["brain"],
        ["layers"],
        ["neurons", "nodes"],
        ["learn", "learns", "learning", "learned", "training"],
    ],
    "qwen3_moe": [
        ["capital"],
        ["france"],
        ["paris"],
    ],
    "gemma4_moe": [
        ["paris"],
        ["romance", "art", "culture"],
    ],
}


def normalize_output(output: str) -> str:
    output = ANSI_ESCAPE.sub("", output)
    output = output.replace("\r", "\n")
    return re.sub(r"\s+", " ", output).casefold()


def contains_term(normalized_output: str, term: str) -> bool:
    """Match a word or phrase without accepting it inside a larger word."""
    normalized_term = normalize_output(term)
    return re.search(
        rf"(?<![^\W_]){re.escape(normalized_term)}(?![^\W_])", normalized_output
    ) is not None


def validate_output(example: str, output: str):
    normalized_output = normalize_output(output)

    expected_concepts = EXPECTED_CONCEPTS.get(example)
    if expected_concepts is not None:
        missing = [
            concept_group
            for concept_group in expected_concepts
            if not any(contains_term(normalized_output, term) for term in concept_group)
        ]
        if missing:
            expected = "\n  - ".join(" / ".join(group) for group in expected_concepts)
            missing_terms = "\n  - ".join(" / ".join(group) for group in missing)
            raise AssertionError(
                f"Output check failed for {example!r}.\n"
                f"Expected concept groups:\n  - {expected}\n"
                f"Missing concept groups:\n  - {missing_terms}"
            )

        expected = ", ".join(" / ".join(group) for group in expected_concepts)
        print(f"\nOutput check passed for {example!r}: found concepts {expected}")
        return

    expected_phrases = EXPECTED_OUTPUT.get(example)
    if expected_phrases is None:
        raise ValueError(f"No expected output phrases configured for example {example!r}")

    for phrase in expected_phrases:
        if contains_term(normalized_output, phrase):
            print(f"\nOutput check passed for {example!r}: found {phrase!r}")
            return

    expected = "\n  - ".join(expected_phrases)
    raise AssertionError(
        f"Output check failed for {example!r}. Expected one of:\n  - {expected}"
    )


def parse_perf_metrics(output: str) -> dict[str, float | None]:
    """Parse TTFT/TPOT/tok/s from an example's stdout."""
    metrics: dict[str, float | None] = {"ttft_ms": None, "tpot_ms": None, "tps": None}
    for line in output.splitlines():
        if "TTFT attribution" in line:
            continue
        if "TTFT:" in line:
            metrics["ttft_ms"] = parse_number_after(line, "TTFT:") or metrics["ttft_ms"]
        if "TPOT:" in line:
            metrics["tpot_ms"] = parse_number_after(line, "TPOT:") or metrics["tpot_ms"]
        if "tok/s" in line:
            metrics["tps"] = parse_tok_per_second(line) or metrics["tps"]
    if metrics["tps"] is None and metrics["tpot_ms"]:
        metrics["tps"] = 1000.0 / metrics["tpot_ms"]
    return metrics


def parse_number_after(line: str, marker: str) -> float | None:
    tail = line.split(marker, 1)[1].lstrip()
    chars = []
    for char in tail:
        if char.isdigit() or char == ".":
            chars.append(char)
        else:
            break
    if not chars:
        return None
    return float("".join(chars))


def parse_tok_per_second(line: str) -> float | None:
    head = line.split("tok/s", 1)[0]
    parts = head.split()
    if not parts:
        return None
    try:
        return float(parts[-1].strip("("))
    except ValueError:
        return None


def validate_perf(example: str, output: str):
    gate = PERF_GATES.get(example)
    if gate is None:
        raise ValueError(f"No performance gate configured for example {example!r}")

    metrics = parse_perf_metrics(output)
    failures = []
    checks = []

    for metric_key, gate_key, worse_if_higher, unit in (
        ("ttft_ms", "max_ttft_ms", True, "ms"),
        ("tpot_ms", "max_tpot_ms", True, "ms"),
        ("tps", "min_tps", False, "tok/s"),
    ):
        limit = gate.get(gate_key)
        if limit is None:
            continue
        value = metrics[metric_key]
        bound = "<=" if worse_if_higher else ">="
        if value is None:
            failures.append(f"{metric_key} not found in output (limit {bound} {limit} {unit})")
        elif (value > limit) if worse_if_higher else (value < limit):
            failures.append(f"{metric_key} {value:.2f} {unit} (limit {bound} {limit} {unit})")
        else:
            checks.append(f"{metric_key} {value:.2f} {bound} {limit} {unit}")

    if failures:
        details = "\n  - ".join(failures)
        raise AssertionError(
            f"Performance check failed for {example!r}:\n  - {details}"
        )
    print(f"\nPerformance check passed for {example!r}: {'; '.join(checks)}")
