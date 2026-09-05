import re

ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

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
