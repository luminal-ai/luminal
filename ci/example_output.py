import re

ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

EXPECTED_OUTPUT = {
    "gemma4_moe": [
        "city of romance, art and culture",
    ],
    "whisper": [
        "ask not what your country can do for you",
    ],
}

EXPECTED_CONCEPTS = {
    "llama": [
        ["layers"],
        ["neurons", "nodes"],
        ["learn", "learning", "adapt"],
        ["data", "patterns", "features"],
    ],
    "gemma": [
        ["neural network", "neural networks"],
        ["nodes", "neurons"],
        ["layers"],
        ["weights"],
        ["training", "learn", "learns"],
    ],
    "qwen": [
        ["neural network", "neural networks"],
        ["computational model", "computational system"],
        ["brain"],
        ["layers"],
        ["neurons", "nodes"],
        ["learn", "learning", "training"],
    ],
    "qwen3_moe": [
        ["capital"],
        ["france"],
        ["paris"],
    ],
}


def normalize_output(output: str) -> str:
    output = ANSI_ESCAPE.sub("", output)
    output = output.replace("\r", "\n")
    return re.sub(r"\s+", " ", output).casefold()


def validate_output(example: str, output: str):
    normalized_output = normalize_output(output)

    expected_concepts = EXPECTED_CONCEPTS.get(example)
    if expected_concepts is not None:
        missing = [
            concept_group
            for concept_group in expected_concepts
            if not any(normalize_output(term) in normalized_output for term in concept_group)
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
        if normalize_output(phrase) in normalized_output:
            print(f"\nOutput check passed for {example!r}: found {phrase!r}")
            return

    expected = "\n  - ".join(expected_phrases)
    raise AssertionError(
        f"Output check failed for {example!r}. Expected one of:\n  - {expected}"
    )
