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
