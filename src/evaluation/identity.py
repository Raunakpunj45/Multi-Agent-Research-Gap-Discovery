from collections import Counter


def validate_exact_paper_ids(observed: list[str], expected: list[str], context: str) -> None:
    """Require a complete, unique six-paper evaluation identity set."""
    if not isinstance(observed, list) or not isinstance(expected, list):
        raise ValueError(f"{context} must provide paper IDs as lists.")
    if len(expected) != 6 or len(set(expected)) != 6:
        raise ValueError("The frozen evaluation identity set must contain six unique paper IDs.")
    if any(not isinstance(paper_id, str) or not paper_id for paper_id in observed):
        raise ValueError(f"{context} contains an empty or invalid paper ID.")

    counts = Counter(observed)
    duplicates = sorted(paper_id for paper_id, count in counts.items() if count > 1)
    expected_set = set(expected)
    observed_set = set(observed)
    missing = sorted(expected_set - observed_set)
    unexpected = sorted(observed_set - expected_set)
    if duplicates or missing or unexpected or len(observed) != 6:
        raise ValueError(
            f"{context} must contain exactly the six frozen paper IDs; "
            f"missing={missing}, duplicate={duplicates}, unexpected={unexpected}."
        )
