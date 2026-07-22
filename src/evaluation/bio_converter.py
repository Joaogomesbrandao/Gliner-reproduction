from __future__ import annotations

from typing import Any, Mapping, Sequence

TokenSpan = tuple[int, int, str]

_LABEL_MAP = {
    "person": "PER",
    "per": "PER",
    "human": "PER",
    "human being": "PER",
    "organization": "ORG",
    "org": "ORG",
    "company": "ORG",
    "location": "LOC",
    "loc": "LOC",
    "country": "LOC",
    "geographical location": "LOC",
    "gpe": "LOC",
    "fac": "LOC",
    "facility": "LOC",
    "miscellaneous": "MISC",
    "misc": "MISC",
    "other entity": "MISC",
    "miscellaneous named entity": "MISC",
    "norp": "MISC",
    "event": "MISC",
    "language": "MISC",
    "law": "MISC",
    "product": "MISC",
    "work_of_art": "MISC",
}


def normalize_prediction_label(
    label: str,
    label_mapping: Mapping[str, str] | None = None,
) -> str | None:
    """Map model-specific labels to the canonical CoNLL entity types."""
    mapping = label_mapping or _LABEL_MAP
    return mapping.get(label.strip().lower())


def bio_tags_to_spans(tags: Sequence[str]) -> list[TokenSpan]:
    """Convert BIO tags into token spans represented as (start, end, label)."""
    spans: list[TokenSpan] = []
    index = 0

    while index < len(tags):
        tag = tags[index]
        if not tag.startswith("B-"):
            index += 1
            continue

        label = tag[2:]
        start = index
        index += 1

        while index < len(tags) and tags[index] == f"I-{label}":
            index += 1

        spans.append((start, index - 1, label))

    return spans


def token_spans_to_bio(num_tokens: int, spans: Sequence[TokenSpan]) -> list[str]:
    """Convert non-overlapping token spans into BIO tags."""
    bio_tags = ["O"] * num_tokens

    for start, end, label in sorted(spans, key=lambda span: (span[0], span[1], span[2])):
        bio_tags[start] = f"B-{label}"
        for token_idx in range(start + 1, end + 1):
            bio_tags[token_idx] = f"I-{label}"

    return bio_tags


def character_span_to_token_span(
    start: int,
    end: int,
    token_offsets: Sequence[tuple[int, int]],
) -> tuple[int, int] | None:
    """Project a character span from GLiNER back onto token boundaries."""
    overlapping_tokens = [
        index
        for index, (token_start, token_end) in enumerate(token_offsets)
        if token_start < end and token_end > start
    ]

    if not overlapping_tokens:
        return None

    return overlapping_tokens[0], overlapping_tokens[-1]


def predictions_to_token_spans(
    predictions: Sequence[Mapping[str, Any]],
    token_offsets: Sequence[tuple[int, int]],
    label_mapping: Mapping[str, str] | None = None,
) -> list[TokenSpan]:
    """
    Convert GLiNER-style predictions into canonical token spans.

    Predictions whose labels cannot be mapped to the CoNLL schema are ignored so
    evaluation remains faithful to the target dataset.
    """
    spans: list[TokenSpan] = []

    for prediction in predictions:
        label = normalize_prediction_label(str(prediction["label"]), label_mapping=label_mapping)
        if label is None:
            continue

        token_span = character_span_to_token_span(
            start=int(prediction["start"]),
            end=int(prediction["end"]),
            token_offsets=token_offsets,
        )
        if token_span is None:
            continue

        spans.append((token_span[0], token_span[1], label))

    return sorted(set(spans), key=lambda span: (span[0], span[1], span[2]))


def predictions_to_bio(
    tokens: Sequence[str],
    predictions: Sequence[Mapping[str, Any]],
    token_offsets: Sequence[tuple[int, int]],
    label_mapping: Mapping[str, str] | None = None,
) -> list[str]:
    """Compatibility helper used by the qualitative zero-shot script."""
    spans = predictions_to_token_spans(
        predictions,
        token_offsets,
        label_mapping=label_mapping,
    )
    return token_spans_to_bio(len(tokens), spans)
