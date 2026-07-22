from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


@dataclass(frozen=True)
class SequenceLabelingSentence:
    """Text-plus-span view used by the zero-shot evaluation pipeline."""

    sentence_id: str
    tokens: tuple[str, ...]
    text: str
    offsets: tuple[tuple[int, int], ...]
    gold_bio_tags: tuple[str, ...]


@dataclass(frozen=True)
class DatasetResources:
    """
    Runtime bundle for a sequence-labeling benchmark.

    ``prediction_label_mapping`` maps prompt labels or model labels back to the
    canonical dataset entity type so span exact-match evaluation remains stable.
    """

    dataset_name: str
    split: str
    samples: Sequence[Mapping[str, Any]]
    build_sentence: Callable[[Mapping[str, Any]], SequenceLabelingSentence]
    default_prompt_labels: tuple[str, ...]
    prediction_label_mapping: dict[str, str]


_NO_SPACE_BEFORE = {
    ".",
    ",",
    ":",
    ";",
    "!",
    "?",
    "%",
    ")",
    "]",
    "}",
    "'s",
    "'m",
    "'re",
    "'ve",
    "'d",
    "'ll",
    "n't",
}
_NO_SPACE_AFTER = {"(", "[", "{", "$", "``"}
_DOUBLE_QUOTES = {'"', "''", "``"}

_PROMPT_OVERRIDES = {
    "PER": "person",
    "ORG": "organization",
    "LOC": "location",
    "MISC": "miscellaneous",
    "misc": "miscellaneous",
    "academicjournal": "academic journal",
    "astronomicalobject": "astronomical object",
    "chemicalcompound": "chemical compound",
    "chemicalelement": "chemical element",
    "creative_work": "creative work",
    "character_name": "character name",
    "literarygenre": "literary genre",
    "musicalartist": "musical artist",
    "musicalinstrument": "musical instrument",
    "musicgenre": "music genre",
    "organisation": "organization",
    "politicalparty": "political party",
    "programlang": "programming language",
    "restaurant_name": "restaurant name",
}


def reconstruct_text_with_offsets(tokens: Sequence[str]) -> tuple[str, list[tuple[int, int]]]:
    """
    Rebuild sentence text from tokens while preserving token character offsets.

    This is an approximation of the raw sentence required by GLiNER. It is more
    faithful than joining all tokens with spaces because it keeps punctuation and
    contractions aligned with the character spans returned by the public API.
    """
    if not tokens:
        return "", []

    parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    previous_token = ""
    open_double_quote = False

    for token in tokens:
        prefix = ""
        if parts:
            insert_space = True
            if token in _NO_SPACE_BEFORE:
                insert_space = False
            elif previous_token in _NO_SPACE_AFTER:
                insert_space = False
            elif token in _DOUBLE_QUOTES:
                insert_space = not open_double_quote
            prefix = " " if insert_space else ""

        start = cursor + len(prefix)
        end = start + len(token)

        if prefix:
            parts.append(prefix)
        parts.append(token)
        offsets.append((start, end))

        cursor = end
        previous_token = token
        if token in _DOUBLE_QUOTES:
            open_double_quote = not open_double_quote

    return "".join(parts), offsets


def humanize_entity_label(label: str) -> str:
    """Convert dataset-specific entity names into prompt-friendly text."""
    normalized = label.strip()
    if normalized in _PROMPT_OVERRIDES:
        return _PROMPT_OVERRIDES[normalized]

    lower_normalized = normalized.lower()
    if lower_normalized in _PROMPT_OVERRIDES:
        return _PROMPT_OVERRIDES[lower_normalized]

    text = normalized.replace("_", " ").replace("-", " ")
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return _PROMPT_OVERRIDES.get(text, text)


def build_prediction_label_mapping(
    canonical_labels: Sequence[str],
    *,
    prompt_overrides: Mapping[str, str] | None = None,
    extra_aliases: Mapping[str, str] | None = None,
) -> tuple[tuple[str, ...], dict[str, str]]:
    """
    Create prompt labels and a reverse mapping for evaluation.

    The returned mapping accepts both canonical dataset labels and prompt labels.
    """
    prompt_labels: list[str] = []
    prediction_label_mapping: dict[str, str] = {}

    for canonical_label in canonical_labels:
        prompt_label = (
            prompt_overrides[canonical_label]
            if prompt_overrides and canonical_label in prompt_overrides
            else humanize_entity_label(canonical_label)
        )
        prompt_labels.append(prompt_label)

        for alias in {
            canonical_label,
            canonical_label.lower(),
            prompt_label,
            prompt_label.lower(),
        }:
            prediction_label_mapping[alias.lower()] = canonical_label

    if extra_aliases is not None:
        for alias, canonical_label in extra_aliases.items():
            prediction_label_mapping[alias.lower()] = canonical_label

    return tuple(prompt_labels), prediction_label_mapping


def build_sentence(
    sample: Mapping[str, Any],
    *,
    label_names: Sequence[str],
    tag_field: str = "ner_tags",
    id_field: str = "id",
) -> SequenceLabelingSentence:
    """Convert a token-level sample into the span-aware sentence representation."""
    tokens = tuple(sample["tokens"])
    text, offsets = reconstruct_text_with_offsets(tokens)
    gold_bio_tags = tuple(label_names[tag_id] for tag_id in sample[tag_field])
    sentence_id = str(sample.get(id_field, ""))
    return SequenceLabelingSentence(
        sentence_id=sentence_id,
        tokens=tokens,
        text=text,
        offsets=tuple(offsets),
        gold_bio_tags=gold_bio_tags,
    )
