from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download

from src.datasets.conll2003 import (
    DEFAULT_DATASET_NAME,
    DEFAULT_GLINER_LABELS,
    get_ner_labels,
    get_split,
    load_conll2003,
)
from src.datasets.sequence_labeling import (
    DatasetResources,
    SequenceLabelingSentence,
    build_prediction_label_mapping,
    build_sentence,
)


def _apply_max_examples(
    samples: Sequence[Mapping[str, Any]],
    max_examples: int | None,
) -> Sequence[Mapping[str, Any]]:
    if max_examples is None:
        return samples
    if isinstance(samples, Dataset):
        return samples.select(range(min(max_examples, len(samples))))
    return samples[:max_examples]


def _load_json_lines(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file_handle:
        for line in file_handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _invert_label_map(label_map: Mapping[str, int]) -> list[str]:
    return [label for label, _ in sorted(label_map.items(), key=lambda item: item[1])]


def _canonical_labels_from_bio(label_names: Sequence[str]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()

    for label_name in label_names:
        if label_name == "O":
            continue
        canonical_label = label_name.split("-", 1)[1]
        if canonical_label not in seen:
            seen.add(canonical_label)
            labels.append(canonical_label)

    return labels


def _canonical_labels_from_dataset_samples(
    dataset_dict: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    label_names: Sequence[str],
    tag_field: str,
) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()

    for split_name in ("train", "validation", "test"):
        if split_name not in dataset_dict:
            continue
        for sample in dataset_dict[split_name]:
            for tag_id in sample[tag_field]:
                label_name = label_names[tag_id]
                if label_name == "O":
                    continue
                canonical_label = label_name.split("-", 1)[1]
                if canonical_label not in seen:
                    seen.add(canonical_label)
                    labels.append(canonical_label)

    return labels


def _load_tner_dataset_resources(
    *,
    dataset_name: str,
    repo_id: str,
    split: str,
    split_to_file: Mapping[str, str],
    max_examples: int | None,
    prompt_overrides: Mapping[str, str] | None = None,
    extra_aliases: Mapping[str, str] | None = None,
) -> DatasetResources:
    snapshot_path = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir_use_symlinks=False,
        )
    )
    base_dir = snapshot_path / "dataset"
    label_names = _invert_label_map(json.loads((base_dir / "label.json").read_text()))
    canonical_labels = _canonical_labels_from_bio(label_names)
    prompt_labels, prediction_label_mapping = build_prediction_label_mapping(
        canonical_labels,
        prompt_overrides=prompt_overrides,
        extra_aliases=extra_aliases,
    )
    samples = _load_json_lines(base_dir / split_to_file[split])
    samples = _apply_max_examples(samples, max_examples)
    sentence_builder = partial(build_sentence, label_names=label_names, tag_field="tags")
    return DatasetResources(
        dataset_name=dataset_name,
        split=split,
        samples=samples,
        build_sentence=sentence_builder,
        default_prompt_labels=prompt_labels,
        prediction_label_mapping=prediction_label_mapping,
    )


def _load_conll2003_resources(
    *,
    split: str,
    max_examples: int | None,
) -> DatasetResources:
    dataset = load_conll2003()
    label_names = get_ner_labels(dataset)
    samples = get_split(dataset, split, max_examples)
    prompt_labels, prediction_label_mapping = build_prediction_label_mapping(
        ("PER", "ORG", "LOC", "MISC"),
        prompt_overrides={
            "PER": "person",
            "ORG": "organization",
            "LOC": "location",
            "MISC": "miscellaneous",
        },
        extra_aliases={
            "per": "PER",
            "org": "ORG",
            "loc": "LOC",
            "misc": "MISC",
            "human": "PER",
            "human being": "PER",
            "company": "ORG",
            "country": "LOC",
            "geographical location": "LOC",
            "other entity": "MISC",
            "miscellaneous named entity": "MISC",
            "gpe": "LOC",
            "fac": "LOC",
            "facility": "LOC",
            "norp": "MISC",
            "language": "MISC",
            "law": "MISC",
            "event": "MISC",
            "product": "MISC",
            "work_of_art": "MISC",
        },
    )
    sentence_builder = partial(build_sentence, label_names=label_names, tag_field="ner_tags")
    return DatasetResources(
        dataset_name=DEFAULT_DATASET_NAME,
        split=split,
        samples=samples,
        build_sentence=sentence_builder,
        default_prompt_labels=prompt_labels or tuple(DEFAULT_GLINER_LABELS),
        prediction_label_mapping=prediction_label_mapping,
    )


def _load_crossner_resources(
    *,
    domain: str,
    split: str,
    max_examples: int | None,
) -> DatasetResources:
    snapshot_path = Path(
        snapshot_download(
            repo_id="DFKI-SLT/cross_ner",
            repo_type="dataset",
            local_dir_use_symlinks=False,
        )
    )
    dataset = load_dataset(str(snapshot_path / "cross_ner.py"), domain, trust_remote_code=True)
    label_names = dataset["train"].features["ner_tags"].feature.names
    samples = get_split(dataset, split, max_examples)
    canonical_labels = _canonical_labels_from_dataset_samples(
        dataset,
        label_names=label_names,
        tag_field="ner_tags",
    )
    prompt_labels, prediction_label_mapping = build_prediction_label_mapping(canonical_labels)
    sentence_builder = partial(build_sentence, label_names=label_names, tag_field="ner_tags")
    return DatasetResources(
        dataset_name=f"crossner_{domain}",
        split=split,
        samples=samples,
        build_sentence=sentence_builder,
        default_prompt_labels=prompt_labels,
        prediction_label_mapping=prediction_label_mapping,
    )


def load_dataset_resources(
    dataset_name: str,
    *,
    split: str = "test",
    max_examples: int | None = None,
) -> DatasetResources:
    """Dataset registry used by the experiment pipeline."""
    if dataset_name == DEFAULT_DATASET_NAME:
        return _load_conll2003_resources(split=split, max_examples=max_examples)

    if dataset_name == "mit_restaurant":
        return _load_tner_dataset_resources(
            dataset_name=dataset_name,
            repo_id="tner/mit_restaurant",
            split=split,
            split_to_file={
                "train": "train.json",
                "validation": "valid.json",
                "test": "test.json",
            },
            max_examples=max_examples,
        )

    if dataset_name == "mit_movie":
        return _load_tner_dataset_resources(
            dataset_name=dataset_name,
            repo_id="tner/mit_movie_trivia",
            split=split,
            split_to_file={
                "train": "train.json",
                "validation": "valid.json",
                "test": "test.json",
            },
            max_examples=max_examples,
        )

    if dataset_name in {
        "crossner_ai",
        "crossner_literature",
        "crossner_music",
        "crossner_politics",
        "crossner_science",
    }:
        domain = dataset_name.split("_", 1)[1]
        return _load_crossner_resources(domain=domain, split=split, max_examples=max_examples)

    if dataset_name == "tweetner7_2021":
        return _load_tner_dataset_resources(
            dataset_name=dataset_name,
            repo_id="tner/tweetner7",
            split=split,
            split_to_file={
                "train": "2021.train.json",
                "validation": "2021.dev.json",
                "test": "2021.test.json",
            },
            max_examples=max_examples,
        )

    raise ValueError(
        f"Dataset '{dataset_name}' is not registered. "
        "Available datasets: conll2003, mit_restaurant, mit_movie, "
        "crossner_ai, crossner_literature, crossner_music, "
        "crossner_politics, crossner_science, tweetner7_2021."
    )
