from __future__ import annotations

from typing import Any, Sequence

from datasets import Dataset, DatasetDict, load_dataset

from src.datasets.sequence_labeling import SequenceLabelingSentence, build_sentence as build_generic_sentence
from src.datasets.sequence_labeling import reconstruct_text_with_offsets
from src.utils import DATA_DIR

DEFAULT_DATASET_NAME = "conll2003"
DEFAULT_GLINER_LABELS = (
    "person",
    "organization",
    "location",
    "miscellaneous",
)
CONLL_ENTITY_TYPES = ("PER", "ORG", "LOC", "MISC")

ConllSentence = SequenceLabelingSentence


def load_conll2003() -> DatasetDict:
    """Load CoNLL-2003 from the local cache directory used by this project."""
    return load_dataset(
        DEFAULT_DATASET_NAME,
        cache_dir=str(DATA_DIR),
        trust_remote_code=True,
    )


def get_ner_labels(dataset: DatasetDict) -> list[str]:
    """Return the NER label names in BIO format."""
    return dataset["train"].features["ner_tags"].feature.names


def get_split(dataset: DatasetDict, split: str, max_examples: int | None = None) -> Dataset:
    """Return a dataset split, optionally truncated for fast experiments."""
    samples = dataset[split]
    if max_examples is None:
        return samples

    capped_size = min(max_examples, len(samples))
    return samples.select(range(capped_size))


def build_sentence(sample: dict[str, Any], ner_labels: Sequence[str]) -> ConllSentence:
    """Convert a raw CoNLL sample into the text/span representation used here."""
    return build_generic_sentence(
        sample,
        label_names=ner_labels,
        tag_field="ner_tags",
    )


def main() -> None:
    dataset = load_conll2003()
    ner_labels = get_ner_labels(dataset)
    sample = build_sentence(dataset["train"][0], ner_labels)

    print("=" * 60)
    print("Quantidade de exemplos")
    print("=" * 60)
    print(f"Train      : {len(dataset['train'])}")
    print(f"Validation : {len(dataset['validation'])}")
    print(f"Test       : {len(dataset['test'])}")

    print("\n" + "=" * 60)
    print("NER Labels")
    print("=" * 60)
    for idx, label in enumerate(ner_labels):
        print(f"{idx}: {label}")

    print("\n" + "=" * 60)
    print("Primeira sentença reconstruída")
    print("=" * 60)
    print(sample.text)
    print(sample.offsets)


if __name__ == "__main__":
    main()
