from __future__ import annotations

import argparse
from typing import Any

from src.datasets.conll2003 import DEFAULT_DATASET_NAME
from src.experiments.common import ExperimentConfig, run_experiment
from src.models.gliner_model import GLiNERModel
from src.utils import configure_logging, ensure_results_dir, format_table, write_rows_to_csv

LABEL_PROMPT_CONFIGS = {
    "natural_labels": ("person", "organization", "location", "miscellaneous"),
    "conll_tags": ("PER", "ORG", "LOC", "MISC"),
    "semantic_aliases": ("human", "company", "country", "other entity"),
    "descriptive_phrases": (
        "human being",
        "organization",
        "geographical location",
        "miscellaneous named entity",
    ),
}


def run_label_prompt_experiment(
    *,
    checkpoint: str = "urchade/gliner_large-v2.1",
    dataset_name: str = DEFAULT_DATASET_NAME,
    split: str = "test",
    max_examples: int | None = None,
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Compare alternative entity descriptions for the same zero-shot GLiNER model.

    This approximates the paper's prompt-conditioned setup by changing only the
    entity type text while preserving the same underlying checkpoint.
    """
    results_dir = ensure_results_dir()
    logger = configure_logging()
    model = GLiNERModel(checkpoint)
    rows: list[dict[str, Any]] = []

    for prompt_name, labels in LABEL_PROMPT_CONFIGS.items():
        config = ExperimentConfig(
            experiment_name="label_prompt",
            dataset_name=dataset_name,
            split=split,
            labels=labels,
            max_examples=max_examples,
            threshold=threshold,
        )
        row, _ = run_experiment(
            model=model,
            config=config,
            logger=logger,
        )
        row["prompt_name"] = prompt_name
        rows.append(row)

    rows = sorted(rows, key=lambda row: row["prompt_name"])
    write_rows_to_csv(results_dir / "label_prompt.csv", rows)

    print("\n" + "=" * 80)
    print("LABEL PROMPT SUMMARY")
    print("=" * 80)
    print(
        format_table(
            rows,
            [
                "prompt_name",
                "precision",
                "recall",
                "f1",
                "time_seconds_total",
                "time_seconds_average",
                "predicted_entities",
                "reference_entities",
            ],
        )
    )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GLiNER label prompt experiment.")
    parser.add_argument("--checkpoint", default="urchade/gliner_large-v2.1")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_label_prompt_experiment(
        checkpoint=args.checkpoint,
        dataset_name=args.dataset,
        split=args.split,
        max_examples=args.max_examples,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
