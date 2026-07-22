from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from src.datasets.conll2003 import DEFAULT_DATASET_NAME, DEFAULT_GLINER_LABELS
from src.experiments.common import ExperimentConfig, run_experiment
from src.models.gliner_model import GLiNERModel
from src.utils import configure_logging, ensure_results_dir, format_table, write_rows_to_csv


def run_zero_shot_experiment(
    *,
    checkpoint: str = "urchade/gliner_large-v2.1",
    labels: Sequence[str] = DEFAULT_GLINER_LABELS,
    dataset_name: str = DEFAULT_DATASET_NAME,
    split: str = "test",
    max_examples: int | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Evaluate GLiNER zero-shot on CoNLL-2003.

    Relative to the paper, this remains a partial reproduction because we use
    the public pretrained checkpoint instead of re-training GLiNER on Pile-NER.
    The evaluation criterion, however, follows exact span/type matching.
    """
    results_dir = ensure_results_dir()
    logger = configure_logging()
    model = GLiNERModel(checkpoint)
    config = ExperimentConfig(
        experiment_name="zero_shot",
        dataset_name=dataset_name,
        split=split,
        labels=tuple(labels),
        max_examples=max_examples,
        threshold=threshold,
    )

    row, _ = run_experiment(
        model=model,
        config=config,
        classification_report_path=results_dir / "classification_report.txt",
        logger=logger,
    )

    output_path = results_dir / "zero_shot.csv"
    write_rows_to_csv(output_path, [row])

    print("\n" + "=" * 80)
    print("ZERO-SHOT SUMMARY")
    print("=" * 80)
    print(
        format_table(
            [row],
            [
                "model_name",
                "dataset",
                "split",
                "examples",
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

    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GLiNER zero-shot experiment.")
    parser.add_argument("--checkpoint", default="urchade/gliner_large-v2.1")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--labels",
        nargs="+",
        default=list(DEFAULT_GLINER_LABELS),
        help="Entity prompts passed to GLiNER.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_zero_shot_experiment(
        checkpoint=args.checkpoint,
        labels=args.labels,
        dataset_name=args.dataset,
        split=args.split,
        max_examples=args.max_examples,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
