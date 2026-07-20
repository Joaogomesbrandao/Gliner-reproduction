from __future__ import annotations

import argparse
from typing import Any

from src.experiments.common import ExperimentConfig, run_experiment
from src.models.gliner_model import GLiNERModel
from src.utils import configure_logging, ensure_results_dir, format_table, write_rows_to_csv


def run_tweetner7_zero_shot(
    *,
    checkpoint: str = "urchade/gliner_large-v2.1",
    split: str = "test",
    max_examples: int | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Run an additional zero-shot experiment on TweetNER7.

    TweetNER7 appears in the paper's 20-dataset benchmark and is useful here
    because it stresses noisy, tweet-like text that the paper itself highlights
    as harder for GLiNER than several cleaner domains.
    """
    results_dir = ensure_results_dir()
    logger = configure_logging()
    model = GLiNERModel(checkpoint)
    config = ExperimentConfig(
        experiment_name="tweetner7_zero_shot",
        dataset_name="tweetner7_2021",
        split=split,
        max_examples=max_examples,
        threshold=threshold,
    )
    row, _ = run_experiment(
        model=model,
        config=config,
        classification_report_path=results_dir / "tweetner7_classification_report.txt",
        logger=logger,
    )

    write_rows_to_csv(results_dir / "tweetner7_zero_shot.csv", [row])

    print("\n" + "=" * 80)
    print("TWEETNER7 ZERO-SHOT SUMMARY")
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
    parser = argparse.ArgumentParser(description="Run GLiNER zero-shot on TweetNER7.")
    parser.add_argument("--checkpoint", default="urchade/gliner_large-v2.1")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_tweetner7_zero_shot(
        checkpoint=args.checkpoint,
        split=args.split,
        max_examples=args.max_examples,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
