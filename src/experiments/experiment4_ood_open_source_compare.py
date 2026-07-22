from __future__ import annotations

import argparse
from collections import OrderedDict
from typing import Any, Mapping, Sequence

from src.experiments.common import ExperimentConfig, run_experiment
from src.models.gliner_model import GLiNERModel
from src.utils import configure_logging, ensure_results_dir, format_table, write_rows_to_csv

OOD_DATASETS = OrderedDict(
    [
        ("Movie", "mit_movie"),
        ("Restaurant", "mit_restaurant"),
        ("AI", "crossner_ai"),
        ("Literature", "crossner_literature"),
        ("Music", "crossner_music"),
        ("Politics", "crossner_politics"),
        ("Science", "crossner_science"),
    ]
)

PAPER_BASELINES = {
    "GLiNER-L (paper)": {
        "Movie": 57.2,
        "Restaurant": 42.9,
        "AI": 57.2,
        "Literature": 64.4,
        "Music": 69.6,
        "Politics": 72.6,
        "Science": 62.6,
        "Average": 60.9,
    },
    "UniNER-7B": {
        "Movie": 42.4,
        "Restaurant": 31.7,
        "AI": 53.6,
        "Literature": 59.3,
        "Music": 67.0,
        "Politics": 60.9,
        "Science": 61.1,
        "Average": 53.7,
    },
    "GoLLIE": {
        "Movie": 63.0,
        "Restaurant": 43.4,
        "AI": 59.1,
        "Literature": 62.7,
        "Music": 67.8,
        "Politics": 57.2,
        "Science": 55.5,
        "Average": 58.0,
    },
    "Vicuna-7B": {
        "Movie": 6.0,
        "Restaurant": 5.3,
        "AI": 12.8,
        "Literature": 16.1,
        "Music": 17.0,
        "Politics": 20.5,
        "Science": 13.0,
        "Average": 13.0,
    },
    "Vicuna-13B": {
        "Movie": 0.9,
        "Restaurant": 0.4,
        "AI": 22.7,
        "Literature": 22.7,
        "Music": 26.6,
        "Politics": 27.0,
        "Science": 22.0,
        "Average": 17.5,
    },
    "InstructUIE": {
        "Movie": 63.0,
        "Restaurant": 21.0,
        "AI": 49.0,
        "Literature": 47.2,
        "Music": 53.2,
        "Politics": 48.1,
        "Science": 49.2,
        "Average": 47.2,
    },
    "UniNER-13B": {
        "Movie": 48.7,
        "Restaurant": 36.2,
        "AI": 54.2,
        "Literature": 60.9,
        "Music": 64.5,
        "Politics": 61.4,
        "Science": 63.5,
        "Average": 55.6,
    },
}

DEFAULT_OPEN_SOURCE_BASELINES = ("UniNER-7B", "GoLLIE")


def _build_paper_comparison_rows(
    gliner_scores: Mapping[str, float],
    baseline_names: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "model_name": "GLiNER (our reproduction)",
            "source": "local_reproduction",
            **gliner_scores,
        }
    )
    rows.append(
        {
            "model_name": "GLiNER-L (paper)",
            "source": "paper_reported",
            **PAPER_BASELINES["GLiNER-L (paper)"],
        }
    )
    for baseline_name in baseline_names:
        rows.append(
            {
                "model_name": baseline_name,
                "source": "paper_reported",
                **PAPER_BASELINES[baseline_name],
            }
        )
    return rows


def run_ood_open_source_comparison(
    *,
    checkpoint: str = "urchade/gliner_large-v2.1",
    split: str = "test",
    max_examples: int | None = None,
    threshold: float = 0.5,
    paper_baselines: Sequence[str] = DEFAULT_OPEN_SOURCE_BASELINES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Reproduce the OOD benchmark domains used in Table 1 for GLiNER locally.

    The two comparison baselines are paper-reported open-source models. This is
    intentional: reproducing those 7B instruction-tuned systems is outside the
    scope of the current repository, while still keeping the paper-style table
    informative and scientifically honest.
    """
    if len(paper_baselines) > 2:
        raise ValueError("At most two open-source baselines can be compared at once.")

    invalid = [name for name in paper_baselines if name not in PAPER_BASELINES]
    if invalid:
        raise ValueError(f"Unsupported paper baselines: {invalid}")

    results_dir = ensure_results_dir()
    logger = configure_logging()
    model = GLiNERModel(
        checkpoint,
        model_name=f"GLiNER ({checkpoint})",
    )
    domain_rows: list[dict[str, Any]] = []

    for domain_name, dataset_name in OOD_DATASETS.items():
        config = ExperimentConfig(
            experiment_name="ood_open_source_compare",
            dataset_name=dataset_name,
            split=split,
            max_examples=max_examples,
            threshold=threshold,
        )
        row, _ = run_experiment(model=model, config=config, logger=logger)
        row["domain"] = domain_name
        domain_rows.append(row)

    write_rows_to_csv(results_dir / "ood_benchmark_gliner.csv", domain_rows)

    gliner_scores = {
        row["domain"]: round(row["f1"] * 100, 1)
        for row in domain_rows
    }
    gliner_scores["Average"] = round(
        sum(row["f1"] for row in domain_rows) / len(domain_rows) * 100,
        1,
    )
    comparison_rows = _build_paper_comparison_rows(gliner_scores, paper_baselines)
    write_rows_to_csv(results_dir / "ood_open_source_compare.csv", comparison_rows)

    print("\n" + "=" * 80)
    print("OOD BENCHMARK - GLiNER")
    print("=" * 80)
    print(
        format_table(
            domain_rows,
            [
                "domain",
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

    print("\n" + "=" * 80)
    print("PAPER-STYLE OPEN-SOURCE COMPARISON")
    print("=" * 80)
    print(
        format_table(
            comparison_rows,
            [
                "model_name",
                "source",
                "Movie",
                "Restaurant",
                "AI",
                "Literature",
                "Music",
                "Politics",
                "Science",
                "Average",
            ],
        )
    )

    return domain_rows, comparison_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the Table 1 OOD domains with GLiNER and compare against two paper-reported open-source baselines."
    )
    parser.add_argument("--checkpoint", default="urchade/gliner_large-v2.1")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--paper-baselines",
        nargs="+",
        default=list(DEFAULT_OPEN_SOURCE_BASELINES),
        choices=sorted(PAPER_BASELINES),
        help="At most two baselines from the paper table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_ood_open_source_comparison(
        checkpoint=args.checkpoint,
        split=args.split,
        max_examples=args.max_examples,
        threshold=args.threshold,
        paper_baselines=args.paper_baselines,
    )


if __name__ == "__main__":
    main()
