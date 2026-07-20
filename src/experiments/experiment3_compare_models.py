from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from src.datasets.conll2003 import DEFAULT_DATASET_NAME, DEFAULT_GLINER_LABELS
from src.experiments.common import ExperimentConfig, run_experiment
from src.evaluation.bio_converter import normalize_prediction_label
from src.models.gliner_model import GLiNERModel
from src.utils import configure_logging, ensure_results_dir, format_table, write_rows_to_csv


@dataclass
class SpaCyNERModel:
    checkpoint: str = "en_core_web_sm"

    def __post_init__(self) -> None:
        import spacy

        self._nlp = spacy.load(self.checkpoint)
        self.model_name = f"spaCy ({self.checkpoint})"

    def predict(
        self,
        text: str,
        labels: Sequence[str],
        threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        del labels, threshold
        doc = self._nlp(text)
        predictions: list[dict[str, Any]] = []
        for entity in doc.ents:
            label = normalize_prediction_label(entity.label_)
            if label is None:
                continue
            predictions.append(
                {
                    "start": entity.start_char,
                    "end": entity.end_char,
                    "text": entity.text,
                    "label": label,
                    "score": 1.0,
                }
            )
        return predictions


@dataclass
class FlairNERModel:
    checkpoint: str = "ner"

    def __post_init__(self) -> None:
        from flair.models import SequenceTagger

        self._tagger = SequenceTagger.load(self.checkpoint)
        self.model_name = f"Flair ({self.checkpoint})"

    def predict(
        self,
        text: str,
        labels: Sequence[str],
        threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        del labels, threshold

        from flair.data import Sentence

        sentence = Sentence(text)
        self._tagger.predict(sentence)

        predictions: list[dict[str, Any]] = []
        for span in sentence.get_spans("ner"):
            label = normalize_prediction_label(span.tag)
            if label is None:
                continue
            predictions.append(
                {
                    "start": span.start_position,
                    "end": span.end_position,
                    "text": span.text,
                    "label": label,
                    "score": float(span.score),
                }
            )
        return predictions


def build_models(
    *,
    include_spacy: bool = False,
    include_flair: bool = False,
    spacy_model: str = "en_core_web_sm",
    flair_model: str = "ner",
) -> list[tuple[str, Callable[[], Any]]]:
    models: list[tuple[str, Callable[[], Any]]] = [
        (
            "GLiNER Large",
            lambda: GLiNERModel(
                "urchade/gliner_large-v2.1",
                model_name="GLiNER Large",
            ),
        ),
        (
            "GLiNER Base",
            lambda: GLiNERModel(
                "urchade/gliner_base",
                model_name="GLiNER Base",
            ),
        ),
        (
            "GLiNER Small",
            lambda: GLiNERModel(
                "urchade/gliner_small-v2.1",
                model_name="GLiNER Small",
            ),
        ),
    ]

    if include_spacy:
        models.append((f"spaCy ({spacy_model})", lambda: SpaCyNERModel(spacy_model)))

    if include_flair:
        models.append((f"Flair ({flair_model})", lambda: FlairNERModel(flair_model)))

    return models


def run_compare_models_experiment(
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    split: str = "test",
    max_examples: int | None = None,
    threshold: float = 0.5,
    labels: Sequence[str] = DEFAULT_GLINER_LABELS,
    include_spacy: bool = False,
    include_flair: bool = False,
    spacy_model: str = "en_core_web_sm",
    flair_model: str = "ner",
) -> list[dict[str, Any]]:
    """
    Compare GLiNER checkpoints, optionally adding closed-label baselines.

    spaCy and Flair do not implement the paper's open-type prompt interface, so
    they are included only as optional baselines within the same evaluation
    pipeline, not as paper-equivalent comparisons.
    """
    results_dir = ensure_results_dir()
    logger = configure_logging()
    rows: list[dict[str, Any]] = []

    try:
        models = build_models(
            include_spacy=include_spacy,
            include_flair=include_flair,
            spacy_model=spacy_model,
            flair_model=flair_model,
        )
    except ImportError as error:
        raise SystemExit(
            "Optional baseline dependency missing. Install it first or rerun "
            "without --include-spacy / --include-flair."
        ) from error

    for _, build_model in models:
        model = build_model()
        config = ExperimentConfig(
            experiment_name="compare_models",
            dataset_name=dataset_name,
            split=split,
            labels=tuple(labels),
            max_examples=max_examples,
            threshold=threshold,
        )
        row, _ = run_experiment(model=model, config=config, logger=logger)
        rows.append(row)

    write_rows_to_csv(results_dir / "compare_models.csv", rows)

    print("\n" + "=" * 80)
    print("COMPARE MODELS SUMMARY")
    print("=" * 80)
    print(
        format_table(
            rows,
            [
                "model_name",
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
    parser = argparse.ArgumentParser(description="Compare models on the same evaluation pipeline.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--labels",
        nargs="+",
        default=list(DEFAULT_GLINER_LABELS),
        help="Prompt labels for GLiNER checkpoints.",
    )
    parser.add_argument("--include-spacy", action="store_true")
    parser.add_argument("--include-flair", action="store_true")
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--flair-model", default="ner")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_compare_models_experiment(
        dataset_name=args.dataset,
        split=args.split,
        max_examples=args.max_examples,
        threshold=args.threshold,
        labels=args.labels,
        include_spacy=args.include_spacy,
        include_flair=args.include_flair,
        spacy_model=args.spacy_model,
        flair_model=args.flair_model,
    )


if __name__ == "__main__":
    main()
