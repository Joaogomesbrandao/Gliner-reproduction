from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from tqdm import tqdm

from src.datasets.conll2003 import DEFAULT_DATASET_NAME
from src.datasets.registry import load_dataset_resources
from src.evaluation.bio_converter import (
    bio_tags_to_spans,
    predictions_to_token_spans,
    token_spans_to_bio,
)
from src.evaluation.metrics import EvaluationResult, evaluate_predictions
from src.utils import key_value_lines, serialize_labels, write_text


class PredictsEntities(Protocol):
    model_name: str
    checkpoint: str | None

    def predict(
        self,
        text: str,
        labels: Sequence[str],
        threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        ...

    def predict_batch(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        threshold: float = 0.5,
    ) -> list[list[dict[str, Any]]]:
        ...


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    dataset_name: str = DEFAULT_DATASET_NAME
    split: str = "test"
    labels: tuple[str, ...] = ()
    max_examples: int | None = None
    threshold: float = 0.5
    batch_size: int = 8


def run_experiment(
    model: PredictsEntities,
    config: ExperimentConfig,
    *,
    classification_report_path: Path | None = None,
    logger: logging.Logger | None = None,
) -> tuple[dict[str, Any], EvaluationResult]:
    """
    Run a dataset/model experiment and return both the summary row and metrics.

    The experiment loop is dataset-agnostic so each benchmark can reuse the same
    reconstruction, inference, exact-match evaluation, and reporting pipeline.
    """
    resources = load_dataset_resources(
        config.dataset_name,
        split=config.split,
        max_examples=config.max_examples,
    )
    samples = resources.samples
    labels = tuple(config.labels) or resources.default_prompt_labels

    if logger is not None:
        logger.info(
            "Starting experiment=%s model=%s dataset=%s split=%s checkpoint=%s "
            "examples=%s labels=%s threshold=%.2f",
            config.experiment_name,
            model.model_name,
            config.dataset_name,
            config.split,
            model.checkpoint or "",
            len(samples),
            list(labels),
            config.threshold,
        )

    if len(samples):
        warmup_sentence = resources.build_sentence(samples[0])
        model.predict(
            text=warmup_sentence.text,
            labels=labels,
            threshold=config.threshold,
        )

    gold_spans_by_sentence: list[list[tuple[int, int, str]]] = []
    pred_spans_by_sentence: list[list[tuple[int, int, str]]] = []
    gold_bio_tags: list[list[str]] = []
    pred_bio_tags: list[list[str]] = []
    raw_predicted_entities = 0
    total_time = 0.0
    supports_batch = hasattr(model, "predict_batch") and config.batch_size > 1

    for batch_start in tqdm(
        range(0, len(samples), config.batch_size),
        desc=config.experiment_name,
    ):
        batch_indices = range(batch_start, min(batch_start + config.batch_size, len(samples)))
        batch_samples = [samples[index] for index in batch_indices]
        sentences = [resources.build_sentence(sample) for sample in batch_samples]
        gold_batch_bio = [list(sentence.gold_bio_tags) for sentence in sentences]
        gold_batch_spans = [bio_tags_to_spans(tags) for tags in gold_batch_bio]

        start_time = time.perf_counter()
        if supports_batch:
            batch_predictions = model.predict_batch(
                texts=[sentence.text for sentence in sentences],
                labels=labels,
                threshold=config.threshold,
            )
        else:
            batch_predictions = [
                model.predict(
                    text=sentence.text,
                    labels=labels,
                    threshold=config.threshold,
                )
                for sentence in sentences
            ]
        elapsed = time.perf_counter() - start_time
        total_time += elapsed

        for sentence, gold_bio, gold_spans, predictions in zip(
            sentences,
            gold_batch_bio,
            gold_batch_spans,
            batch_predictions,
        ):
            raw_predicted_entities += len(predictions)
            pred_spans = predictions_to_token_spans(
                predictions,
                sentence.offsets,
                label_mapping=resources.prediction_label_mapping,
            )
            pred_bio = token_spans_to_bio(len(sentence.tokens), pred_spans)

            gold_spans_by_sentence.append(gold_spans)
            pred_spans_by_sentence.append(pred_spans)
            gold_bio_tags.append(gold_bio)
            pred_bio_tags.append(pred_bio)

    metrics = evaluate_predictions(
        gold_spans_by_sentence=gold_spans_by_sentence,
        pred_spans_by_sentence=pred_spans_by_sentence,
        gold_bio_tags=gold_bio_tags,
        pred_bio_tags=pred_bio_tags,
    )

    average_time = total_time / len(samples) if len(samples) else 0.0
    row = {
        "experiment": config.experiment_name,
        "model_name": model.model_name,
        "dataset": config.dataset_name,
        "split": config.split,
        "checkpoint": model.checkpoint or "",
        "labels": serialize_labels(labels),
        "examples": len(samples),
        "threshold": config.threshold,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "time_seconds_total": total_time,
        "time_seconds_average": average_time,
        "predicted_entities": metrics.predicted_entities,
        "reference_entities": metrics.reference_entities,
        "raw_predicted_entities": raw_predicted_entities,
        "exact_matches": metrics.exact_matches,
    }

    if logger is not None:
        logger.info(
            "Finished experiment=%s model=%s precision=%.4f recall=%.4f f1=%.4f "
            "time_total=%.2fs time_avg=%.4fs predicted=%s reference=%s",
            config.experiment_name,
            model.model_name,
            metrics.precision,
            metrics.recall,
            metrics.f1,
            total_time,
            average_time,
            metrics.predicted_entities,
            metrics.reference_entities,
        )

    if classification_report_path is not None:
        metadata = key_value_lines(
            [
                ("experiment", config.experiment_name),
                ("model_name", model.model_name),
                ("dataset", config.dataset_name),
                ("split", config.split),
                ("checkpoint", model.checkpoint or ""),
                ("examples", len(samples)),
                ("labels", list(labels)),
                ("threshold", config.threshold),
                ("precision", f"{metrics.precision:.4f}"),
                ("recall", f"{metrics.recall:.4f}"),
                ("f1", f"{metrics.f1:.4f}"),
                ("time_seconds_total", f"{total_time:.4f}"),
                ("time_seconds_average", f"{average_time:.4f}"),
                ("predicted_entities", metrics.predicted_entities),
                ("reference_entities", metrics.reference_entities),
                ("raw_predicted_entities", raw_predicted_entities),
                ("exact_matches", metrics.exact_matches),
            ]
        )
        report_content = (
            f"{metadata}\n\n"
            "classification_report\n"
            "=====================\n"
            f"{metrics.classification_report_text}"
        )
        write_text(classification_report_path, report_content)

    return row, metrics
