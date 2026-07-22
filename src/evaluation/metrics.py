from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from seqeval.metrics import classification_report

from src.evaluation.bio_converter import TokenSpan


@dataclass(frozen=True)
class EvaluationResult:
    """Exact-match metrics aligned with the paper's span-based evaluation."""

    precision: float
    recall: float
    f1: float
    exact_matches: int
    predicted_entities: int
    reference_entities: int
    classification_report_text: str
    per_label: dict[str, dict[str, float]]


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_predictions(
    gold_spans_by_sentence: Sequence[Sequence[TokenSpan]],
    pred_spans_by_sentence: Sequence[Sequence[TokenSpan]],
    gold_bio_tags: Sequence[Sequence[str]],
    pred_bio_tags: Sequence[Sequence[str]],
) -> EvaluationResult:
    """
    Compute exact-match NER metrics.

    The paper evaluates on exact span boundary plus entity type match. We follow
    that criterion directly on token spans and keep the BIO report only as a
    human-readable diagnostic.
    """
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    per_label_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )

    for gold_spans, pred_spans in zip(gold_spans_by_sentence, pred_spans_by_sentence):
        gold_set = set(gold_spans)
        pred_set = set(pred_spans)

        true_positives = gold_set & pred_set
        false_positives = pred_set - gold_set
        false_negatives = gold_set - pred_set

        total_true_positives += len(true_positives)
        total_false_positives += len(false_positives)
        total_false_negatives += len(false_negatives)

        for _, _, label in true_positives:
            per_label_counts[label]["tp"] += 1
        for _, _, label in false_positives:
            per_label_counts[label]["fp"] += 1
        for _, _, label in false_negatives:
            per_label_counts[label]["fn"] += 1

    precision = _safe_divide(
        total_true_positives,
        total_true_positives + total_false_positives,
    )
    recall = _safe_divide(
        total_true_positives,
        total_true_positives + total_false_negatives,
    )
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    per_label_metrics: dict[str, dict[str, float]] = {}
    for label, counts in sorted(per_label_counts.items()):
        label_precision = _safe_divide(counts["tp"], counts["tp"] + counts["fp"])
        label_recall = _safe_divide(counts["tp"], counts["tp"] + counts["fn"])
        label_f1 = _safe_divide(
            2 * label_precision * label_recall,
            label_precision + label_recall,
        )
        per_label_metrics[label] = {
            "precision": label_precision,
            "recall": label_recall,
            "f1": label_f1,
            "support": float(counts["tp"] + counts["fn"]),
        }

    report = classification_report(
        gold_bio_tags,
        pred_bio_tags,
        digits=4,
        zero_division=0,
    )

    return EvaluationResult(
        precision=precision,
        recall=recall,
        f1=f1,
        exact_matches=total_true_positives,
        predicted_entities=sum(len(spans) for spans in pred_spans_by_sentence),
        reference_entities=sum(len(spans) for spans in gold_spans_by_sentence),
        classification_report_text=report,
        per_label=per_label_metrics,
    )
