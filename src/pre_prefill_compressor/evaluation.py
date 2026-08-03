"""Paired binary evaluation with explicit denominators and exact McNemar."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BinaryMetrics:
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    unknown_negative: int
    valid_predictions: int
    invalid_predictions: int
    total_rows: int
    precision: float
    recall: float
    f1: float
    accuracy: float

    @property
    def denominator(self) -> int:
        return self.total_rows


@dataclass(frozen=True)
class PairedBinaryEvaluation:
    baseline: BinaryMetrics
    candidate: BinaryMetrics
    total_rows: int
    baseline_valid_rows: int
    baseline_invalid_rows: int
    candidate_valid_rows: int
    candidate_invalid_rows: int
    baseline_wrong_candidate_correct: int
    baseline_correct_candidate_wrong: int
    both_correct: int
    both_wrong: int
    mcnemar_exact_p_value: float

    @property
    def delta_precision(self) -> float:
        return self.candidate.precision - self.baseline.precision

    @property
    def delta_recall(self) -> float:
        return self.candidate.recall - self.baseline.recall

    @property
    def delta_f1(self) -> float:
        return self.candidate.f1 - self.baseline.f1

    @property
    def delta_accuracy(self) -> float:
        return self.candidate.accuracy - self.baseline.accuracy


def _validate_binary(values: Sequence[int], name: str) -> list[int]:
    normalized = list(values)
    if any(type(value) is not int or value not in (0, 1) for value in normalized):
        raise ValueError(f"{name} must contain only exact binary 0/1 values")
    return normalized


def _normalize_predictions(values: Sequence[int | None]) -> list[int | None]:
    return [
        value if type(value) is int and value in (0, 1) else None for value in values
    ]


def binary_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int | None],
) -> BinaryMetrics:
    """Compute binary metrics on a fixed manifest denominator.

    An invalid prediction on a positive row is a false negative.  An invalid
    prediction on a negative row is recorded as ``unknown_negative`` rather
    than a true negative.  In both cases the row remains in accuracy's total.
    """

    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true and y_pred must be non-empty and equally sized")
    truth = _validate_binary(y_true, "y_true")
    predictions = _normalize_predictions(y_pred)
    tp = sum(
        actual == 1 and predicted == 1 for actual, predicted in zip(truth, predictions)
    )
    fp = sum(
        actual == 0 and predicted == 1 for actual, predicted in zip(truth, predictions)
    )
    tn = sum(
        actual == 0 and predicted == 0 for actual, predicted in zip(truth, predictions)
    )
    fn = sum(
        actual == 1 and predicted != 1 for actual, predicted in zip(truth, predictions)
    )
    unknown_negative = sum(
        actual == 0 and predicted is None
        for actual, predicted in zip(truth, predictions)
    )
    invalid_predictions = sum(predicted is None for predicted in predictions)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(truth)
    return BinaryMetrics(
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        unknown_negative=unknown_negative,
        valid_predictions=len(truth) - invalid_predictions,
        invalid_predictions=invalid_predictions,
        total_rows=len(truth),
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
    )


def exact_mcnemar_p_value(
    baseline_wrong_candidate_correct: int,
    baseline_correct_candidate_wrong: int,
) -> float:
    """Two-sided exact McNemar p-value using the binomial null."""

    b = int(baseline_wrong_candidate_correct)
    c = int(baseline_correct_candidate_wrong)
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative")
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = min(b, c)
    numerator = sum(math.comb(discordant, value) for value in range(tail + 1))
    return min(1.0, 2.0 * numerator / (1 << discordant))


def paired_binary_evaluation(
    sample_ids: Sequence[str],
    y_true: Sequence[int],
    baseline_pred: Sequence[int | None],
    candidate_pred: Sequence[int | None],
) -> PairedBinaryEvaluation:
    """Evaluate two predictions on the exact same ordered, unique sample IDs.

    Invalid model outputs remain in the fixed denominator and count as wrong
    for the paired correctness table.  Validity is tracked independently for
    baseline and candidate, so one model cannot hide the other's invalid rows.
    """

    total = len(sample_ids)
    if total == 0 or len(set(sample_ids)) != total:
        raise ValueError("sample_ids must be non-empty and unique")
    if not (len(y_true) == len(baseline_pred) == len(candidate_pred) == total):
        raise ValueError("all paired inputs must have the same length")
    truth = _validate_binary(y_true, "y_true")
    baseline = _normalize_predictions(baseline_pred)
    candidate = _normalize_predictions(candidate_pred)
    baseline_correct = [
        prediction is not None and prediction == actual
        for prediction, actual in zip(baseline, truth)
    ]
    candidate_correct = [
        prediction is not None and prediction == actual
        for prediction, actual in zip(candidate, truth)
    ]
    b = sum(
        (not baseline_is_correct) and candidate_is_correct
        for baseline_is_correct, candidate_is_correct in zip(
            baseline_correct, candidate_correct
        )
    )
    c = sum(
        baseline_is_correct and (not candidate_is_correct)
        for baseline_is_correct, candidate_is_correct in zip(
            baseline_correct, candidate_correct
        )
    )
    both_correct = sum(
        baseline_is_correct and candidate_is_correct
        for baseline_is_correct, candidate_is_correct in zip(
            baseline_correct, candidate_correct
        )
    )
    both_wrong = sum(
        (not baseline_is_correct) and (not candidate_is_correct)
        for baseline_is_correct, candidate_is_correct in zip(
            baseline_correct, candidate_correct
        )
    )
    return PairedBinaryEvaluation(
        baseline=binary_classification_metrics(truth, baseline),
        candidate=binary_classification_metrics(truth, candidate),
        total_rows=total,
        baseline_valid_rows=sum(value is not None for value in baseline),
        baseline_invalid_rows=sum(value is None for value in baseline),
        candidate_valid_rows=sum(value is not None for value in candidate),
        candidate_invalid_rows=sum(value is None for value in candidate),
        baseline_wrong_candidate_correct=b,
        baseline_correct_candidate_wrong=c,
        both_correct=both_correct,
        both_wrong=both_wrong,
        mcnemar_exact_p_value=exact_mcnemar_p_value(b, c),
    )
