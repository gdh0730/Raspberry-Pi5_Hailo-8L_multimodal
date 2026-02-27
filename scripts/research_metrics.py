#!/usr/bin/env python3
"""Minimal offline metrics utilities (stdlib only)."""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Callable, Iterable, List, Optional, Sequence, Tuple


def accuracy_score(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if not y_true:
        return float("nan")
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def macro_f1_score(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if not y_true:
        return float("nan")

    labels = sorted(set(y_true))
    f1_sum = 0.0
    for label in labels:
        tp = 0
        fp = 0
        fn = 0
        for t, p in zip(y_true, y_pred):
            if t == label and p == label:
                tp += 1
            elif t != label and p == label:
                fp += 1
            elif t == label and p != label:
                fn += 1
        denom = (2 * tp + fp + fn)
        f1 = 0.0 if denom == 0 else (2 * tp) / denom
        f1_sum += f1
    return f1_sum / len(labels)


def mean_absolute_error(
    y_true: Sequence[Optional[int]], y_pred: Sequence[Optional[int]]
) -> float:
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if t is not None and p is not None]
    if not pairs:
        return float("nan")
    return sum(abs(t - p) for t, p in pairs) / len(pairs)


def percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    values_sorted = sorted(values)
    if len(values_sorted) == 1:
        return values_sorted[0]
    rank = q * (len(values_sorted) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return values_sorted[lo]
    alpha = rank - lo
    return values_sorted[lo] * (1 - alpha) + values_sorted[hi] * alpha


def bootstrap_ci(
    y_true: Sequence,
    y_pred: Sequence,
    metric_fn: Callable[[Sequence, Sequence], float],
    n_boot: int = 1000,
    seed: int = 42,
    low_q: float = 0.025,
    high_q: float = 0.975,
) -> Tuple[float, float]:
    if not y_true:
        return float("nan"), float("nan")
    n = len(y_true)
    rng = random.Random(seed)
    sampled = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        sampled.append(metric_fn(yt, yp))
    return percentile(sampled, low_q), percentile(sampled, high_q)


def majority_vote(values: Iterable) -> Optional:
    c = Counter(v for v in values if v is not None)
    if not c:
        return None
    max_count = max(c.values())
    candidates = [k for k, v in c.items() if v == max_count]
    return sorted(candidates)[0]
