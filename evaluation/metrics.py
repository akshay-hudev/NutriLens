import math
from typing import Iterable


def mae(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return float("nan")
    return sum(abs(v) for v in values) / len(values)


def mape(actual: Iterable[float], predicted: Iterable[float]) -> float:
    pairs = [(a, p) for a, p in zip(actual, predicted) if a not in (0, None)]
    if not pairs:
        return float("nan")
    return 100.0 * sum(abs(p - a) / abs(a) for a, p in pairs) / len(pairs)


def rmse(actual: Iterable[float], predicted: Iterable[float]) -> float:
    pairs = [(a, p) for a, p in zip(actual, predicted) if a is not None and p is not None]
    if not pairs:
        return float("nan")
    return math.sqrt(sum((p - a) ** 2 for a, p in pairs) / len(pairs))


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return float("nan")
    return sum(values) / len(values)


def std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((v - avg) ** 2 for v in values) / (len(values) - 1))


def iou(mask_a, mask_b) -> float:
    intersection = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)
