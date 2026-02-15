from __future__ import annotations

from sklearn.linear_model import LogisticRegression


def build_model() -> LogisticRegression:
    # dataset looks imbalanced, so balanced helps a bit
    return LogisticRegression(max_iter=2000, class_weight="balanced")
