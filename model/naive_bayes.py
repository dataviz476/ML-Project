from __future__ import annotations

from sklearn.naive_bayes import GaussianNB


def build_model() -> GaussianNB:
    # very simple baseline model
    # works well for numeric features
    model = GaussianNB()
    return model
