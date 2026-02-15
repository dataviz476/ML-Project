from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier


def build_model() -> KNeighborsClassifier:
    # KNN is sensitive to scaling, so our preprocessing must scale features
    # keeping it simple
    model = KNeighborsClassifier(
        n_neighbors=15,
        weights="uniform",  # can try "distance" later
    )
    return model
