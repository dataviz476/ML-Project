from __future__ import annotations

from sklearn.tree import DecisionTreeClassifier


def build_model(random_state: int = 42) -> DecisionTreeClassifier:
    # simple tree, nothing fancy
    # using balanced because dataset seems imbalanced
    model = DecisionTreeClassifier(
        random_state=random_state,
        class_weight="balanced",
        max_depth=None,  # letting it grow for now
    )
    return model
