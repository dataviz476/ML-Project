from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier


def build_model(random_state: int = 42) -> RandomForestClassifier:
    # simple random forest
    # balanced helps for imbalanced defect dataset
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    return model
