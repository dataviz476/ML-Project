from __future__ import annotations

from typing import Optional, Any


def build_model(random_state: int = 42) -> Optional[Any]:
    # keeping xgboost optional so project doesn't crash if not installed
    try:
        from xgboost import XGBClassifier
    except Exception:
        print("XGBoost not installed. Skipping this model.")
        return None

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        eval_metric="logloss",
        n_jobs=-1,
    )

    return model
