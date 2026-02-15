from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:

    out: Dict[str, float] = {}

    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    # AUC only makes sense if we have probabilities (and binary target)
    if y_proba is not None:
        try:
            out["auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            # happens if only one class in y_true for this split
            out["auc"] = float("nan")
    else:
        out["auc"] = float("nan")

    try:
        out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        out["mcc"] = float("nan")

    return out


# quick test (optional)
if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression

    from model.preprocess import get_feature_columns, build_preprocessor

    TARGET = "defect"

    df = pd.read_csv("data/raw/software_defect_prediction_dataset.csv")

    # make sure we get both classes in the sample
    df0 = df[df[TARGET] == 0]
    df1 = df[df[TARGET] == 1]

    if len(df0) == 0 or len(df1) == 0:
        raise ValueError("Dataset has only one class overall. Can't train a classifier.")

    # take a small balanced-ish sample
    n0 = 5
    n1 = 5

    df_small = pd.concat(
        [
            df0.sample(n=min(n0, len(df0)), random_state=42),
            df1.sample(n=min(n1, len(df1)), random_state=42),
        ],
        axis=0,
    ).sample(frac=1.0, random_state=42)  # shuffle

    print("Using small sample shape:", df_small.shape)
    print("Class counts:\n", df_small[TARGET].value_counts())


