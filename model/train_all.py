from __future__ import annotations

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from model.utils import ensure_dir, save_pkl
from model.preprocess import get_feature_columns, build_preprocessor
from model.evaluate import compute_metrics

from model import (
    logistic_regression,
    decision_tree,
    knn,
    naive_bayes,
    random_forest,
    xgboost_model,
)


TARGET = "defect"
RANDOM_STATE = 42


def main():

    print("Loading dataset...")
    df = pd.read_csv("data/raw/software_defect_prediction_dataset.csv")

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found")

    print("Dataset shape:", df.shape)
    print("Target distribution:\n", df[TARGET].value_counts())

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # build preprocessing
    num_cols, cat_cols = get_feature_columns(df, TARGET)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    models = {
        "logistic_regression": logistic_regression.build_model(),
        "decision_tree": decision_tree.build_model(),
        "knn": knn.build_model(),
        "naive_bayes": naive_bayes.build_model(),
        "random_forest": random_forest.build_model(),
        "xgboost": xgboost_model.build_model(),
    }

    results = []

    ensure_dir(Path("model/saved_models"))

    for name, model in models.items():

        if model is None:
            print(f"Skipping {name}")
            continue

        print(f"\nTraining {name}...")

        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        y_proba = None
        if hasattr(pipe, "predict_proba"):
            try:
                y_proba = pipe.predict_proba(X_test)[:, 1]
            except Exception:
                pass

        metrics = compute_metrics(y_test.to_numpy(), y_pred, y_proba)

        print("Metrics:", metrics)

        save_pkl(pipe, f"model/saved_models/{name}.pkl")

        row = {"model": name}
        row.update(metrics)
        results.append(row)

    print("\n==== Model Comparison ====")
    print(pd.DataFrame(results).sort_values("f1", ascending=False))


if __name__ == "__main__":
    main()
