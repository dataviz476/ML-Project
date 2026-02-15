from __future__ import annotations

from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_feature_columns(df: pd.DataFrame, target_col: str) -> tuple[List[str], List[str]]:
    # separate target
    X = df.drop(columns=[target_col])

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # just printing to see what's going on
    print("Numeric columns:", numeric_cols)
    print("Categorical columns:", categorical_cols)

    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:

    # numeric pipeline
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # categorical pipeline
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("data/raw/software_defect_prediction_dataset.csv")

    target = "defect"

    numeric_cols, categorical_cols = get_feature_columns(df, target)

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    X = df.drop(columns=[target])

    # just testing fit
    X_transformed = preprocessor.fit_transform(X)

    print("Original shape:", X.shape)
    print("Transformed shape:", X_transformed.shape)
    print("Preprocess test successful.")