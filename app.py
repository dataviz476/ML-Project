from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from sklearn.metrics import confusion_matrix, classification_report

from model.evaluate import compute_metrics

st.set_page_config(page_title="Software Defect Prediction System", layout="wide")

st.title("Software Defect Prediction System")
st.markdown(
    """
This application predicts whether a software module is likely to contain defects
based on engineering metrics such as code complexity, coupling, test coverage,
and development activity indicators.
"""
)

# ===============================
# Load Models
# ===============================

MODEL_DIR = Path("model/saved_models")

available_models = [f.stem for f in MODEL_DIR.glob("*.pkl")]

if not available_models:
    st.error("No trained models found. Train models before deploying.")
    st.stop()

model_name = st.selectbox("Select Machine Learning Model", available_models)

# ===============================
# Upload Dataset
# ===============================

uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file containing the 'defect' column.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("Dataset Overview")
st.write("Rows:", df.shape[0])
st.write("Columns:", df.shape[1])
st.dataframe(df.head())

if "defect" not in df.columns:
    st.error("Dataset must contain 'defect' column.")
    st.stop()

# ===============================
# Load Model
# ===============================

model_path = MODEL_DIR / f"{model_name}.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

# ===============================
# Prediction
# ===============================

X = df.drop(columns=["defect"])
y_true = df["defect"]

y_pred = model.predict(X)

y_proba = None
if hasattr(model, "predict_proba"):
    try:
        y_proba = model.predict_proba(X)[:, 1]
    except Exception:
        pass

metrics = compute_metrics(y_true.to_numpy(), y_pred, y_proba)

# ===============================
# Display Metrics
# ===============================

st.subheader("Model Evaluation Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", round(metrics["accuracy"], 4))
col1.metric("Precision", round(metrics["precision"], 4))
col2.metric("Recall", round(metrics["recall"], 4))
col2.metric("F1 Score", round(metrics["f1"], 4))
col3.metric("AUC", round(metrics["auc"], 4) if not np.isnan(metrics["auc"]) else "N/A")
col3.metric("MCC", round(metrics["mcc"], 4) if not np.isnan(metrics["mcc"]) else "N/A")

# ===============================
# Confusion Matrix
# ===============================

st.subheader("Confusion Matrix")

cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

st.pyplot(fig)

# ===============================
# Classification Report
# ===============================

st.subheader("Classification Report")
report = classification_report(y_true, y_pred)
st.text(report)

# ===============================
# Feature Importance (Tree Models)
# ===============================

if hasattr(model.named_steps["model"], "feature_importances_"):
    st.subheader("Feature Importance")

    importances = model.named_steps["model"].feature_importances_
    feature_names = model.named_steps["preprocess"].get_feature_names_out()

    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(15)

    fig2, ax2 = plt.subplots(figsize=(8,6))
    sns.barplot(data=feat_df, x="Importance", y="Feature", ax=ax2)
    st.pyplot(fig2)

# ===============================
# Interpretation Section
# ===============================

st.subheader("Model Interpretation")

if metrics["mcc"] > 0.9:
    st.success("The model shows extremely strong correlation in predictions.")
elif metrics["mcc"] > 0.7:
    st.info("The model demonstrates strong predictive capability.")
else:
    st.warning("Model performance may need improvement.")
