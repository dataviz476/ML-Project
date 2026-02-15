from __future__ import annotations

import pickle
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from model.evaluate import compute_metrics

# ----------------------------------
# Page Setup
# ----------------------------------
st.set_page_config(page_title="Software Defect Prediction", layout="wide")

st.title("Software Defect Prediction System")
st.caption(
    "Train models in the cloud, upload test data, evaluate metrics, and visualize results."
)

# ----------------------------------
# Paths
# ----------------------------------
TARGET_COL = "defect"

MODEL_DIR = Path("model/saved_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = Path("data/raw/software_defect_prediction_dataset.csv")

# ----------------------------------
# Sidebar Setup
# ----------------------------------
st.sidebar.header("Setup")

st.sidebar.write("Training dataset path:")
st.sidebar.code(str(DATA_PATH))

# ----------------------------------
# Train Models Button
# ----------------------------------
train_clicked = st.sidebar.button("Train models now")

if train_clicked:
    if not DATA_PATH.exists():
        st.error("Training dataset not found in repository.")
        st.stop()

    st.info("Training started... please wait.")

    proc = subprocess.run(
        ["python", "-m", "model.train_all"],
        capture_output=True,
        text=True,
    )

    st.subheader("Training Output")
    st.code(proc.stdout)

    if proc.stderr:
        st.subheader("Errors")
        st.code(proc.stderr)

    if proc.returncode != 0:
        st.error("Training failed. Check errors above.")
        st.stop()

    st.success("Training completed. Reloading...")
    st.rerun()

# ----------------------------------
# Load Available Models
# ----------------------------------
available_models = sorted([p.stem for p in MODEL_DIR.glob("*.pkl")])

if not available_models:
    st.warning("No trained models found. Click 'Train models now' in sidebar.")
    st.stop()

model_name = st.sidebar.selectbox("Select Model", available_models)

# ----------------------------------
# Upload Test CSV
# ----------------------------------
st.subheader("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV containing all features and the 'defect' column.",
    type=["csv"],
)

if uploaded_file is None:
    st.info("Upload a CSV file to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.write("Preview of Uploaded Data:")
st.dataframe(df.head())

if TARGET_COL not in df.columns:
    st.error("Uploaded dataset must contain 'defect' column.")
    st.stop()

# ----------------------------------
# Load Model
# ----------------------------------
model_path = MODEL_DIR / f"{model_name}.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

# ----------------------------------
# Prediction
# ----------------------------------
X = df.drop(columns=[TARGET_COL])
y_true = df[TARGET_COL].to_numpy()

y_pred = model.predict(X)

y_proba = None
if hasattr(model, "predict_proba"):
    try:
        y_proba = model.predict_proba(X)[:, 1]
    except Exception:
        y_proba = None

metrics = compute_metrics(y_true, y_pred, y_proba)

# ----------------------------------
# Display Metrics
# ----------------------------------
st.subheader("Evaluation Metrics")

c1, c2, c3 = st.columns(3)

c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
c1.metric("Precision", f"{metrics['precision']:.4f}")

c2.metric("Recall", f"{metrics['recall']:.4f}")
c2.metric("F1 Score", f"{metrics['f1']:.4f}")

auc_val = metrics.get("auc", np.nan)
mcc_val = metrics.get("mcc", np.nan)

c3.metric("AUC", "N/A" if np.isnan(auc_val) else f"{auc_val:.4f}")
c3.metric("MCC", "N/A" if np.isnan(mcc_val) else f"{mcc_val:.4f}")

# ----------------------------------
# Confusion Matrix
# ----------------------------------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)

# ----------------------------------
# Classification Report
# ----------------------------------
st.subheader("Classification Report")
st.text(classification_report(y_true, y_pred))

# ----------------------------------
# Show Predictions
# ----------------------------------
with st.expander("View Predictions"):
    out_df = df[[TARGET_COL]].copy()
    out_df["predicted"] = y_pred
    if y_proba is not None:
        out_df["prob_defect_1"] = y_proba
    st.dataframe(out_df.head(50))
