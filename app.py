from __future__ import annotations

import sys
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
st.caption("Train models in the cloud, upload test data, evaluate metrics, and visualize results.")

TARGET_COL = "defect"

# project-root-safe paths
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model" / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = ROOT / "data" / "raw" / "software_defect_prediction_dataset.csv"

# ----------------------------------
# Sidebar: Environment + Setup
# ----------------------------------
st.sidebar.header("Environment")
st.sidebar.write("Python:", sys.version.split()[0])
st.sidebar.write("Python executable:")
st.sidebar.code(sys.executable)

# quick dependency check (so you can SEE it, not guess)
for pkg in ["numpy", "pandas", "sklearn"]:
    try:
        __import__(pkg)
        st.sidebar.success(f"{pkg} OK")
    except Exception:
        st.sidebar.error(f"{pkg} missing")

st.sidebar.header("Setup")
st.sidebar.write("Training dataset path:")
st.sidebar.code(str(DATA_PATH))

if not DATA_PATH.exists():
    st.sidebar.error("Dataset not found at the above path (case-sensitive on Linux).")
    st.stop()

# ----------------------------------
# Train Models Button
# ----------------------------------
train_clicked = st.sidebar.button("Train models now")

if train_clicked:
    st.info("Training started...")

    # IMPORTANT FIX: use sys.executable, NOT "python"
    proc = subprocess.run(
        [sys.executable, "-m", "model.train_all"],
        cwd=str(ROOT),  # ensure working directory = repo root
        capture_output=True,
        text=True,
    )

    st.subheader("Training Output (stdout)")
    st.code(proc.stdout if proc.stdout else "No stdout output.")

    if proc.stderr:
        st.subheader("Training Output (stderr)")
        st.code(proc.stderr)

    if proc.returncode != 0:
        st.error("Training failed. See stderr above.")
        st.stop()

    # show what got created
    created = sorted([p.name for p in MODEL_DIR.glob("*.pkl")])
    st.success("Training completed.")
    st.write("Saved models:")
    st.code("\n".join(created) if created else "NO PKL FILES FOUND (check train_all.py save path)")

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
st.subheader("Upload Test Dataset (CSV)")
uploaded_file = st.file_uploader(
    "Upload CSV containing all features and the 'defect' column.",
    type=["csv"],
)

if uploaded_file is None:
    st.info("Upload a CSV file to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.write("Preview:")
st.dataframe(df.head())

if TARGET_COL not in df.columns:
    st.error("Uploaded dataset must contain 'defect' column for evaluation.")
    st.stop()

# ----------------------------------
# Load Model
# ----------------------------------
model_path = MODEL_DIR / f"{model_name}.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# ----------------------------------
# Predict + Evaluate
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

with st.expander("View Predictions"):
    out_df = df[[TARGET_COL]].copy()
    out_df["predicted"] = y_pred
    if y_proba is not None:
        out_df["prob_defect_1"] = y_proba
    st.dataframe(out_df.head(50))
