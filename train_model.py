# -*- coding: utf-8 -*-
"""
Universal Risk Prediction Training Script (Stable Version)
-----------------------------------------------------------
✔ Auto-detects number of classes (2 / 3 / 4)
✔ Epoch-based training with tqdm
✔ Early stopping (fixed)
✔ L2 regularization
✔ Prevents overfitting
✔ Clean classification report
✔ Flask compatible pipeline
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import copy
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier


# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = r"D:\Heart Attack\dataset"
CSV_NAME  = "Lifestyle_and_Health_Risk_Prediction_Synthetic_Dataset.csv"
TARGET_COL = "health_risk"

ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model_pipeline.joblib")

RANDOM_STATE = 42
EPOCHS = 30
PATIENCE = 10
TEST_SIZE = 0.20


# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(os.path.join(DATA_PATH, CSV_NAME))
print("Loaded dataset:", df.shape)

if TARGET_COL not in df.columns:
    raise ValueError(f"{TARGET_COL} not found in dataset.")

df = df.dropna(subset=[TARGET_COL]).copy()

# Convert text labels safely
df[TARGET_COL] = df[TARGET_COL].astype(str).str.lower().str.strip()
labels = sorted(df[TARGET_COL].unique())
label_map = {l: i for i, l in enumerate(labels)}
df[TARGET_COL] = df[TARGET_COL].map(label_map)

df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
df = df.dropna(subset=[TARGET_COL])
df[TARGET_COL] = df[TARGET_COL].astype(int)

print("\nClass Distribution:")
print(df[TARGET_COL].value_counts().sort_index())

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]


# -----------------------------
# PREPROCESSING
# -----------------------------
numeric_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler())
    ]), numeric_cols),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), categorical_cols)
])


# -----------------------------
# SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

X_train_processed = preprocess.fit_transform(X_train)
X_test_processed = preprocess.transform(X_test)


# -----------------------------
# MODEL
# -----------------------------
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    alpha=0.005,                   # L2 regularization
    batch_size=64,
    learning_rate_init=0.001,
    learning_rate="adaptive",
    max_iter=1,                    # 1 iteration per epoch
    warm_start=True,
    random_state=RANDOM_STATE
)

classes = np.unique(y_train)

print("\nTraining with tqdm (Early Stopping Enabled)...")

best_val_acc = 0
no_improve = 0
best_model = None

for epoch in tqdm(range(EPOCHS), desc="Epochs"):

    model.partial_fit(X_train_processed, y_train, classes=classes)

    train_acc = accuracy_score(y_train, model.predict(X_train_processed))
    val_acc = accuracy_score(y_test, model.predict(X_test_processed))

    print(f"Epoch {epoch+1}/{EPOCHS} | Train: {train_acc:.4f} | Val: {val_acc:.4f}")

    # Early stopping logic
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = copy.deepcopy(model)
        no_improve = 0
    else:
        no_improve += 1

    if no_improve >= PATIENCE:
        print("\nEarly stopping triggered.")
        break

# Restore best model
if best_model is not None:
    model = best_model


# -----------------------------
# FINAL EVALUATION
# -----------------------------
pred = model.predict(X_test_processed)
final_acc = accuracy_score(y_test, pred)

print("\nFinal Test Accuracy:", round(final_acc, 4))

unique_classes = sorted(np.unique(y_test))

if len(unique_classes) == 2:
    class_names = ["LOW", "HIGH"]
elif len(unique_classes) == 3:
    class_names = ["LOW", "MEDIUM", "HIGH"]
elif len(unique_classes) == 4:
    class_names = ["LOW", "MEDIUM", "HIGH", "VERY HIGH"]
else:
    class_names = [f"Class_{i}" for i in unique_classes]

print("\nClassification Report:\n",
      classification_report(
          y_test,
          pred,
          labels=unique_classes,
          target_names=class_names
      ))


# -----------------------------
# SAVE PIPELINE
# -----------------------------
pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", model)
])

os.makedirs(ARTIFACT_DIR, exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)

print("\nModel Saved Successfully!")
print("Saved to:", MODEL_PATH)