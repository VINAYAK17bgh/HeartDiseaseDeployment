# =============================================================================
# train_model.py
# Heart Disease Prediction – Model Training Script
# Assignment 10: End-to-End ML Deployment using GitHub and Render
# =============================================================================

# ─── REQUIRED LIBRARIES ──────────────────────────────────────────────────────
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend – safe in terminals and on servers
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# =============================================================================
# TASK 1: DATA UNDERSTANDING AND PREPROCESSING
# =============================================================================

# ── Step 1: Load the dataset ──────────────────────────────────────────────────
print("=" * 60)
print("TASK 1: DATA UNDERSTANDING AND PREPROCESSING")
print("=" * 60)

# Build path relative to this script file (works on Windows, Linux, Render)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(SCRIPT_DIR, "heart.csv"))   # Load heart disease CSV file
print("\n[INFO] Dataset loaded successfully!")
print(f"[INFO] Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# ── Step 2: Display the first five rows ──────────────────────────────────────
print("\n--- First 5 Rows of the Dataset ---")
print(df.head())

# ── Step 3: Identify numerical features ──────────────────────────────────────
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
print("\n--- Numerical Features ---")
print(numerical_features)

# ── Step 4: Identify the target variable ─────────────────────────────────────
# 'target' column: 1 = Heart Disease Present, 0 = No Heart Disease
target_variable = "target"
print(f"\n--- Target Variable ---\n'{target_variable}'")
print(df[target_variable].value_counts())

# ── Step 5: Check for missing values ─────────────────────────────────────────
print("\n--- Missing Values per Column ---")
missing = df.isnull().sum()
print(missing)
if missing.sum() == 0:
    print("[INFO] No missing values found. Dataset is clean.")
else:
    print("[WARNING] Missing values detected. Handle before training.")

# ── Step 6: Split into features (X) and target (y) ───────────────────────────
X = df.drop(columns=[target_variable])   # All columns except target
y = df[target_variable]                  # Target column

# 80% training, 20% testing; random_state=42 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"\n--- Train/Test Split (80/20) ---")
print(f"Training samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")

# =============================================================================
# TASK 2: MODEL DEVELOPMENT – LOGISTIC REGRESSION
# =============================================================================
print("\n" + "=" * 60)
print("TASK 2: MODEL DEVELOPMENT")
print("=" * 60)

# ── Train Logistic Regression model ──────────────────────────────────────────
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
print("\n[INFO] Logistic Regression model trained successfully!")

# ── Predictions on test set ───────────────────────────────────────────────────
y_pred = model.predict(X_test)

# ── Accuracy Score ────────────────────────────────────────────────────────────
accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- Model Accuracy ---")
print(f"Accuracy Score: {accuracy * 100:.2f}%")

# ── Confusion Matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
print("\n--- Confusion Matrix ---")
print(cm)

# ── Classification Report ─────────────────────────────────────────────────────
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

# ── Save Confusion Matrix as image (optional) ────────────────────────────────
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Disease", "Disease"],
    yticklabels=["No Disease", "Disease"],
)
plt.title("Confusion Matrix – Heart Disease Prediction")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "confusion_matrix.png"), dpi=150)
print("\n[INFO] Confusion matrix saved as 'confusion_matrix.png'")
plt.show()

# ── Save trained model using joblib ──────────────────────────────────────────
joblib.dump(model, os.path.join(SCRIPT_DIR, "model.pkl"))
print("[INFO] Trained model saved as 'model.pkl'")

print("\n[DONE] Training complete. Model is ready for deployment.")
