# =============================================================================
# app.py
# Heart Disease Prediction – Flask REST API
# Assignment 10: End-to-End ML Deployment using GitHub and Render
# =============================================================================

import os
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template

# ── Initialize Flask Application ──────────────────────────────────────────────
app = Flask(__name__)

# ── Load the Pre-trained Model ────────────────────────────────────────────────
# model.pkl is saved by train_model.py using joblib
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
model = joblib.load(MODEL_PATH)
print("[INFO] model.pkl loaded successfully.")

# =============================================================================
# FEATURE ORDER (must match training data column order)
# age, sex, cp, trestbps, chol, fbs, restecg, thalach,
# exang, oldpeak, slope, ca, thal
# =============================================================================
FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal",
]


# =============================================================================
# ROUTE 1: Home Page  →  GET /
# =============================================================================
@app.route("/", methods=["GET"])
def home():
    """
    Renders the index.html template (optional UI for manual testing).
    Falls back to a plain JSON welcome message if no template exists.
    """
    try:
        return render_template("index.html")
    except Exception:
        return jsonify({
            "message": "Heart Disease Prediction API",
            "status": "running",
            "usage": "POST /predict with JSON body containing patient features",
        })


# =============================================================================
# ROUTE 2: Prediction Endpoint  →  POST /predict
# =============================================================================
@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a JSON body with patient clinical features.
    Returns prediction: 'Heart Disease Detected' or 'No Heart Disease'.

    Expected JSON input:
    {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
    }
    """
    try:
        # ── Parse incoming JSON data ──────────────────────────────────────────
        data = request.get_json(force=True)

        if not data:
            return jsonify({"error": "No JSON data received."}), 400

        # ── Extract features in the correct order ─────────────────────────────
        features = [data.get(f) for f in FEATURE_NAMES]

        # ── Validate: all required features must be present ───────────────────
        missing_fields = [
            name for name, val in zip(FEATURE_NAMES, features) if val is None
        ]
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {missing_fields}"
            }), 400

        # ── Convert to 2-D NumPy array (1 sample × 13 features) ──────────────
        input_array = np.array(features, dtype=float).reshape(1, -1)

        # ── Run prediction ────────────────────────────────────────────────────
        prediction = model.predict(input_array)[0]

        # ── Map numeric output to human-readable label ────────────────────────
        label = "Heart Disease Detected" if int(prediction) == 1 else "No Heart Disease"

        return jsonify({"prediction": label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================================================================
# Run the Flask development server
# On Render, gunicorn is used; host="0.0.0.0" is required for cloud hosting
# =============================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
