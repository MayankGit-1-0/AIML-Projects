"""
Credit Card Fraud Detection — Flask Backend
Trains 4 ML models and exposes REST APIs for comparison & prediction.
"""

import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Data Loading & Preprocessing ─────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "creditcard 2.csv")


def load_and_prepare_data():
    """Load CSV, scale features, undersample majority class, split."""
    df = pd.read_csv(DATA_PATH)

    # Scale Amount and Time
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])
    df["Time"] = scaler.fit_transform(df[["Time"]])

    # ── Undersample majority class ───────────────────────────────────────
    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0]
    legit_sample = legit.sample(n=len(fraud), random_state=42)
    balanced = pd.concat([fraud, legit_sample]).sample(frac=1, random_state=42)

    X = balanced.drop("Class", axis=1)
    y = balanced["Class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, scaler


# ── Model Training ───────────────────────────────────────────────────────────

MODELS = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, random_state=42
    ),
}


def train_all_models(X_train, X_test, y_train, y_test):
    """Train every model, compute metrics, return results dict."""
    results = {}
    trained_models = {}

    for name, model in MODELS.items():
        print(f"  ▸ Training {name} …")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else model.decision_function(X_test)
        )

        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
            "precision": round(precision_score(y_test, y_pred) * 100, 2),
            "recall": round(recall_score(y_test, y_pred) * 100, 2),
            "f1": round(f1_score(y_test, y_pred) * 100, 2),
            "roc_auc": round(roc_auc_score(y_test, y_proba) * 100, 2),
            "confusion_matrix": cm,
        }
        trained_models[name] = model

    # Determine best model by F1
    best = max(results, key=lambda k: results[k]["f1"])
    return results, trained_models, best


# ── Boot ─────────────────────────────────────────────────────────────────────
print("🔄  Loading data …")
X_train, X_test, y_train, y_test, scaler = load_and_prepare_data()
print(f"   Train: {len(X_train)}  |  Test: {len(X_test)}")

print("🧠  Training models …")
metrics, trained_models, best_model = train_all_models(
    X_train, X_test, y_train, y_test
)
print(f"✅  Best model: {best_model}  (F1 = {metrics[best_model]['f1']}%)")

FEATURE_NAMES = list(X_train.columns)

# ── Routes ───────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/metrics")
def api_metrics():
    return jsonify({"metrics": metrics, "best_model": best_model})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    model_name = data.get("model", best_model)
    features = data.get("features", {})

    if model_name not in trained_models:
        return jsonify({"error": f"Unknown model: {model_name}"}), 400

    # Build feature vector in correct column order
    try:
        row = [float(features.get(f, 0)) for f in FEATURE_NAMES]
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid feature values: {e}"}), 400

    arr = np.array(row).reshape(1, -1)
    model = trained_models[model_name]
    pred = int(model.predict(arr)[0])

    # Confidence
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(arr)[0]
        confidence = round(float(max(proba)) * 100, 2)
    else:
        confidence = None

    return jsonify({
        "model": model_name,
        "prediction": "Fraud" if pred == 1 else "Legitimate",
        "prediction_code": pred,
        "confidence": confidence,
    })


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, port=5000)
