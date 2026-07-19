"""Flask web application serving the trained malware-vs-goodware classifier.

Routes
------
GET  /              Manual-entry form + batch-upload form.
POST /predict_form   Handle manual form submission -> render single prediction.
POST /upload         Handle CSV upload -> batch predictions (+ metrics if the
                     file includes the ground-truth 'Label' column).
POST /predict        JSON API: accepts a single feature dict or a list of
                     feature dicts, returns predictions as JSON.
GET  /health         Liveness/readiness probe used by smoke tests & the
                     hosting platform.
"""

import io
import json
import os
import sys

import pandas as pd
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ALL_INPUT_COLS, TARGET_COL  # noqa: E402
from src.model_wrapper import ModelWrapper  # noqa: E402

APP_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB upload cap

_model = None


def get_model():
    """Lazily loads the model once per process (avoids import cost at
    module-import time, which is helpful for testing)."""
    global _model
    if _model is None:
        _model = ModelWrapper()
    return _model


def load_demo_rows():
    demo_path = os.path.join(APP_DIR, "demo_rows.json")
    with open(demo_path) as f:
        return json.load(f)


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        columns=ALL_INPUT_COLS,
        demo_rows=load_demo_rows(),
        result=None,
        upload_result=None,
    )


@app.route("/predict_form", methods=["POST"])
def predict_form():
    feature_dict = {col: request.form.get(col, "") for col in ALL_INPUT_COLS}
    try:
        numeric_cols = [c for c in ALL_INPUT_COLS if c not in
                        ("Identify", "ImportedDlls", "ImportedSymbols",
                         "FirstSeenDate", "SHA1")]
        for col in numeric_cols:
            if col == "Machine" or col == "PE_TYPE":
                continue
            val = feature_dict[col]
            feature_dict[col] = float(val) if val not in ("", None) else 0.0

        result = get_model().predict_one(feature_dict)
        error = None
    except Exception as exc:  # noqa: BLE001
        result = None
        error = str(exc)

    return render_template(
        "index.html",
        columns=ALL_INPUT_COLS,
        demo_rows=load_demo_rows(),
        result=result,
        error=error,
        form_values=feature_dict,
        upload_result=None,
    )


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    error = None
    upload_result = None

    if file is None or file.filename == "":
        error = "No file selected."
    else:
        try:
            content = file.read()
            df = pd.read_csv(io.BytesIO(content))
            has_labels = TARGET_COL in df.columns

            if has_labels:
                metrics = get_model().evaluate_df(df)
                upload_result = {
                    "has_labels": True,
                    "n_rows": len(df),
                    "auc": round(metrics["auc"], 4),
                    "accuracy": round(metrics["accuracy"], 4),
                    "confusion_matrix": metrics["confusion_matrix"],
                    "predictions": metrics["predictions"][:20],
                }
            else:
                preds = get_model().predict_df(df)
                upload_result = {
                    "has_labels": False,
                    "n_rows": len(df),
                    "predictions": preds[:20],
                }
        except Exception as exc:  # noqa: BLE001
            error = str(exc)

    return render_template(
        "index.html",
        columns=ALL_INPUT_COLS,
        demo_rows=load_demo_rows(),
        result=None,
        upload_result=upload_result,
        error=error,
    )


@app.route("/predict", methods=["POST"])
def predict_api():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be JSON"}), 400

    instances = payload if isinstance(payload, list) else [payload]

    try:
        df = pd.DataFrame(instances)
        predictions = get_model().predict_df(df)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400

    if isinstance(payload, list):
        return jsonify({"predictions": predictions})
    return jsonify(predictions[0])


@app.route("/health", methods=["GET"])
def health():
    try:
        get_model()
        return jsonify({"status": "ok"}), 200
    except Exception as exc:  # noqa: BLE001
        return jsonify({"status": "error", "detail": str(exc)}), 503


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
