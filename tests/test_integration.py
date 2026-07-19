"""Integration tests for the Flask app (app/app.py) using Flask's built-in
test client. Exercises the routes end-to-end: JSON /predict API, batch CSV
/upload, manual /predict_form, and the /health liveness probe.
"""

import io
import json
import os

import pytest

from src.model_wrapper import DEFAULT_MODEL_PATH

pytestmark = pytest.mark.skipif(
    not os.path.exists(DEFAULT_MODEL_PATH),
    reason="production_model.joblib not found; run `python -m src.train` first",
)


@pytest.fixture(scope="module")
def client():
    from app.app import app

    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture(scope="module")
def demo_rows():
    app_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app")
    with open(os.path.join(app_dir, "demo_rows.json")) as f:
        return json.load(f)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_index_loads(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"PE Malware Classifier" in resp.data


def test_predict_api_single_instance(client, demo_rows):
    resp = client.post("/predict", json=demo_rows["malware"])
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["label"] == "malware"
    assert body["prediction"] == 1


def test_predict_api_goodware_instance(client, demo_rows):
    resp = client.post("/predict", json=demo_rows["goodware"])
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["label"] == "goodware"
    assert body["prediction"] == 0


def test_predict_api_batch_list(client, demo_rows):
    resp = client.post("/predict", json=[demo_rows["goodware"], demo_rows["malware"]])
    assert resp.status_code == 200
    body = resp.get_json()
    assert len(body["predictions"]) == 2


def test_predict_api_rejects_non_json(client):
    resp = client.post("/predict", data="not json", content_type="text/plain")
    assert resp.status_code == 400


def test_predict_api_missing_fields(client):
    resp = client.post("/predict", json={"Entropy": 5.0})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_upload_csv_with_labels_returns_metrics(client):
    from src.config import DEMO_SAMPLE_CSV

    with open(DEMO_SAMPLE_CSV, "rb") as f:
        data = {"file": (io.BytesIO(f.read()), "demo_test_sample.csv")}
        resp = client.post("/upload", data=data, content_type="multipart/form-data")

    assert resp.status_code == 200
    assert b"AUC" in resp.data
    assert b"Accuracy" in resp.data


def test_upload_no_file_shows_error(client):
    resp = client.post("/upload", data={}, content_type="multipart/form-data")
    assert resp.status_code == 200
    assert b"No file selected" in resp.data


def test_predict_form_manual_entry(client, demo_rows):
    resp = client.post("/predict_form", data=demo_rows["malware"])
    assert resp.status_code == 200
    assert b"MALWARE" in resp.data
