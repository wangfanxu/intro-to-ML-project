# Deployed Application

**Live URL:** https://intro-to-ml-malware-detector.onrender.com

Hosted on [Render](https://render.com) (free tier web service), deployed
via the `render.yaml` Blueprint in this repo. The service builds with
`pip install -r requirements-app.txt` and runs
`gunicorn app.app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`.

> Note: the free Render instance spins down after ~15 minutes of
> inactivity. The first request after idle time may take 30-50 seconds to
> respond while the instance wakes up — this is expected free-tier
> behavior, not an application issue.

## How to use it

1. **Manual single prediction** — visit the URL, click "Load demo goodware
   row" or "Load demo malware row" to auto-fill the 27-feature form (or
   enter your own values), then click **Predict**.
2. **Batch prediction / evaluation** — upload a CSV with the 27 raw
   feature columns (optionally including a `Label` column). If `Label` is
   present, the app also reports AUC, accuracy, and a confusion matrix.
   A ready-made sample is included in the repo at
   `data/demo_test_sample.csv` (300 held-out rows, never used in
   training).
3. **JSON API** — `POST /predict` with a JSON object (single instance) or
   a JSON array (batch) of the 27 feature keys; returns predicted
   label/probability for each.
   ```bash
   curl -X POST https://intro-to-ml-malware-detector.onrender.com/predict \
     -H "Content-Type: application/json" \
     -d @app/demo_rows.json   # (send just one of the two rows in practice)
   ```
4. **Health check** — `GET /health` returns `{"status": "ok"}` when the
   model is loaded and the service is ready.

## Model served

LightGBM classifier (see `evaluation-and-design.md` for full comparison
and metrics): 10-fold CV AUC 0.9984 ± 0.0002, hold-out test AUC 0.9986 /
accuracy 0.9879. Artifact: `models/production_model.joblib`.

## CI/CD

`.github/workflows/ci.yml` runs the full pytest suite on every push/PR.
On pushes to `main` that pass tests, it triggers a Render deploy hook and
then runs `scripts/smoke_test.py` against the live URL to confirm the new
deployment is healthy and returning correct predictions.
