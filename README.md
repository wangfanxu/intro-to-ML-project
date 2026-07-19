# Intro to ML — PE Malware Classifier

Solo project for Quantic "Introduction to Machine Learning": trains several
classifiers to distinguish malware from goodware using static PE-file
features, selects the best model, and serves it via a Flask web app with
CI/CD.

## Project layout
```
src/            training/eval pipeline (data split, features, models, CV)
app/            Flask web app (manual form, batch upload, /predict API)
tests/          pytest unit + integration tests
scripts/        smoke_test.py (post-deploy check)
data/           demo_test_sample.csv only (full train/test regenerated locally)
models/         production_model.joblib (trained LightGBM pipeline)
results/        cv_results.json, cv_results_torch.json, final_test_results.json
.github/workflows/ci.yml   test -> deploy -> smoke-test pipeline
```

## Setup (full training environment)
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.split_data        # downloads/splits raw dataset -> data/train.csv, data/test.csv
python -m src.cv_evaluate        # 10-fold CV for LR/DT/RF/XGBoost/LightGBM/CatBoost
python -m src.cv_evaluate_torch  # 10-fold CV for the PyTorch MLP (isolated process)
python -m src.train              # retrains best model (LightGBM) on full train set
python -m src.evaluate --data data/test.csv --model models/production_model.joblib
```

## Run the app locally
```bash
pip install -r requirements-app.txt   # lean deployment/runtime deps
python app/app.py                     # http://localhost:5000
# or, production-style:
gunicorn app.app:app --bind 0.0.0.0:5000
```

## Tests
```bash
python -m pytest tests/ -v
```

## Deployment
See `render.yaml` (Render Blueprint) and `deployed.md` for the live URL and
deployment notes. CI/CD (`.github/workflows/ci.yml`) runs tests on every
push/PR, then triggers a Render deploy hook and post-deploy smoke test on
pushes to `main`.

## Documentation
- `evaluation-and-design.md` — model comparison, metrics, design decisions.
- `ai-tooling.md` — how AI tools were used in this project.
- `deployed.md` — live app URL and how to use it.
