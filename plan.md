# Intro to ML Project Plan

## Scope
Solo submission. Build ML-based static malware detector (goodware vs malware),
select best model, serve via Flask web app, deploy publicly, wrap in CI/CD with
automated testing (unit, integration, smoke).

## Key decisions
- Dataset source: https://github.com/fabriciojoc/brazilian-malware-dataset
  (raw zip: https://raw.githubusercontent.com/fabriciojoc/brazilian-malware-dataset/master/brazilian-malware.zip)
- Deployment host: no strong preference from user — will choose Render (simple,
  good free tier for Flask) unless issues arise.
- Additional models (beyond LR/DT/RF/PyTorch MLP baselines): XGBoost, LightGBM,
  CatBoost (3 families, satisfies >=2-family requirement) — default choice, no
  strong user preference.
- Random seed: fixed constant (42) used throughout for split/CV/model training.

## Success metrics
- Primary: AUC. Secondary: accuracy.
- CV: stratified 10-fold on 80% train split. Test: untouched 20% hold-out,
  evaluated once at the end on the single best CV model.

## High-level phases (see `todos` table for full detail + deps)
1. Dataset acquisition & verification
2. Env/repo scaffold + reproducibility (venv, requirements.txt, seeds)
3. Stratified 80/20 split (before any preprocessing)
4. EDA on train set only
5. Preprocessing pipeline (fit-on-train-fold only)
6. Baselines: Logistic Regression, Decision Tree, Random Forest, PyTorch MLP
   (10-fold CV, record AUC/accuracy mean±std)
7. Additional models: XGBoost, LightGBM, CatBoost (10-fold CV)
8. Model selection -> retrain on full train -> single hold-out test evaluation
9. train.py / eval.py reproducible scripts
10. Flask app: manual entry form (w/ demo prefill), file upload (batch predict +
    optional metrics/confusion matrix if labeled), /predict API, /health
11. Tests: unit (preprocessing + model wrapper), integration (/predict),
    smoke (/health post-deploy)
12. GitHub Actions CI/CD: test gate -> deploy only if pass
13. Deploy to free-tier host, verify public URL
14. Docs: deployed.md, evaluation-and-design.md, ai-tooling.md
15. Repo setup: private repo + add quantic-grader collaborator
16. Record 5-10 min demo video, compile submission PDF (video link + repo link)

## Status tracking
Full task breakdown and dependency graph tracked in the `todos` / `todo_deps`
SQL tables of this session. Query `todos` for current status before starting
new work.

## Progress snapshot (updated)
Done (15/19): dataset-acquire, env-setup, data-split, eda,
preprocessing-pipeline, baseline-models, pytorch-mlp, additional-models,
model-selection, train-eval-scripts, flask-app, unit-tests,
integration-tests, cicd-pipeline, smoke-test.

- **Model selected**: LightGBM. CV AUC 0.9984±0.0002, hold-out test AUC
  0.9986 / Accuracy 0.9879 (see `results/*.json`, `evaluation-and-design.md`).
- **Flask app**: manual form, CSV batch upload w/ metrics, `/predict` JSON
  API, `/health` — all tested locally (21 pytest unit+integration tests
  passing).
- **CI/CD**: `.github/workflows/ci.yml` (test -> Render deploy hook ->
  smoke test), `render.yaml` blueprint, lean `requirements-app.txt`.
- **Repo**: pushed to https://github.com/wangfanxu/intro-to-ML-project
  (no AI co-author trailer per user preference — plain commit messages
  going forward).
- **Docs**: `evaluation-and-design.md` and `ai-tooling.md` written and
  committed. `deployed.md` pending live URL.

In progress (2/19):
- `deployment` — user created a Render account (via Google login);
  connecting GitHub App to Render now, then deploying via Blueprint
  (`render.yaml`). Once live, need to: verify with `scripts/smoke_test.py`,
  set `RENDER_DEPLOY_HOOK_URL` + `APP_URL` GitHub Actions secrets, write
  `deployed.md`.
- `docs-deliverables` — `deployed.md` still pending (needs live URL).

Pending (2/19):
- `repo-setup` — add `quantic-grader` as a GitHub collaborator (needs repo
  Settings > Collaborators, requires user action); confirm repo is private.
- `demo-recording` — 5-10 min demo video + submission PDF. **User asked for
  presentation guidance — offer this once deployment + docs are wrapped
  up, before they record.**
