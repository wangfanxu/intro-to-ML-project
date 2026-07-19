# Evaluation & Design

## 1. Problem & data

Binary classification: goodware (0) vs malware (1) from static PE-file
features. Source dataset: [Brazilian Malware Dataset](https://github.com/fabriciojoc/brazilian-malware-dataset)
(50,181 rows, 27 raw features + `Label`).

**Data quality issue found and handled**: the raw CSV contains 2,480 exact
duplicate rows and repeats several unique files (`SHA1`) more than once
(only 43,411 unique hashes across 50,181 rows). Both exact-row duplicates
and duplicate-`SHA1` rows were removed *before* the train/test split
(`src/split_data.py`) to prevent the same underlying binary from appearing
in both the training and hold-out sets, which would otherwise leak
information into the test metrics.

**Split**: stratified 80/20 (`train_test_split(..., stratify=y, random_state=42)`)
on the de-duplicated data → 34,728 train rows / 8,683 test rows. The test
split was set aside and touched exactly once, at the very end, for the
final reported metrics below.

## 2. Feature engineering (`src/features.py`, `src/pipeline.py`)

- Dropped: `Magic`, `PE_TYPE` (zero variance — constant across all rows),
  `SHA1`, `FirstSeenDate` (identifiers / potential leakage, not causally
  predictive of maliciousness).
- `Identify` (free-text packer/compiler signature, ~30% missing): converted
  to a `identify_missing` flag plus binary keyword flags (`has_upx`,
  `has_delphi`, `has_dotnet`, `has_msvc`, `has_installer`,
  `has_packer_generic`); the raw text column is then dropped.
- `ImportedDlls` / `ImportedSymbols` (space-separated token lists): derived
  token-count features (`num_imported_dlls`, `num_imported_symbols`) *and*
  kept the raw text so a `CountVectorizer` (binary, whitespace-tokenized,
  top 40 / 60 tokens respectively) can extract common-token indicator
  features inside the pipeline.
- `Machine` (3 unique values): one-hot encoded.
- All other numeric columns (`Entropy`, `Size`, header/section counts,
  etc.): passed through a `StandardScaler`.

**Leakage safety**: the keyword/count derivations in `FeatureEngineer` use
fixed rules, not statistics learned from data, so they are safe to apply
identically everywhere. The actual data-dependent steps
(`StandardScaler`, `OneHotEncoder`, `CountVectorizer`) live inside the same
`sklearn.Pipeline` passed to `cross_validate`, so scikit-learn fits them
only on each CV fold's training portion and transforms the held-out fold —
satisfying the no-leakage requirement automatically.

## 3. Models compared (10-fold stratified CV on the 34,728-row train set)

| Model | CV AUC (mean ± std) | CV Accuracy (mean ± std) |
|---|---|---|
| Logistic Regression | 0.9795 ± 0.0018 | 0.9375 ± 0.0030 |
| Decision Tree | 0.9876 ± 0.0025 | 0.9785 ± 0.0016 |
| Random Forest | 0.9978 ± 0.0006 | 0.9849 ± 0.0016 |
| PyTorch MLP (hidden=64,32) | 0.9948 ± 0.0010 | 0.9744 ± 0.0020 |
| XGBoost | 0.9984 ± 0.0003 | 0.9866 ± 0.0014 |
| **LightGBM** | **0.9984 ± 0.0002** | **0.9872 ± 0.0014** |
| CatBoost | 0.9977 ± 0.0004 | 0.9847 ± 0.0018 |

Required families covered: linear model (Logistic Regression), tree
(Decision Tree), ensemble/bagging (Random Forest), neural network (PyTorch
MLP), plus three additional gradient-boosting models (XGBoost, LightGBM,
CatBoost) spanning ≥2 additional families relative to the baselines.

Raw numbers: `results/cv_results.json` (sklearn/XGBoost/LightGBM/CatBoost),
`results/cv_results_torch.json` (PyTorch MLP, run in an isolated subprocess
— see §5).

## 4. Model selection & final evaluation

**LightGBM** was selected: highest mean CV AUC (0.9984) with the lowest
variance (std 0.00022) among all seven candidates, and a solid accuracy
(0.9872). Hyperparameters were set to reasonable, lightly-tuned defaults
(`n_estimators=300`, `learning_rate=0.05`, `num_leaves=31`, see
`src/models.py`) rather than an exhaustive grid search, to keep repeated
10-fold CV across seven models tractable in wall-clock time; this is a
deliberate scope trade-off rather than an oversight.

The selected model was retrained on the **full** 34,728-row training set
and evaluated **once** on the untouched 8,683-row hold-out test set:

- **AUC: 0.9986**
- **Accuracy: 0.9879**
- Confusion matrix:

  |  | Predicted goodware | Predicted malware |
  |---|---|---|
  | **True goodware** | 4172 (TN) | 52 (FP) |
  | **True malware** | 53 (FN) | 4406 (TP) |

The hold-out metrics closely track the CV estimates (no meaningful
optimism gap), which corroborates that the de-duplication step
successfully prevented train/test leakage.

Full numbers: `results/final_test_results.json`. Model artifact:
`models/production_model.joblib` (the complete
`FeatureEngineer → ColumnTransformer → LightGBMClassifier` pipeline).

## 5. Implementation notes / design decisions

- **Reproducibility**: a single `RANDOM_STATE = 42` constant
  (`src/config.py`) is used for the train/test split, all CV folds, and
  model training.
- **PyTorch/GBM import-order segfault (macOS ARM)**: importing
  `xgboost`/`lightgbm`/`catboost` in the same process as `torch`, then
  using `torch.utils.data.DataLoader`, reproducibly segfaults on this
  development machine (Apple Silicon). Workaround: the PyTorch MLP's CV
  loop runs in its own subprocess (`python -m src.cv_evaluate_torch`),
  writing its own results file. This does not affect the deployed app,
  which only needs the LightGBM pipeline (no torch import at runtime).
- **Deployment dependency footprint**: `requirements-app.txt` is a lean
  subset of `requirements.txt` containing only what the Flask app and test
  suite need at runtime (excludes `torch`, `xgboost`, `catboost`,
  `matplotlib`, `seaborn`), keeping CI and the production image small and
  avoiding the segfault risk above entirely in production.
