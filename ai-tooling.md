# AI Tooling Disclosure

This document describes how AI coding tools were used while building this
project, per the assignment's AI-tooling disclosure requirement.

## Tools used

- **GitHub Copilot CLI** (agentic terminal assistant) was used throughout
  the project as a pair-programming aid: scaffolding the repository
  structure, writing the data-processing/feature-engineering/model-training
  scripts (`src/`), the Flask web app (`app/`), the test suite (`tests/`),
  the GitHub Actions CI/CD workflow, and this documentation.

## How it was used

- **Requirements gathering**: the assignment PDF was parsed with the tool
  to extract requirements and the dataset link embedded in it.
- **Planning**: a task breakdown with dependencies was drafted
  collaboratively and tracked to keep the multi-day project organized
  (see `plan.md`).
- **Implementation**: the AI tool wrote first-draft code for each stage
  (data split/dedup logic, the `FeatureEngineer` transformer, the
  preprocessing pipeline, model factory functions, the PyTorch MLP wrapper,
  the CV evaluation scripts, the Flask routes/templates, and the pytest
  suite), which was then reviewed, run, and verified by the author at each
  step (checking CV metrics, running the local server, running the test
  suite, testing the deployed smoke test).
- **Debugging**: the AI tool diagnosed a macOS-ARM-specific segfault caused
  by importing `torch` and `lightgbm`/`xgboost`/`catboost` in the same
  process, and implemented the subprocess-isolation workaround
  (`src/cv_evaluate_torch.py`).
- **Documentation**: this file, `evaluation-and-design.md`, and
  `deployed.md` were drafted by the AI tool from the actual experiment
  results (`results/*.json`) and then reviewed for accuracy.

## What was NOT AI-generated

- All modeling decisions (which models to compare, the metric to optimize,
  the train/test split strategy, and the final model selection) were made
  and reviewed by the author, based on the CV results produced by the
  pipeline.
- The dataset itself is a pre-existing public dataset
  (Brazilian Malware Dataset), not AI-generated.
- All code was run and its output (CV scores, test-set metrics, test suite
  results, live smoke tests) was independently verified by the author
  before being accepted, rather than trusted blindly.

## Human oversight

Every AI-suggested change was reviewed before being committed. Model
metrics, test results, and the deployed app's behavior were checked
directly (via `pytest`, manual UI testing, and `curl`/`requests` calls)
rather than assumed correct from the tool's own claims.
