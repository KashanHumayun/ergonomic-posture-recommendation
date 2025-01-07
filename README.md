# Ergonomic Posture Assessment and Personalised Recommendation

Wearable-sensor posture assessment pipeline for detecting at-risk postures and translating them into practical ergonomic guidance. The repo ships with a synthetic demo dataset so the full workflow can be run immediately while keeping the code structure ready for real nursing-task or occupational-health data.

## What This Repo Includes

- Posture-risk classification with Gradient Boosting and k-NN
- Leave-one-subject-out cross-validation for cross-user generalisation
- Features representing trunk flexion, arm elevation, asymmetry, load, and motion intensity
- Rule-based recommendation engine that turns posture predictions into actionable guidance
- CLI that exports predictions, model artifacts, and metrics

## Quick Start

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m ergonomic_posture.cli --output-dir reports/demo
```

## Project Structure

- `src/ergonomic_posture/` classification and recommendation logic
- `tests/` smoke test for the demo run
- `reports/` generated outputs such as metrics and recommendations
- `data/` place real wearable or joint-angle datasets here
- `models/` reserved for persisted trained estimators

## Replacing The Demo Data

The current synthetic generator creates subject-specific posture windows with neutral, lumbar-flexion, elevated-arm, and combined-strain patterns. To switch to real data, supply a dataframe with:

- `subject_id`
- one row per posture window or task segment
- the feature columns listed in `src/ergonomic_posture/pipeline.py`
- a target label column named `posture_label`

## Output Artifacts

Running the demo writes:

- `reports/demo/metrics.json`
- `reports/demo/posture_predictions.csv`
- `reports/demo/gradient_boosting_model.joblib` or the best model artifact
- `reports/demo/synthetic_posture_sample.csv`
