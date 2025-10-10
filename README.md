# Ergonomic Posture Assessment and Recommendation

This repository documents a small student research project on ergonomic posture assessment using wearable IMU data. The work uses the UCI Smartphone-Based Recognition of Human Activities and Postural Transitions dataset and studies whether the original benchmark labels can be regrouped into broader posture-risk categories that are easier to interpret in an occupational-health context.

Rather than treating the task purely as activity recognition, the pipeline maps the original labels into four ergonomic states and then attaches a short recommendation to the selected prediction. The objective is to explore a simple, reproducible baseline for posture-state assessment rather than to claim a deployable intervention system.

![Posture overview](reports/results/posture_overview.png)

## Research question

Can smartphone IMU benchmark data be reorganised into ergonomic posture-state groups and still support high-quality classification on the official subject-disjoint split?

## Dataset

The project uses the UCI Smartphone-Based Recognition of Human Activities and Postural Transitions dataset stored locally under `data/raw/postural_transitions/`.

Relevant characteristics:
- 30 subjects
- waist-mounted smartphone inertial sensors
- official train/test split with subject separation
- original transition labels remapped into:
  - `static_posture`
  - `dynamic_gait`
  - `sit_stand_transition`
  - `floor_transfer_transition`

## Method

The pipeline:
- loads the released feature matrices from the benchmark split
- maps the original activity and transition labels into ergonomic groups
- trains two baseline classifiers:
  - Histogram Gradient Boosting
  - k-NN with feature scaling
- selects the stronger model by test-set accuracy
- exports predictions and a rule-based recommendation for the selected posture group

The recommendation layer is deliberately simple. It is included to show how model output could be translated into a human-readable ergonomic note, not as a validated intervention protocol.

## Results

Checked-in results from the current run are:

| Model | Evaluation | Accuracy | Macro F1 |
| --- | --- | ---: | ---: |
| HistGradientBoosting | Official subject-disjoint test split | 0.997 | 0.969 |
| k-NN | Official subject-disjoint test split | 0.992 | 0.940 |

## Reproducing the pipeline

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m ergonomic_posture.cli --output-dir reports/results --model-dir models/results
```

## Repository outputs

- `reports/results/metrics.json`
- `reports/results/posture_predictions.csv`
- `reports/results/posture_overview.png`
- `models/results/best_posture_model.joblib`
- `notebooks/real_data_walkthrough.ipynb`

## Limitations

- This is an offline benchmark study, not a live monitoring system.
- The ergonomic categories are manually defined from the benchmark labels.
- The recommendation text is heuristic and should be read as a simple post-processing layer.
- Raw downloaded files live in `data/raw/` and are intentionally excluded from version control.
