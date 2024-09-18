# Ergonomic Posture Assessment and Personalised Recommendation

Wearable-sensor posture classification system for detecting at-risk working postures and generating ergonomic guidance.

## Scope

- Inputs: joint angles and acceleration signals
- Target postures: excessive lumbar flexion, elevated arm positions, and related strain patterns
- Models: Gradient Boosting and k-NN
- Evaluation: leave-one-subject-out cross-validation

## Planned Workflow

1. Ingest wearable sensor measurements
2. Build posture labels and preprocessing steps
3. Train posture classification models
4. Evaluate cross-user generalisation
5. Convert posture predictions into rule-based ergonomic recommendations

## Repository Structure

- `data/` for raw or cleaned posture datasets
- `notebooks/` for prototyping and model analysis
- `src/` for feature engineering, training, and inference code
- `models/` for trained models
- `reports/` for results and visual outputs
