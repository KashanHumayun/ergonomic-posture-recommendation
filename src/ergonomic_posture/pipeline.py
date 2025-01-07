from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "trunk_flexion_deg",
    "arm_elevation_deg",
    "neck_flexion_deg",
    "lateral_bend_deg",
    "acceleration_rms",
    "jerk_rms",
    "symmetry_index",
    "load_kg",
    "repetition_rate",
]

POSTURE_LABELS = [
    "neutral",
    "lumbar_flexion_risk",
    "elevated_arm_risk",
    "combined_strain",
]


def generate_synthetic_posture_dataset(
    n_subjects: int = 14,
    windows_per_subject: int = 72,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, Any]] = []
    tasks = ["bed_transfer", "patient_turning", "medication_cart", "reaching", "lifting"]

    for subject_id in range(1, n_subjects + 1):
        trunk_baseline = rng.normal(25.0, 4.0)
        arm_baseline = rng.normal(55.0, 7.0)
        symmetry_baseline = np.clip(rng.normal(0.18, 0.05), 0.05, 0.45)
        for window_id in range(windows_per_subject):
            task_name = tasks[(window_id + subject_id) % len(tasks)]
            risk_profile = rng.choice(
                POSTURE_LABELS,
                p=[0.3, 0.24, 0.24, 0.22],
            )

            feature_row = _sample_posture_features(
                rng=rng,
                trunk_baseline=trunk_baseline,
                arm_baseline=arm_baseline,
                symmetry_baseline=symmetry_baseline,
                risk_profile=risk_profile,
                task_name=task_name,
            )
            rows.append(
                {
                    "subject_id": subject_id,
                    "window_id": f"S{subject_id:02d}_W{window_id:03d}",
                    "task_name": task_name,
                    "posture_label": risk_profile,
                    **feature_row,
                }
            )

    return pd.DataFrame(rows)


def _sample_posture_features(
    rng: np.random.Generator,
    trunk_baseline: float,
    arm_baseline: float,
    symmetry_baseline: float,
    risk_profile: str,
    task_name: str,
) -> dict[str, float]:
    task_load = {
        "bed_transfer": 10.0,
        "patient_turning": 6.5,
        "medication_cart": 2.0,
        "reaching": 3.5,
        "lifting": 8.5,
    }[task_name]

    trunk = rng.normal(trunk_baseline, 6.0)
    arm = rng.normal(arm_baseline, 9.0)
    neck = rng.normal(18.0, 5.0)
    lateral = rng.normal(10.0, 4.0)
    acceleration = np.abs(rng.normal(0.55, 0.18))
    jerk = np.abs(rng.normal(0.35, 0.12))
    symmetry = np.clip(rng.normal(symmetry_baseline, 0.06), 0.02, 0.7)
    load = max(0.5, rng.normal(task_load, 1.5))
    repetition = max(0.1, rng.normal(7.0, 2.5))

    if risk_profile == "lumbar_flexion_risk":
        trunk += rng.normal(28.0, 5.0)
        neck += rng.normal(10.0, 4.0)
        load += rng.normal(3.0, 1.0)
    elif risk_profile == "elevated_arm_risk":
        arm += rng.normal(36.0, 6.0)
        lateral += rng.normal(9.0, 3.0)
        repetition += rng.normal(2.0, 1.0)
    elif risk_profile == "combined_strain":
        trunk += rng.normal(22.0, 4.0)
        arm += rng.normal(28.0, 6.0)
        neck += rng.normal(12.0, 3.0)
        acceleration += rng.normal(0.35, 0.08)
        jerk += rng.normal(0.28, 0.08)
        symmetry += rng.normal(0.12, 0.04)
        load += rng.normal(4.5, 1.2)
        repetition += rng.normal(2.5, 1.0)

    return {
        "trunk_flexion_deg": float(np.clip(trunk, 0.0, 95.0)),
        "arm_elevation_deg": float(np.clip(arm, 0.0, 140.0)),
        "neck_flexion_deg": float(np.clip(neck, 0.0, 70.0)),
        "lateral_bend_deg": float(np.clip(lateral, 0.0, 50.0)),
        "acceleration_rms": float(np.clip(acceleration, 0.02, 3.5)),
        "jerk_rms": float(np.clip(jerk, 0.02, 3.5)),
        "symmetry_index": float(np.clip(symmetry, 0.02, 1.0)),
        "load_kg": float(np.clip(load, 0.5, 20.0)),
        "repetition_rate": float(np.clip(repetition, 0.1, 20.0)),
    }


def evaluate_models(
    frame: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    features = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    labels = frame["posture_label"].to_numpy(dtype=object)
    groups = frame["subject_id"].to_numpy()
    logo = LeaveOneGroupOut()

    estimators = {
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
        "knn": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=7, weights="distance")),
            ]
        ),
    }

    metrics: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    fitted_models: dict[str, Any] = {}

    for model_name, estimator in estimators.items():
        out_of_fold = np.empty(len(frame), dtype=object)
        folds: list[dict[str, float]] = []
        for fold_index, (train_idx, test_idx) in enumerate(logo.split(features, labels, groups), start=1):
            model = clone(estimator)
            model.fit(features[train_idx], labels[train_idx])
            fold_predictions = model.predict(features[test_idx])
            out_of_fold[test_idx] = fold_predictions
            folds.append(
                {
                    "fold": fold_index,
                    "accuracy": float(accuracy_score(labels[test_idx], fold_predictions)),
                    "macro_f1": float(
                        f1_score(labels[test_idx], fold_predictions, average="macro", zero_division=0)
                    ),
                }
            )

        fitted_models[model_name] = clone(estimator).fit(features, labels)
        predictions[model_name] = out_of_fold
        metrics[model_name] = {
            "mean_accuracy": float(np.mean([fold["accuracy"] for fold in folds])),
            "mean_macro_f1": float(np.mean([fold["macro_f1"] for fold in folds])),
            "folds": folds,
            "classification_report": classification_report(labels, out_of_fold, output_dict=True, zero_division=0),
        }

    return metrics, predictions, fitted_models


def recommend_adjustments(row: pd.Series, predicted_label: str) -> str:
    recommendations: list[str] = []
    if predicted_label in {"lumbar_flexion_risk", "combined_strain"} or row["trunk_flexion_deg"] > 45:
        recommendations.append("Reduce trunk flexion and move the task closer to waist height.")
    if predicted_label in {"elevated_arm_risk", "combined_strain"} or row["arm_elevation_deg"] > 85:
        recommendations.append("Adjust arm height or task surface to keep shoulders below high reach.")
    if row["load_kg"] > 10 or row["acceleration_rms"] > 1.0:
        recommendations.append("Slow the transfer pace and use handling support for heavier loads.")
    if row["symmetry_index"] > 0.3 or row["lateral_bend_deg"] > 18:
        recommendations.append("Square the stance to the task and limit asymmetric twisting.")
    if not recommendations:
        recommendations.append("Maintain neutral posture and continue short recovery breaks.")
    return " ".join(recommendations)


def run_demo(
    output_dir: Path,
    n_subjects: int = 14,
    windows_per_subject: int = 72,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = generate_synthetic_posture_dataset(
        n_subjects=n_subjects,
        windows_per_subject=windows_per_subject,
    )
    metrics, predictions, fitted_models = evaluate_models(frame)
    best_model_name = max(metrics, key=lambda name: metrics[name]["mean_accuracy"])
    best_model = fitted_models[best_model_name]

    feature_matrix = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    confidence = best_model.predict_proba(feature_matrix).max(axis=1)

    prediction_frame = frame.copy()
    for model_name, values in predictions.items():
        prediction_frame[f"{model_name}_prediction"] = values
    prediction_frame["selected_prediction"] = prediction_frame[f"{best_model_name}_prediction"]
    prediction_frame["confidence"] = confidence
    prediction_frame["recommendation"] = prediction_frame.apply(
        lambda row: recommend_adjustments(row, str(row["selected_prediction"])),
        axis=1,
    )
    prediction_frame.to_csv(output_dir / "posture_predictions.csv", index=False)

    summary = {
        "samples": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "models": metrics,
        "best_model": {
            "name": best_model_name,
            "accuracy": metrics[best_model_name]["mean_accuracy"],
            "macro_f1": metrics[best_model_name]["mean_macro_f1"],
        },
        "recommendation_examples": prediction_frame[
            ["task_name", "selected_prediction", "recommendation"]
        ].head(5).to_dict(orient="records"),
    }

    joblib.dump(best_model, output_dir / f"{best_model_name}_model.joblib")
    frame.head(500).to_csv(output_dir / "synthetic_posture_sample.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(summary), handle, indent=2)

    return summary


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
