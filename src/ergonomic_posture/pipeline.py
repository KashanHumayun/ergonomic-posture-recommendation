from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ACTIVITY_LABEL_MAP = {
    1: "walking",
    2: "walking_upstairs",
    3: "walking_downstairs",
    4: "sitting",
    5: "standing",
    6: "laying",
    7: "stand_to_sit",
    8: "sit_to_stand",
    9: "sit_to_lie",
    10: "lie_to_sit",
    11: "stand_to_lie",
    12: "lie_to_stand",
}

ERGONOMIC_GROUP_MAP = {
    "sitting": "static_posture",
    "standing": "static_posture",
    "laying": "static_posture",
    "walking": "dynamic_gait",
    "walking_upstairs": "dynamic_gait",
    "walking_downstairs": "dynamic_gait",
    "stand_to_sit": "sit_stand_transition",
    "sit_to_stand": "sit_stand_transition",
    "sit_to_lie": "floor_transfer_transition",
    "lie_to_sit": "floor_transfer_transition",
    "stand_to_lie": "floor_transfer_transition",
    "lie_to_stand": "floor_transfer_transition",
}


def find_dataset_root(project_root: Path) -> Path:
    return project_root / "data" / "raw" / "postural_transitions"


def load_feature_split(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray, np.ndarray]:
    X_train = pd.read_csv(root / "Train" / "X_train.txt", sep=r"\s+", header=None)
    X_test = pd.read_csv(root / "Test" / "X_test.txt", sep=r"\s+", header=None)
    y_train = pd.read_csv(root / "Train" / "y_train.txt", header=None)[0].map(ACTIVITY_LABEL_MAP)
    y_test = pd.read_csv(root / "Test" / "y_test.txt", header=None)[0].map(ACTIVITY_LABEL_MAP)
    subjects_train = pd.read_csv(root / "Train" / "subject_id_train.txt", header=None)[0].to_numpy()
    subjects_test = pd.read_csv(root / "Test" / "subject_id_test.txt", header=None)[0].to_numpy()
    return X_train, X_test, y_train, y_test, subjects_train, subjects_test


def map_to_ergonomic_groups(labels: pd.Series) -> pd.Series:
    return labels.map(ERGONOMIC_GROUP_MAP)


def evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    estimators = {
        "gradient_boosting": HistGradientBoostingClassifier(random_state=42),
        "knn": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=7, weights="distance")),
            ]
        ),
    }

    metrics: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    fitted: dict[str, Any] = {}

    for model_name, estimator in estimators.items():
        estimator.fit(X_train, y_train)
        predicted = estimator.predict(X_test)
        metrics[model_name] = {
            "accuracy": float(accuracy_score(y_test, predicted)),
            "macro_f1": float(f1_score(y_test, predicted, average="macro", zero_division=0)),
            "classification_report": classification_report(y_test, predicted, output_dict=True, zero_division=0),
        }
        predictions[model_name] = predicted
        fitted[model_name] = estimator

    return metrics, predictions, fitted


def recommend_adjustments(posture_group: str) -> str:
    if posture_group == "static_posture":
        return "Maintain neutral spine alignment, avoid prolonged static holds, and schedule brief posture resets."
    if posture_group == "dynamic_gait":
        return "Control step cadence, keep loads close to the body, and use hand support when navigating level changes."
    if posture_group == "sit_stand_transition":
        return "Use leg drive during sit-to-stand transitions and control descent during stand-to-sit movements."
    return "Use a supported transition strategy, minimise trunk collapse, and move through floor-level posture changes in stages."


def _plot_overview(metrics: dict[str, Any], predictions: np.ndarray, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    model_names = ["gradient_boosting", "knn"]
    scores = [metrics[name]["accuracy"] for name in model_names]
    macro_f1 = [metrics[name]["macro_f1"] for name in model_names]
    axes[0].bar(model_names, scores, color=["#457b9d", "#2a9d8f"])
    axes[0].plot(model_names, macro_f1, color="#e76f51", marker="o", linewidth=2)
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("Real Data Performance")

    pred_counts = pd.Series(predictions).value_counts()
    axes[1].bar(pred_counts.index, pred_counts.values, color=["#8ecae6", "#ffb703", "#fb8500", "#d62828"])
    axes[1].set_title("Predicted Ergonomic Groups")
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_pipeline(
    project_root: Path,
    output_dir: Path,
    model_dir: Path,
) -> dict[str, Any]:
    dataset_root = find_dataset_root(project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train_raw, y_test_raw, subjects_train, subjects_test = load_feature_split(dataset_root)
    y_train = map_to_ergonomic_groups(y_train_raw)
    y_test = map_to_ergonomic_groups(y_test_raw)

    metrics, predictions, fitted = evaluate_models(X_train, X_test, y_train, y_test)
    best_model_name = max(metrics, key=lambda item: metrics[item]["accuracy"])

    prediction_frame = pd.DataFrame(
        {
            "subject_id": subjects_test,
            "true_activity": y_test_raw,
            "true_group": y_test,
            "gradient_boosting_prediction": predictions["gradient_boosting"],
            "knn_prediction": predictions["knn"],
        }
    )
    prediction_frame["selected_prediction"] = prediction_frame[f"{best_model_name}_prediction"]
    prediction_frame["recommendation"] = prediction_frame["selected_prediction"].map(recommend_adjustments)
    prediction_frame.to_csv(output_dir / "posture_predictions.csv", index=False)

    _plot_overview(metrics, prediction_frame["selected_prediction"].to_numpy(), output_dir / "posture_overview.png")
    joblib.dump(fitted[best_model_name], model_dir / "best_posture_model.joblib")

    summary = {
        "dataset": {
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "train_subjects": int(len(np.unique(subjects_train))),
            "test_subjects": int(len(np.unique(subjects_test))),
        },
        "target_definition": "Real UCI postural-transition labels grouped into ergonomic posture states.",
        "models": metrics,
        "best_model": best_model_name,
    }

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
