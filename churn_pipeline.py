"""Train and run a customer subscription-churn prediction pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_VALUES = {
    "0": 0,
    "false": 0,
    "no": 0,
    "active": 0,
    "retained": 0,
    "stay": 0,
    "stayed": 0,
    "1": 1,
    "true": 1,
    "yes": 1,
    "churn": 1,
    "churned": 1,
    "cancel": 1,
    "cancelled": 1,
    "canceled": 1,
}


def normalize_target(values: pd.Series) -> pd.Series:
    """Convert common churn labels into 0 (stays) and 1 (cancels)."""
    normalized = values.astype("string").str.strip().str.lower().map(TARGET_VALUES)
    if normalized.isna().any():
        invalid = values[normalized.isna()].drop_duplicates().astype(str).tolist()
        raise ValueError(
            "Target values must be 0/1, yes/no, true/false, or common "
            f"churn labels; unrecognized values: {invalid[:10]}"
        )
    return normalized.astype("int8")


def build_model(features: pd.DataFrame) -> Pipeline:
    numeric_columns = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )

    preprocessing = ColumnTransformer(transformers=transformers)
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(class_weight="balanced", max_iter=1000),
            ),
        ]
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    data = pd.read_csv(path)
    if data.empty:
        raise ValueError(f"CSV file has no data rows: {path}")
    if data.columns.duplicated().any():
        raise ValueError("CSV must not contain duplicate column names.")
    return data


def train(
    data_path: Path,
    target_column: str,
    model_path: Path,
    id_column: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    date_column: str | None = None,
) -> dict[str, Any]:
    data = _read_csv(data_path)
    if target_column not in data.columns:
        raise ValueError(f"Target column {target_column!r} is not present in the CSV.")
    if id_column and id_column not in data.columns:
        raise ValueError(f"ID column {id_column!r} is not present in the CSV.")
    if date_column and date_column not in data.columns:
        raise ValueError(f"Date column {date_column!r} is not present in the CSV.")
    if date_column and date_column in {target_column, id_column}:
        raise ValueError("The date column must be different from the target and ID columns.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1 (exclusive).")

    target = normalize_target(data[target_column])
    excluded_columns = [target_column]
    if id_column:
        excluded_columns.append(id_column)
    if date_column:
        excluded_columns.append(date_column)
    features = data.drop(columns=excluded_columns)
    if features.shape[1] == 0:
        raise ValueError("At least one feature column is required.")
    if target.nunique() != 2 or target.value_counts().min() < 2:
        raise ValueError(
            "Training requires both churn classes, with at least two rows per class."
        )

    if date_column:
        snapshot_dates = pd.to_datetime(data[date_column], errors="coerce", utc=True)
        if snapshot_dates.isna().any():
            invalid_rows = snapshot_dates[snapshot_dates.isna()].index[:10].tolist()
            raise ValueError(
                f"Date column {date_column!r} contains missing or invalid dates "
                f"at row indexes: {invalid_rows}."
            )
        unique_dates = snapshot_dates.drop_duplicates().sort_values()
        if len(unique_dates) < 2:
            raise ValueError(
                "A chronological holdout requires at least two distinct snapshot dates."
            )
        test_date_count = max(1, int(np.ceil(len(unique_dates) * test_size)))
        test_start = unique_dates.iloc[-test_date_count]
        test_mask = snapshot_dates >= test_start
        x_train, x_test = features.loc[~test_mask], features.loc[test_mask]
        y_train, y_test = target.loc[~test_mask], target.loc[test_mask]
        validation_strategy = "chronological"
        test_period_start = test_start.isoformat()
        test_period_end = snapshot_dates.max().isoformat()
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=random_state,
            stratify=target,
        )
        validation_strategy = "stratified_random"
        test_period_start = None
        test_period_end = None

    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError(
            "The training and holdout partitions must each contain both churn "
            "classes. Adjust --test-size or provide more historical data."
        )

    model = build_model(x_train)
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype("int8")
    baseline_class = int(y_train.mean() >= 0.5)
    baseline_predictions = np.full(len(y_test), baseline_class, dtype="int8")
    metrics: dict[str, Any] = {
        "test_rows": int(len(y_test)),
        "train_rows": int(len(y_train)),
        "positive_rate": float(target.mean()),
        "test_positive_rate": float(y_test.mean()),
        "validation_strategy": validation_strategy,
        "test_period_start": test_period_start,
        "test_period_end": test_period_end,
        "majority_baseline_class": baseline_class,
        "majority_baseline_accuracy": float(
            accuracy_score(y_test, baseline_predictions)
        ),
        "average_precision_baseline": float(y_test.mean()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "predicted_positive_rate": float(predictions.mean()),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "confusion_matrix_labels_0_then_1": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["negative", "positive"],
            zero_division=0,
            output_dict=True,
        ),
    }
    if id_column:
        metrics["train_customers"] = int(data.loc[x_train.index, id_column].nunique())
        metrics["test_customers"] = int(data.loc[x_test.index, id_column].nunique())
        metrics["customer_overlap"] = int(
            len(
                set(data.loc[x_train.index, id_column])
                & set(data.loc[x_test.index, id_column])
            )
        )

    artifact = {
        "model": model,
        "feature_columns": features.columns.tolist(),
        "target_column": target_column,
        "id_column": id_column,
        "date_column": date_column,
        "threshold": 0.5,
        "metrics": metrics,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    return metrics


def score(data_path: Path, model_path: Path, output_path: Path) -> int:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or not {
        "model",
        "feature_columns",
        "threshold",
    }.issubset(artifact):
        raise ValueError("Model file is not a supported churn pipeline artifact.")

    data = _read_csv(data_path)
    feature_columns = artifact["feature_columns"]
    id_column = artifact.get("id_column")
    date_column = artifact.get("date_column")
    missing = [column for column in feature_columns if column not in data.columns]
    unexpected = [
        column
        for column in data.columns
        if column not in feature_columns and column not in {id_column, date_column}
    ]
    if missing or unexpected:
        raise ValueError(
            f"Scoring columns do not match training. Missing: {missing}; "
            f"unexpected: {unexpected}."
        )

    probabilities = artifact["model"].predict_proba(data[feature_columns])[:, 1]
    target_column = artifact.get("target_column", "churn")
    if target_column == "churn":
        probability_column = "churn_probability"
        prediction_column = "predicted_to_cancel"
    else:
        probability_column = f"{target_column}_probability"
        prediction_column = f"predicted_{target_column}"
    output = pd.DataFrame(
        {
            probability_column: probabilities,
            prediction_column: probabilities >= artifact["threshold"],
        }
    )
    if id_column:
        output.insert(0, id_column, data[id_column].to_numpy())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return len(output)


def generate_demo(output_path: Path, rows: int = 800, random_state: int = 7) -> None:
    if rows < 20:
        raise ValueError("Demo data requires at least 20 rows.")
    rng = np.random.default_rng(random_state)
    tenure = rng.integers(1, 73, size=rows)
    monthly_charges = np.round(rng.normal(68, 22, size=rows).clip(15, 160), 2)
    contract = rng.choice(
        ["month-to-month", "one-year", "two-year"],
        size=rows,
        p=[0.55, 0.28, 0.17],
    )
    payment = rng.choice(
        ["card", "bank-transfer", "electronic-check"], size=rows
    )
    support_tickets = rng.poisson(0.8, size=rows)
    usage_change = np.round(rng.normal(-3, 18, size=rows), 1)
    late_payments = rng.poisson(0.4, size=rows)
    autopay = rng.choice(["yes", "no"], size=rows, p=[0.62, 0.38])
    plan = rng.choice(["basic", "standard", "premium"], size=rows)

    log_odds = (
        -1.25
        + 0.85 * (contract == "month-to-month")
        - 0.65 * (contract == "two-year")
        - 0.025 * tenure
        + 0.55 * (payment == "electronic-check")
        + 0.35 * (autopay == "no")
        + 0.32 * support_tickets
        - 0.018 * usage_change
        + 0.42 * late_payments
        + 0.008 * (monthly_charges - 68)
    )
    churn_probability = 1 / (1 + np.exp(-log_odds))
    churn = rng.binomial(1, churn_probability)
    data = pd.DataFrame(
        {
            "customer_id": [f"C{index:05d}" for index in range(1, rows + 1)],
            "tenure_months": tenure,
            "monthly_charges": monthly_charges,
            "contract_type": contract,
            "payment_method": payment,
            "support_tickets_90d": support_tickets,
            "usage_change_pct": usage_change,
            "late_payments_6m": late_payments,
            "autopay": autopay,
            "plan_type": plan,
            "churn": churn,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and score a subscription churn prediction model."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="Train and evaluate a model.")
    train_parser.add_argument("--data", type=Path, required=True)
    train_parser.add_argument("--target", default="churn")
    train_parser.add_argument("--id-column", default="customer_id")
    train_parser.add_argument("--model-out", type=Path, default=Path("artifacts/churn_model.joblib"))
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.add_argument(
        "--date-column",
        help="Snapshot date column; enables a chronological holdout and is excluded from features.",
    )

    score_parser = commands.add_parser("score", help="Score new customers.")
    score_parser.add_argument("--data", type=Path, required=True)
    score_parser.add_argument("--model", type=Path, default=Path("artifacts/churn_model.joblib"))
    score_parser.add_argument("--output", type=Path, default=Path("scored_customers.csv"))

    demo_parser = commands.add_parser("generate-demo", help="Create synthetic sample data.")
    demo_parser.add_argument("--output", type=Path, default=Path("data/demo_customers.csv"))
    demo_parser.add_argument("--rows", type=int, default=800)
    demo_parser.add_argument("--random-state", type=int, default=7)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "train":
        metrics = train(
            args.data,
            args.target,
            args.model_out,
            args.id_column,
            args.test_size,
            args.random_state,
            args.date_column,
        )
        print(json.dumps(metrics, indent=2))
        print(f"Model saved to {args.model_out}")
    elif args.command == "score":
        rows = score(args.data, args.model, args.output)
        print(f"Scored {rows} customers; results saved to {args.output}")
    elif args.command == "generate-demo":
        generate_demo(args.output, args.rows, args.random_state)
        print(f"Synthetic customer data saved to {args.output}")


if __name__ == "__main__":
    main()
