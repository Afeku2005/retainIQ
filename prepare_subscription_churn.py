"""Create point-in-time 30-day subscription cancellation examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TARGET_COLUMN = "churn_next_30d"
ELIGIBLE_STATUSES = {"active", "paused"}
HISTORICAL_EVENT_TYPES = ("paused", "resumed", "reactivated", "cancelled")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    data = pd.read_csv(path)
    if data.empty:
        raise ValueError(f"CSV file has no data rows: {path}")
    if data.columns.duplicated().any():
        raise ValueError(f"CSV file has duplicate column names: {path}")
    return data


def build_dataset(
    snapshots_path: Path,
    subscriptions_path: Path,
    history_path: Path,
    output_path: Path,
    horizon_days: int = 30,
) -> dict[str, Any]:
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1.")

    snapshots = _read_csv(snapshots_path)
    subscriptions = _read_csv(subscriptions_path)
    history = _read_csv(history_path)

    required = {
        "snapshot_dates.csv": (snapshots, {"snapshot_date", "label_window_end", "label_complete"}),
        "subscriptions.csv": (
            subscriptions,
            {"customer_id", "subscription_start_date"},
        ),
        "status_history.csv": (
            history,
            {"customer_id", "event_date", "event_type", "status_after"},
        ),
    }
    for file_name, (data, required_columns) in required.items():
        missing = sorted(required_columns - set(data.columns))
        if missing:
            raise ValueError(f"{file_name} is missing required columns: {missing}.")

    if subscriptions["customer_id"].isna().any():
        raise ValueError("Subscription customer_id values must not be missing.")
    if history["customer_id"].isna().any():
        raise ValueError("Status-history customer_id values must not be missing.")
    if subscriptions["customer_id"].duplicated().any():
        raise ValueError("subscriptions.csv must have one row per customer_id.")

    snapshots = snapshots.copy()
    subscriptions = subscriptions.copy()
    history = history.copy()
    snapshots["snapshot_date"] = pd.to_datetime(
        snapshots["snapshot_date"], errors="coerce"
    ).dt.normalize()
    snapshots["label_window_end"] = pd.to_datetime(
        snapshots["label_window_end"], errors="coerce"
    ).dt.normalize()
    if snapshots[["snapshot_date", "label_window_end"]].isna().any().any():
        raise ValueError("Snapshot date columns contain missing or invalid dates.")
    if snapshots["snapshot_date"].duplicated().any():
        raise ValueError("snapshot_dates.csv must have one row per snapshot_date.")
    complete_values = snapshots["label_complete"].astype("string").str.strip().str.lower()
    invalid_complete = ~complete_values.isin({"true", "false", "1", "0", "yes", "no"})
    if invalid_complete.any():
        raise ValueError("label_complete must contain true/false values.")
    snapshots["label_complete"] = complete_values.isin({"true", "1", "yes"})
    excluded_incomplete_snapshots = int((~snapshots["label_complete"]).sum())
    snapshots = snapshots.loc[snapshots["label_complete"]].sort_values(
        "snapshot_date"
    )
    if snapshots.empty:
        raise ValueError("No complete label windows are available.")
    expected_window_days = (
        snapshots["label_window_end"] - snapshots["snapshot_date"]
    ).dt.days
    if not expected_window_days.eq(horizon_days).all():
        raise ValueError(
            f"Every complete label window must be exactly {horizon_days} days "
            "after its snapshot date."
        )

    subscriptions["subscription_start_date"] = pd.to_datetime(
        subscriptions["subscription_start_date"], errors="coerce"
    ).dt.normalize()
    history["event_date"] = pd.to_datetime(
        history["event_date"], errors="coerce"
    ).dt.normalize()
    if subscriptions["subscription_start_date"].isna().any():
        raise ValueError("subscription_start_date contains missing or invalid dates.")
    if history["event_date"].isna().any():
        raise ValueError("status_history event_date contains missing or invalid dates.")

    subscription_ids = set(subscriptions["customer_id"])
    history_ids = set(history["customer_id"])
    if subscription_ids != history_ids:
        missing_history = len(subscription_ids - history_ids)
        unknown_history = len(history_ids - subscription_ids)
        raise ValueError(
            "Subscription and status-history customer IDs must match exactly; "
            f"missing history: {missing_history}, unknown history: {unknown_history}."
        )

    history["event_type"] = history["event_type"].astype("string").str.strip().str.lower()
    history["status_after"] = history["status_after"].astype("string").str.strip().str.lower()
    max_label_end = snapshots["label_window_end"].max()
    if history["event_date"].max() < max_label_end:
        raise ValueError(
            "status_history.csv does not cover the end of every complete "
            f"label window ({max_label_end.date()})."
        )

    subscription_starts = subscriptions.set_index("customer_id")[
        "subscription_start_date"
    ]
    examples: list[pd.DataFrame] = []
    for snapshot in snapshots.itertuples(index=False):
        snapshot_date = snapshot.snapshot_date
        window_end = snapshot.label_window_end

        as_of_events = history.loc[history["event_date"] <= snapshot_date].sort_values(
            "event_date", kind="stable"
        )
        status_at_snapshot = as_of_events.drop_duplicates(
            "customer_id", keep="last"
        ).set_index("customer_id")["status_after"]
        eligible_ids = status_at_snapshot[
            status_at_snapshot.isin(ELIGIBLE_STATUSES)
        ].index
        if (subscription_starts.reindex(eligible_ids) > snapshot_date).any():
            raise ValueError(
                f"Status history marks a subscription active before its start "
                f"date at snapshot {snapshot_date.date()}."
            )
        prior_events = history.loc[history["event_date"] < snapshot_date]
        prior_90_days = prior_events.loc[
            prior_events["event_date"] > snapshot_date - pd.Timedelta(days=90)
        ]

        frame = pd.DataFrame({"customer_id": eligible_ids})
        frame["snapshot_date"] = snapshot_date
        frame["status_at_snapshot"] = frame["customer_id"].map(status_at_snapshot)
        frame["tenure_days"] = (
            snapshot_date - frame["customer_id"].map(subscription_starts)
        ).dt.days

        for event_type in HISTORICAL_EVENT_TYPES:
            all_counts = prior_events.loc[
                prior_events["event_type"] == event_type
            ].groupby("customer_id").size()
            recent_counts = prior_90_days.loc[
                prior_90_days["event_type"] == event_type
            ].groupby("customer_id").size()
            frame[f"{event_type}_events_before"] = (
                frame["customer_id"].map(all_counts).fillna(0).astype("int64")
            )
            frame[f"{event_type}_events_prior_90d"] = (
                frame["customer_id"].map(recent_counts).fillna(0).astype("int64")
            )

        prior_reactivations = prior_events.loc[
            prior_events["event_type"] == "reactivated"
        ].groupby("customer_id")["event_date"].max()
        last_reactivation_dates = pd.to_datetime(
            frame["customer_id"].map(prior_reactivations.to_dict())
        )
        frame["days_since_last_reactivation"] = (
            snapshot_date - last_reactivation_dates
        ).dt.days

        cancellation_ids = set(
            history.loc[
                (history["event_type"] == "cancelled")
                & (history["event_date"] > snapshot_date)
                & (history["event_date"] <= window_end),
                "customer_id",
            ]
        )
        frame[TARGET_COLUMN] = frame["customer_id"].isin(cancellation_ids).astype(
            "int8"
        )
        examples.append(frame)

    if not examples:
        raise ValueError("No snapshot rows were created.")
    result = pd.concat(examples, ignore_index=True)
    if result.empty:
        raise ValueError("No active or paused subscriptions were found at snapshots.")
    if result.duplicated(["customer_id", "snapshot_date"]).any():
        raise ValueError("Generated data has duplicate customer/snapshot rows.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    positive_rate = float(result[TARGET_COLUMN].mean())
    return {
        "rows": len(result),
        "customers": int(result["customer_id"].nunique()),
        "complete_snapshots": int(len(snapshots)),
        "excluded_incomplete_snapshots": excluded_incomplete_snapshots,
        "snapshot_start": snapshots["snapshot_date"].min().date().isoformat(),
        "snapshot_end": snapshots["snapshot_date"].max().date().isoformat(),
        "label_window_days": horizon_days,
        "positive_rows": int(result[TARGET_COLUMN].sum()),
        "positive_rate": positive_rate,
        "target": TARGET_COLUMN,
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build real 30-day subscription-cancellation training rows."
    )
    parser.add_argument(
        "--snapshots",
        type=Path,
        default=Path("data/raw/snapshot_dates.csv"),
    )
    parser.add_argument(
        "--subscriptions",
        type=Path,
        default=Path("data/raw/subscriptions.csv"),
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("data/raw/status_history.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/subscription-churn-training.csv"),
    )
    parser.add_argument("--horizon-days", type=int, default=30)
    args = parser.parse_args()
    summary = build_dataset(
        args.snapshots,
        args.subscriptions,
        args.history,
        args.output,
        args.horizon_days,
    )
    print(json.dumps(summary, indent=2))
    print(
        "Eligible rows are subscriptions active or paused at the snapshot. "
        "Only point-in-time history features are included."
    )


if __name__ == "__main__":
    main()
