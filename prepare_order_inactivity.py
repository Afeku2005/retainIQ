"""Build a privacy-minimized, clearly labeled purchase-inactivity proxy dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def build_dataset(
    customers_path: Path,
    orders_path: Path,
    output_path: Path,
    snapshot_date: str = "2026-06-30",
    horizon_days: int = 30,
) -> dict[str, Any]:
    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1.")
    if not customers_path.is_file():
        raise FileNotFoundError(f"Customer CSV not found: {customers_path}")
    if not orders_path.is_file():
        raise FileNotFoundError(f"Orders CSV not found: {orders_path}")

    customers = pd.read_csv(customers_path)
    orders = pd.read_csv(orders_path)
    customer_columns = {"customer_id", "signup_date", "segment", "city", "country"}
    order_columns = {"customer_id", "date", "product", "quantity", "total"}
    missing_customer_columns = sorted(customer_columns - set(customers.columns))
    missing_order_columns = sorted(order_columns - set(orders.columns))
    if missing_customer_columns or missing_order_columns:
        raise ValueError(
            f"Missing customer columns: {missing_customer_columns}; "
            f"missing order columns: {missing_order_columns}."
        )
    if customers.empty or orders.empty:
        raise ValueError("Both CSV files must contain data rows.")
    if customers["customer_id"].isna().any() or orders["customer_id"].isna().any():
        raise ValueError("Customer IDs must not be missing.")
    if customers["customer_id"].duplicated().any():
        raise ValueError("Customer CSV must contain one row per unique customer_id.")

    snapshot = pd.Timestamp(snapshot_date).normalize()
    horizon_end = snapshot + pd.Timedelta(days=horizon_days)
    signup_dates = pd.to_datetime(customers["signup_date"], errors="coerce")
    order_dates = pd.to_datetime(orders["date"], errors="coerce")
    if signup_dates.isna().any():
        raise ValueError("Customer signup_date contains missing or invalid dates.")
    if order_dates.isna().any():
        raise ValueError("Order date contains missing or invalid dates.")
    if (signup_dates.dt.normalize() > snapshot).any():
        raise ValueError("Some customer signup dates occur after the snapshot date.")
    if order_dates.max().normalize() < horizon_end:
        raise ValueError(
            f"Orders data does not reach the end of the {horizon_days}-day "
            f"label window ({horizon_end.date()})."
        )

    for column in ("quantity", "total"):
        orders[column] = pd.to_numeric(orders[column], errors="coerce")
        if orders[column].isna().any():
            raise ValueError(f"Order column {column!r} contains missing or invalid values.")

    customers = customers.copy()
    customers["signup_date"] = signup_dates.dt.normalize()
    orders = orders.copy()
    orders["date"] = order_dates.dt.normalize()

    unknown_ids = set(orders["customer_id"]) - set(customers["customer_id"])
    if unknown_ids:
        raise ValueError(
            f"Orders contain {len(unknown_ids)} customer IDs missing from customer data."
        )

    before = orders.loc[orders["date"] <= snapshot]
    after = orders.loc[
        (orders["date"] > snapshot) & (orders["date"] <= horizon_end)
    ]
    windows = {
        "prior_30d": before.loc[before["date"] > snapshot - pd.Timedelta(days=30)],
        "prior_90d": before.loc[before["date"] > snapshot - pd.Timedelta(days=90)],
    }

    result = customers[["customer_id", "segment", "city", "country"]].copy()
    result["account_age_days"] = (snapshot - customers["signup_date"]).dt.days

    result["orders_before_snapshot"] = before.groupby("customer_id").size().reindex(
        result["customer_id"], fill_value=0
    ).to_numpy()
    result["spend_before_snapshot"] = before.groupby("customer_id")["total"].sum().reindex(
        result["customer_id"], fill_value=0
    ).to_numpy()
    last_order = before.groupby("customer_id")["date"].max()
    result["days_since_last_order"] = (
        snapshot - result["customer_id"].map(last_order)
    ).dt.days
    result["has_prior_order"] = result["days_since_last_order"].notna().astype("int8")

    for window_name, window_orders in windows.items():
        grouped = window_orders.groupby("customer_id")
        result[f"orders_{window_name}"] = grouped.size().reindex(
            result["customer_id"], fill_value=0
        ).to_numpy()
        result[f"spend_{window_name}"] = grouped["total"].sum().reindex(
            result["customer_id"], fill_value=0
        ).to_numpy()
    product_counts = windows["prior_90d"].groupby("customer_id")["product"].nunique()
    result["unique_products_prior_90d"] = product_counts.reindex(
        result["customer_id"], fill_value=0
    ).to_numpy()

    customers_with_order = set(after["customer_id"])
    result["no_order_record_next_30d_proxy"] = (
        ~result["customer_id"].isin(customers_with_order)
    ).astype("int8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    positive_count = int(result["no_order_record_next_30d_proxy"].sum())
    return {
        "rows": len(result),
        "features": result.shape[1] - 2,
        "snapshot_date": snapshot.date().isoformat(),
        "label_window_end": horizon_end.date().isoformat(),
        "horizon_days": horizon_days,
        "customers_with_no_order_record": positive_count,
        "customers_with_an_order_record": len(result) - positive_count,
        "target_name": "no_order_record_next_30d_proxy",
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a purchase-inactivity proxy dataset (not subscription churn)."
    )
    parser.add_argument(
        "--customers",
        type=Path,
        default=Path("data/raw/sample-customers.csv"),
    )
    parser.add_argument(
        "--orders",
        type=Path,
        default=Path("data/raw/sample-orders.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/sample-order-inactivity-proxy.csv"),
    )
    parser.add_argument("--snapshot-date", default="2026-06-30")
    parser.add_argument("--horizon-days", type=int, default=30)
    args = parser.parse_args()
    summary = build_dataset(
        args.customers,
        args.orders,
        args.output,
        args.snapshot_date,
        args.horizon_days,
    )
    print(json.dumps(summary, indent=2))
    print(
        "WARNING: The target is inferred from purchase records and is not a "
        "subscription cancellation label."
    )


if __name__ == "__main__":
    main()
