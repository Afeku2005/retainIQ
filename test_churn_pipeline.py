import tempfile
import unittest
from pathlib import Path

import pandas as pd

from churn_pipeline import generate_demo, normalize_target, score, train
from dashboard import run_demo
from prepare_order_inactivity import build_dataset
from prepare_subscription_churn import build_dataset as build_subscription_churn_dataset


class ChurnPipelineTests(unittest.TestCase):
    def test_dashboard_demo_returns_metrics_and_synthetic_scores(self) -> None:
        result = run_demo()
        self.assertIn("roc_auc", result["metrics"])
        self.assertEqual(len(result["sample"]), 12)
        self.assertTrue(
            all(0 <= row["churn_probability"] <= 1 for row in result["sample"])
        )

    def test_subscription_builder_uses_point_in_time_status_and_30_day_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            snapshots_path = root / "snapshot_dates.csv"
            subscriptions_path = root / "subscriptions.csv"
            history_path = root / "status_history.csv"
            output_path = root / "training.csv"

            pd.DataFrame(
                {
                    "snapshot_date": ["2025-01-01", "2025-02-01"],
                    "label_window_end": ["2025-01-31", "2025-03-03"],
                    "label_complete": [True, False],
                }
            ).to_csv(snapshots_path, index=False)
            pd.DataFrame(
                {
                    "customer_id": ["C1", "C2", "C3", "C4"],
                    "subscription_start_date": [
                        "2024-12-01",
                        "2024-12-01",
                        "2024-11-01",
                        "2024-10-01",
                    ],
                }
            ).to_csv(subscriptions_path, index=False)
            pd.DataFrame(
                {
                    "customer_id": [
                        "C1",
                        "C1",
                        "C2",
                        "C2",
                        "C3",
                        "C3",
                        "C4",
                        "C4",
                        "C4",
                    ],
                    "event_date": [
                        "2024-12-01",
                        "2025-01-15",
                        "2024-12-01",
                        "2025-02-10",
                        "2024-11-01",
                        "2024-12-15",
                        "2024-10-01",
                        "2024-12-20",
                        "2025-01-31",
                    ],
                    "event_type": [
                        "started",
                        "cancelled",
                        "started",
                        "cancelled",
                        "started",
                        "cancelled",
                        "started",
                        "paused",
                        "cancelled",
                    ],
                    "status_after": [
                        "active",
                        "cancelled",
                        "active",
                        "cancelled",
                        "active",
                        "cancelled",
                        "active",
                        "paused",
                        "cancelled",
                    ],
                }
            ).to_csv(history_path, index=False)

            summary = build_subscription_churn_dataset(
                snapshots_path,
                subscriptions_path,
                history_path,
                output_path,
            )
            training = pd.read_csv(output_path).set_index("customer_id")
            self.assertEqual(summary["excluded_incomplete_snapshots"], 1)
            self.assertEqual(summary["rows"], 3)
            self.assertEqual(training["churn_next_30d"].to_dict(), {
                "C1": 1,
                "C2": 0,
                "C4": 1,
            })
            self.assertEqual(training.loc["C4", "status_at_snapshot"], "paused")
            self.assertEqual(training.loc["C4", "paused_events_before"], 1)
            self.assertNotIn("current_status", training.columns)

    def test_order_builder_creates_privacy_minimized_future_proxy_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            customers_path = root / "customers.csv"
            orders_path = root / "orders.csv"
            output_path = root / "proxy.csv"
            customer_ids = [f"C{index}" for index in range(1, 11)]
            pd.DataFrame(
                {
                    "customer_id": customer_ids,
                    "name": [f"Private Name {index}" for index in range(10)],
                    "email": [f"person{index}@example.test" for index in range(10)],
                    "phone": [f"555{index:07}" for index in range(10)],
                    "signup_date": ["2025-01-01"] * 10,
                    "segment": ["A", "B"] * 5,
                    "city": ["City1", "City2"] * 5,
                    "country": ["X", "Y"] * 5,
                }
            ).to_csv(customers_path, index=False)
            order_customer_ids = [
                "C1",
                "C2",
                "C3",
                "C4",
                "C5",
                "C1",
                "C2",
                "C3",
                "C4",
                "C5",
            ]
            pd.DataFrame(
                {
                    "customer_id": order_customer_ids,
                    "date": [
                        "2026-06-10",
                        "2026-06-10",
                        "2026-06-10",
                        "2026-06-10",
                        "2026-06-10",
                        "2026-07-30",
                        "2026-07-30",
                        "2026-07-30",
                        "2026-07-30",
                        "2026-07-30",
                    ],
                    "product": ["P1"] * 5 + ["P2"] * 5,
                    "quantity": [1] * 10,
                    "total": [10] * 5 + [20] * 5,
                }
            ).to_csv(orders_path, index=False)

            summary = build_dataset(
                customers_path, orders_path, output_path, "2026-06-30", 30
            )
            proxy = pd.read_csv(output_path)
            self.assertEqual(summary["customers_with_no_order_record"], 5)
            self.assertEqual(
                proxy["no_order_record_next_30d_proxy"].tolist(), [0] * 5 + [1] * 5
            )
            self.assertEqual(proxy.loc[0, "orders_prior_30d"], 1)
            self.assertEqual(proxy.loc[0, "orders_prior_90d"], 1)
            self.assertNotIn("name", proxy.columns)
            self.assertNotIn("email", proxy.columns)
            self.assertNotIn("phone", proxy.columns)

            model_path = root / "proxy_model.joblib"
            customers_to_score = root / "customers_to_score.csv"
            scores_path = root / "scores.csv"
            train(
                output_path,
                "no_order_record_next_30d_proxy",
                model_path,
                "customer_id",
            )
            proxy.drop(columns="no_order_record_next_30d_proxy").to_csv(
                customers_to_score, index=False
            )
            score(customers_to_score, model_path, scores_path)
            scores = pd.read_csv(scores_path)
            self.assertIn(
                "no_order_record_next_30d_proxy_probability", scores.columns
            )
            self.assertNotIn("churn_probability", scores.columns)

    def test_normalize_target_maps_common_labels_and_rejects_unknown(self) -> None:
        result = normalize_target(pd.Series(["No", "YES", "retained", "cancelled"]))
        self.assertEqual(result.tolist(), [0, 1, 0, 1])
        with self.assertRaisesRegex(ValueError, "unrecognized values"):
            normalize_target(pd.Series(["maybe"]))

    def test_demo_train_and_score_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            training_data = root / "customers.csv"
            model_path = root / "model.joblib"
            scoring_data = root / "new_customers.csv"
            output_path = root / "scores.csv"

            generate_demo(training_data, rows=300, random_state=12)
            metrics = train(training_data, "churn", model_path, "customer_id")
            self.assertTrue(model_path.is_file())
            self.assertGreaterEqual(metrics["roc_auc"], 0)
            self.assertLessEqual(metrics["roc_auc"], 1)
            self.assertIn("positive_rate", metrics)
            self.assertIn("average_precision_baseline", metrics)
            self.assertIn("positive", metrics["classification_report"])

            customers = pd.read_csv(training_data).drop(columns="churn").head(15)
            customers.to_csv(scoring_data, index=False)
            self.assertEqual(score(scoring_data, model_path, output_path), 15)

            scores = pd.read_csv(output_path)
            self.assertEqual(len(scores), 15)
            self.assertIn("churn_probability", scores.columns)
            self.assertIn("predicted_to_cancel", scores.columns)
            self.assertTrue(scores["churn_probability"].between(0, 1).all())

    def test_score_reports_feature_schema_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            training_data = root / "customers.csv"
            model_path = root / "model.joblib"
            bad_data = root / "bad.csv"
            generate_demo(training_data, rows=200, random_state=5)
            train(training_data, "churn", model_path, "customer_id")
            pd.DataFrame({"customer_id": ["C1"], "tenure_months": [12]}).to_csv(
                bad_data, index=False
            )
            with self.assertRaisesRegex(ValueError, "do not match training"):
                score(bad_data, model_path, root / "scores.csv")

    def test_date_column_uses_chronological_holdout_and_is_not_a_feature(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            training_data = root / "customers.csv"
            model_path = root / "model.joblib"
            scoring_data = root / "new_customers.csv"
            output_path = root / "scores.csv"

            generate_demo(training_data, rows=300, random_state=12)
            data = pd.read_csv(training_data)
            data["snapshot_date"] = pd.date_range(
                "2025-01-01", periods=30, freq="D"
            ).repeat(10)
            data["churn"] = [index % 2 for index in range(len(data))]
            data.to_csv(training_data, index=False)

            metrics = train(
                training_data,
                "churn",
                model_path,
                "customer_id",
                date_column="snapshot_date",
            )
            self.assertEqual(metrics["validation_strategy"], "chronological")
            self.assertEqual(metrics["test_period_start"], "2025-01-25T00:00:00+00:00")
            self.assertEqual(metrics["test_rows"], 60)

            customers = data.drop(columns="churn").tail(10)
            customers.to_csv(scoring_data, index=False)
            self.assertEqual(score(scoring_data, model_path, output_path), 10)
            self.assertTrue(
                pd.read_csv(output_path)["churn_probability"].between(0, 1).all()
            )


if __name__ == "__main__":
    unittest.main()
