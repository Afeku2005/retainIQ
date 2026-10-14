"""Run a local browser demo of the RetainIQ churn pipeline."""

from __future__ import annotations

import argparse
import json
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pandas as pd

from churn_pipeline import generate_demo, score, train


PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RetainIQ | Churn demo</title>
  <style>
    :root { color-scheme: light; font-family: "Segoe UI", system-ui, sans-serif; color: #172033; background: #f4f7fb; }
    * { box-sizing: border-box; }
    body { margin: 0; }
    header { background: #101b35; color: white; padding: 22px max(24px, calc((100% - 1100px) / 2)); }
    header strong { font-size: 20px; letter-spacing: .3px; }
    main { max-width: 1100px; margin: 36px auto; padding: 0 24px 48px; }
    h1 { margin: 0 0 10px; font-size: clamp(28px, 4vw, 40px); }
    p { line-height: 1.55; color: #536078; }
    .intro { margin-bottom: 24px; }
    .warning { border: 1px solid #f0ce74; border-left: 5px solid #e6a700; background: #fff9e7; border-radius: 8px; padding: 14px 16px; color: #664b00; }
    button { border: 0; border-radius: 8px; background: #315ee7; color: white; font-size: 16px; font-weight: 650; padding: 13px 18px; cursor: pointer; }
    button:hover { background: #234bc7; }
    button:disabled { opacity: .65; cursor: wait; }
    #status { min-height: 26px; margin: 14px 0; color: #536078; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(155px, 1fr)); gap: 14px; margin: 20px 0; }
    .card, section { background: white; border: 1px solid #e1e7f0; border-radius: 12px; box-shadow: 0 3px 12px #1828480b; }
    .card { padding: 17px; }
    .card span { display: block; color: #65718a; font-size: 13px; }
    .card strong { display: block; margin-top: 7px; font-size: 25px; }
    section { margin-top: 20px; padding: 20px; }
    h2 { font-size: 19px; margin: 0 0 12px; }
    ol { margin: 0; padding-left: 22px; color: #536078; line-height: 1.8; }
    .table-wrap { overflow-x: auto; }
    table { border-collapse: collapse; width: 100%; min-width: 580px; }
    th, td { padding: 11px 12px; text-align: left; border-bottom: 1px solid #e8edf4; }
    th { color: #5e6b82; font-size: 12px; text-transform: uppercase; letter-spacing: .05em; }
    td { font-variant-numeric: tabular-nums; }
    .pill { display: inline-block; border-radius: 99px; padding: 4px 9px; font-size: 12px; font-weight: 650; background: #edf1f8; }
    .pill.yes { color: #a12a27; background: #ffeded; }
    .hidden { display: none; }
    footer { color: #778197; font-size: 13px; margin-top: 28px; }
  </style>
</head>
<body>
  <header><strong>RetainIQ</strong> <span style="color:#b9c6e4"> / local churn pipeline demo</span></header>
  <main>
    <div class="intro">
      <h1>See the churn pipeline in action</h1>
      <p>Run the same generate → train → evaluate → score flow from the Python pipeline.</p>
    </div>
    <div class="warning"><strong>Demo data only.</strong> Each run creates synthetic customers.
      These predictions are not based on your customer files and must not guide customer decisions.</div>
    <p><button id="run">Run synthetic demo</button></p>
    <div id="status" role="status" aria-live="polite">Ready. Click the button to run the pipeline.</div>
    <div id="results" class="hidden">
      <div class="grid">
        <div class="card"><span>Holdout ROC-AUC</span><strong id="auc">—</strong></div>
        <div class="card"><span>Average precision</span><strong id="ap">—</strong></div>
        <div class="card"><span>Positive-class recall</span><strong id="recall">—</strong></div>
        <div class="card"><span>Positive-class precision</span><strong id="precision">—</strong></div>
        <div class="card"><span>Majority baseline accuracy</span><strong id="baseline">—</strong></div>
        <div class="card"><span>Holdout rows</span><strong id="rows">—</strong></div>
      </div>
      <section>
        <h2>What happened</h2>
        <ol>
          <li>Generated synthetic customer history.</li>
          <li>Split it into training and holdout sets; preprocessing was fit on training data only.</li>
          <li>Trained a class-balanced logistic-regression model and evaluated it on the holdout.</li>
          <li>Scored a small sample to demonstrate the output format.</li>
        </ol>
        <p id="interpretation"></p>
      </section>
      <section>
        <h2>Example scored customers</h2>
        <p>These sample rows illustrate the score output; use holdout metrics to evaluate the model.</p>
        <div class="table-wrap">
          <table>
            <thead><tr><th>Demo customer</th><th>Churn probability</th><th>Prediction at 0.5</th><th>Synthetic outcome</th></tr></thead>
            <tbody id="sample"></tbody>
          </table>
        </div>
      </section>
    </div>
    <footer>Runs on this computer at 127.0.0.1. No customer data is uploaded or transmitted.</footer>
  </main>
  <script>
    const button = document.querySelector("#run");
    const status = document.querySelector("#status");
    const results = document.querySelector("#results");
    const percent = value => `${(value * 100).toFixed(1)}%`;
    button.addEventListener("click", async () => {
      button.disabled = true;
      results.classList.add("hidden");
      status.textContent = "Running synthetic data generation, model training, evaluation, and scoring…";
      try {
        const response = await fetch("/api/run-demo", { method: "POST" });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "The demo could not complete.");
        const m = result.metrics;
        document.querySelector("#auc").textContent = m.roc_auc.toFixed(3);
        document.querySelector("#ap").textContent = m.average_precision.toFixed(3);
        document.querySelector("#recall").textContent = percent(m.recall);
        document.querySelector("#precision").textContent = percent(m.precision);
        document.querySelector("#baseline").textContent = percent(m.majority_baseline_accuracy);
        document.querySelector("#rows").textContent = m.test_rows.toLocaleString();
        document.querySelector("#interpretation").textContent =
          `The holdout positive rate is ${percent(m.test_positive_rate)}; average precision is ` +
          `${m.average_precision.toFixed(3)} versus a ${m.average_precision_baseline.toFixed(3)} ` +
          `positive-rate baseline. Treat this synthetic run as a software demonstration, not a usable model.`;
        const body = document.querySelector("#sample");
        body.replaceChildren();
        for (const customer of result.sample) {
          const row = document.createElement("tr");
          const values = [
            customer.customer_id,
            percent(customer.churn_probability),
            customer.predicted_to_cancel ? "Yes" : "No",
            customer.actual_churn ? "Canceled" : "Stayed"
          ];
          for (let i = 0; i < values.length; i++) {
            const cell = document.createElement("td");
            cell.textContent = String(values[i]);
            if (i === 2) {
              const pill = document.createElement("span");
              pill.className = `pill${customer.predicted_to_cancel ? " yes" : ""}`;
              pill.textContent = String(values[i]);
              cell.replaceChildren(pill);
            }
            row.append(cell);
          }
          body.append(row);
        }
        results.classList.remove("hidden");
        status.textContent = "Demo complete. All generated files were kept in a temporary folder and removed.";
      } catch (error) {
        status.textContent = `Error: ${error.message}`;
      } finally {
        button.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


def run_demo() -> dict[str, Any]:
    """Run the existing pipeline entirely in a temporary directory."""
    with tempfile.TemporaryDirectory(prefix="retiniq-demo-") as directory:
        root = Path(directory)
        data_path = root / "demo_customers.csv"
        model_path = root / "demo_model.joblib"
        scoring_path = root / "customers_to_score.csv"
        output_path = root / "scores.csv"

        generate_demo(data_path)
        metrics = train(data_path, "churn", model_path, id_column="customer_id")
        demo_data = pd.read_csv(data_path)
        to_score = demo_data.drop(columns="churn").head(12)
        to_score.to_csv(scoring_path, index=False)
        score(scoring_path, model_path, output_path)
        predictions = pd.read_csv(output_path)
        sample = [
            {
                "customer_id": str(customer_id),
                "churn_probability": float(probability),
                "predicted_to_cancel": bool(predicted),
                "actual_churn": bool(actual),
            }
            for customer_id, probability, predicted, actual in zip(
                to_score["customer_id"],
                predictions["churn_probability"],
                predictions["predicted_to_cancel"],
                demo_data.loc[to_score.index, "churn"],
                strict=True,
            )
        ]
        return {"metrics": metrics, "sample": sample}


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/health":
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path != "/api/run-demo":
            self._send_json(404, {"error": "Not found"})
            return
        try:
            result = run_demo()
        except (OSError, ValueError, RuntimeError) as error:
            self._send_json(500, {"error": str(error)})
            return
        self._send_json(200, result)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[dashboard] {self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the local RetainIQ synthetic churn demo dashboard."
    )
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 0 < args.port < 65536:
        parser.error("--port must be between 1 and 65535.")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), DashboardHandler)
    print(f"RetainIQ demo running at http://127.0.0.1:{args.port}")
    print("Localhost only. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping RetainIQ demo.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
