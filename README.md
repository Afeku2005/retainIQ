# Subscription churn prediction pipeline

This is a local, end-to-end batch pipeline for estimating whether a customer is
likely to cancel. It generates example data, validates and preprocesses a CSV,
trains and evaluates a logistic-regression model, saves the fitted model, and
scores future customer rows.

## Browser demo

Launch a local dashboard that runs the synthetic generate/train/evaluate/score
flow and displays holdout metrics plus example scores:

```powershell
python dashboard.py
```

Open `http://127.0.0.1:8000` in a browser and click **Run synthetic demo**.
The dashboard binds to localhost only, uses temporary files, and deletes them
when each run finishes. It does not upload or read the provided customer CSVs.
Its synthetic predictions are for demonstration only and must not be used for
customer decisions. Press `Ctrl+C` in the terminal to stop it.

## Quick start

Python 3.10+ is recommended.

```powershell
python -m pip install -r requirements.txt
python churn_pipeline.py generate-demo
python churn_pipeline.py train --data data/demo_customers.csv
python -c "import pandas as pd; d=pd.read_csv('data/demo_customers.csv'); d.drop(columns='churn').to_csv('data/new_customers.csv', index=False)"
python churn_pipeline.py score --data data/new_customers.csv
```

By default, training writes `artifacts/churn_model.joblib`, and scoring writes
`scored_customers.csv`. The example CSV includes the `churn` label and is only
for training/demo use; scoring requires customer features but not the label.
Put private input files under `data/raw/`; generated datasets and models belong
under `data/processed/` and `artifacts/`. Those files are excluded from Git.

For a real training dataset, run:

```powershell
python churn_pipeline.py train `
  --data data/customers.csv `
  --target cancelled `
  --id-column account_id `
  --date-column snapshot_date `
  --model-out artifacts/subscription_churn.joblib
```

When historical snapshot dates are available, use `--date-column` to hold out
the latest portion of distinct dates for validation. The date is excluded from
model features, and the holdout dates may also be included or omitted when
scoring. Both the earlier training period and later holdout period must contain
examples of both churn outcomes. Without this option, training uses a stratified
random holdout.

To score new customers with that model:

```powershell
python churn_pipeline.py score `
  --data data/customers_to_score.csv `
  --model artifacts/subscription_churn.joblib `
  --output scored_customers.csv
```

Use `--id-column ""` when the CSV has no identifier column. The training target
must indicate both classes. Accepted target values include `0`/`1`, `no`/`yes`,
`false`/`true`, `active`/`churned`, and common cancellation labels. `1` always
means the customer cancels.

## Input contract

Supply one row per customer with a binary cancellation outcome in the training
CSV. Each feature column is inferred as numeric or categorical. The optional
snapshot date is only for creating a chronological evaluation split; it is
never used as a model feature. The pipeline
imputes missing numeric values with the training median, imputes categorical
values with the training mode, scales numeric features, and one-hot encodes
categories. Unseen categories at scoring time are supported. The scorer checks
that feature columns match the model's training schema and retains the optional
identifier in the output.

Use point-in-time features available before the prediction date. Exclude fields
that reveal future cancellation (for example, cancellation date or final
account status) and identifiers that might act as proxies or leakage. Prefer
the date-based holdout for forecasting, and compare against a simple baseline
before using scores to contact customers.

## Outputs and evaluation

Training prints holdout accuracy, precision, recall, F1, ROC-AUC,
average-precision, confusion matrix, and a classification report. The model
uses a 0.5 probability threshold and class-balanced logistic regression.
Scoring emits a probability and prediction column named for the configured
target (for example, `churn_probability` / `predicted_to_cancel` for the
default `churn` target), plus the ID column when configured. Tune the threshold
against intervention cost and capacity using validation data rather than
treating 0.5 as universally optimal.

The included generated data is synthetic and its scores are not evidence of
real-world model quality. A serialized `joblib` file can execute code when
loaded: only score using model files from a trusted source.

## Building a proxy dataset from sample customer and order exports

`prepare_order_inactivity.py` builds a feature table from the two provided CSV
files while excluding name, email, and phone. By default it uses features
available on 2026-06-30 and labels a customer positive if there is no order
record from July 1 through July 30, 2026:

```powershell
python prepare_order_inactivity.py
python churn_pipeline.py train `
  --data data\processed\sample-order-inactivity-proxy.csv `
  --target no_order_record_next_30d_proxy `
  --id-column customer_id `
  --model-out artifacts\sample-order-inactivity-proxy.joblib
```

The builder checks that the order export reaches the end of the label window,
and uses only pre-snapshot orders to create features. This label means
**no recorded order**, not subscription cancellation. It is only a pipeline
demonstration and must not be used to claim churn prediction performance or to
make customer decisions. The generated sample is imbalanced, and its holdout
metrics are not evidence of predictive value. Real cancellation labels and
historical snapshots are needed to train and assess a subscription-churn model.

## Build from subscription snapshots and status history

With `snapshot_dates.csv`, `subscriptions.csv`, and `status_history.csv`, build
the real 30-day cancellation examples and train on the latest snapshot dates as
the chronological holdout:

```powershell
python prepare_subscription_churn.py
python churn_pipeline.py train `
  --data data\processed\subscription-churn-training.csv `
  --target churn_next_30d `
  --id-column customer_id `
  --date-column snapshot_date `
  --model-out artifacts\subscription-churn-model.joblib
```

The builder uses only snapshots marked complete, includes subscriptions whose
status was active or paused at the snapshot, and labels a cancellation event
between the snapshot date (exclusive) and the 30-day window end (inclusive).
Features are tenure and status events recorded before each snapshot. It does
not use current plan, billing cycle, current status, or other present-day
subscription fields because these exports do not establish what those values
were historically. It excludes incomplete October 2026 labels.

The training output compares model accuracy with a majority-class baseline and
reports the chronological holdout dates. Repeated customers can occur in both
historical training and later holdout periods, which evaluates forecasting for
an existing subscription base rather than generalization to entirely unseen
customers. Check the confusion matrix, recall, precision, ROC-AUC, and baseline
before using the model. The source data's labels and history quality determine
whether results are meaningful.

### Evaluation on the supplied files

The generated dataset contains 47,427 rows for 4,567 customers; 1,846 rows
(3.9%) are positive cancellation outcomes. The chronological holdout covers
May through September 2026. On that holdout, ROC-AUC is 0.514 and
average-precision is 0.036, versus a 0.035 positive-rate baseline. At the
default threshold, precision is 0.038 and recall is 0.632. These results are
near random ranking with many false positives; **do not use this model to
target customers**. The available features are limited to tenure and prior
status history. Add trustworthy point-in-time usage, payment, support, or
engagement features before considering another evaluation.

## Tests

```powershell
python -m unittest -v
```
