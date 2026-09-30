# Fraud Lab

A reproducible transaction-risk experiment and interactive review-policy demo.

**Current results use simulated data.** No Ramp data, customer data, or real-world
performance claims are involved. The browser executes trained, calibrated models;
the charts and tables are computed from held-out scored transactions.

## The question

With limited review capacity, how much fraud can a team flag, and how many legitimate
transactions would it flag along the way? This project separates model ranking,
probability calibration, operational threshold selection, and hypothetical economics.

## Run locally

Python 3.9+ and Node 18+ are sufficient. Use a virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/train.py
node tests/check_export.cjs
node tests/check_ui.cjs
python -m http.server 8000 --directory dist
```

Open http://localhost:8000. The committed `dist/experiment.json` already contains
the trained models and test predictions, so serving the demo needs only Python.
Training regenerates metrics, models and predictions; it does not rewrite this
snapshot write-up automatically. Consult `reports/metrics.json` after any rerun.

## Data and features

The seeded generator creates 60,000 independent simulated transaction events over
72 hours. Observable inputs include amount, account age, recent velocity, distance,
new device, new merchant, hour, international status, amount relative to a typical
transaction, and failed attempts. These fields simulate information available before
an authorization. They are not aggregates reconstructed from real customer histories.

Fraud outcomes are sampled from a nonlinear risk process with unobserved variation.
New-device frequency and underlying risk drift over time. Labels are never inputs.
The generator is an explicit assumption, not evidence that these features cause fraud.
It favors some learnable relationships and is not a realistic substitute for real data.

## Experiment protocol

1. Audit schema, missingness, labels and exact duplicates.
2. Sort by elapsed time, keeping equal timestamps in the same partition.
3. Use the first 60% for fitting, the next 10% for calibration, the next 10% for
   validation, and the last 20% for one final test evaluation.
4. Fit standardized, balanced logistic regression and balanced gradient boosting
   (70 trees, depth 3). Preprocessing is fit on training only.
5. Fit a separate, unweighted sigmoid calibrator on raw score logits in the calibration
   partition. Calibration needs checking; it does not guarantee correct probabilities.
6. For each candidate, derive a cutoff from the 99th percentile of validation scores.
   Select the candidate with highest validation fraud recall; average precision breaks ties.
7. Lock the selected model and cutoff before scoring the held-out test partition.

Hyperparameters are fixed starting points, not exhaustively tuned. Both candidates
are reported on test for transparency, but test results do not drive selection.
Exploring the test curves afterward is explicitly exploratory and cannot serve as
unbiased evidence for a newly tuned policy. A further time period would be required.

## Measured simulation result — seed 42

| Metric | Logistic regression | Gradient boosting |
| --- | ---: | ---: |
| Validation average precision | 0.0990 | 0.0885 |
| Test average precision | 0.0917 | 0.0874 |
| Test ROC AUC | 0.7912 | 0.7600 |
| Test Brier score | 0.01139 | 0.01148 |
| Fraud flagged at original cutoff | 22 / 144 | 17 / 144 |
| Fraud recall | 15.3% | 11.8% |
| Review precision | 23.4% | 16.3% |
| Actual test review rate | 0.783% | 0.867% |
| Approx. 95% bootstrap recall interval | 9.7–21.5% | 6.3–17.4% |

Logistic regression was selected on validation: both models flagged 9 of 75 validation
fraud cases at a 1% target, and logistic had higher validation average precision.
The test period has 12,000 transactions, including 144 fraud cases (1.2%).
Its review precision is about 19.5 times that prevalence. This is enrichment over
random review, **not** a claim that 19.5 times as much fraud would be prevented.

The bootstrap intervals resample test fraud cases at a fixed fitted model and cutoff.
They are conditional, assume independent cases, and do not capture retraining,
temporal dependence, calibration uncertainty, or full pipeline variability.
The overlapping intervals do not establish statistically significant superiority.

## Demo

- **Policy lab:** adjust a validation review target; apply that cutoff to held-out
  test scores; inspect recall, precision, review volume, confusion counts and queue.
- **Cost sensitivity:** change review cost, additional false-alert cost and the
  fraction of flagged fraud amount assumed prevented. Net savings is a scenario:
  `flagged fraud amount × prevention fraction − review cost × alerts − friction cost × false alerts`.
- **Transaction explorer:** run real browser-side inference on editable feature
  values. Replace one feature at a time with its training median to inspect score
  sensitivity. This is not causal attribution or SHAP, and arbitrary combinations
  may be outside the training distribution.
- **Methodology:** inspect partitions, model comparison, data audit and limitations.

Balanced training scores are recalibrated, but reliability and shift still matter.
The deployed static model has no API key, user accounts, payments integration,
uploads, backend database, or external inference dependency.

## Validation

`tests/check_export.cjs` compares exported browser inference with full-precision
Python outputs for both models at 16 input vectors. It checks the original cutoff,
confusion counts, cost arithmetic, invalid input handling and time boundaries.
Observed maximum score discrepancy: less than 1e-15. Syntax checks also passed.
`tests/check_ui.cjs` exercises the app with a minimal DOM fixture; this is a functional
smoke test, not browser rendering or accessibility certification.

## Rerun on a real public benchmark

The original ULB/Worldline dataset is referenced by TensorFlow's
[imbalanced-data tutorial](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data).
The tutorial documents a downloadable CSV mirror. Review the
[original dataset and terms](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
before redistribution.

```sh
python src/download_data.py
python src/train.py --source ulb --csv data/creditcard.csv
node tests/check_export.cjs
```

The pipeline uses V1–V28 and log-transformed Amount, excluding Time from the model.
Time is used for partitioning. The demo adapts to the model's feature schema and
reports the ULB source label. Do not interpret anonymous PCA components as device,
location, or merchant signals. Upstream PCA fitting provenance is unknown, so this
cannot establish a fully point-in-time raw-data feature pipeline. The original data
cover a short historical interval and have no usable customer identifiers; account-level
leakage, repeated entities, delayed labels and long-term drift cannot be fully audited.

The original CSV is not committed. The initial environment blocked its download, so
**the ULB path has not yet been run or validated on the real dataset**. Do not claim
these simulation results came from it. After a real-data run, replace the simulation
result table with actual new results and rerun all checks before republishing.

## What to do next

Train on real transaction data, compare an amount-aware decision objective, add rolling
time windows and calibration plots, evaluate subgroup reliability, then shadow-score
live traffic. Only after operational validation should a randomized intervention test
estimate actual fraud loss reduction and customer friction. This project has no A/B test.

## Files

`src/train.py` owns data generation, training, calibration, selection, evaluation and
model export. `reports/metrics.json` is the machine-readable result.
`notebooks/analysis.ipynb` is an explanatory notebook. `dist/` is a standalone static
website. `INTERVIEW.md` provides discussion notes; learn the implementation before
presenting it as work you can explain.
