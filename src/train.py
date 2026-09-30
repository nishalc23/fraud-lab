"""Reproducible fraud experiment. Run from the project root: python src/train.py."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
FEATURES = ["amount", "account_age_days", "transactions_1h", "distance_km", "new_device",
            "new_merchant", "hour", "international", "amount_to_median", "failed_attempts_24h"]


def simulate_transactions(n=60000, seed=SEED):
    """Observables are generated first; labels include hidden risk and irreducible noise.

    These are synthetic independent events, not customer histories. Velocity and ratio
    fields simulate pre-transaction aggregates; no labels or future rows create them.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    amount = np.round(np.clip(rng.lognormal(3.6, 1.25, n), 0.5, 5000), 2)
    age = np.clip(rng.exponential(240, n), 1, 2000).astype(int)
    velocity = rng.poisson(1.3, n)
    distance = np.round(np.clip(rng.exponential(30, n), 0, 1000), 1)
    new_device = rng.binomial(1, 0.08 + 0.04 * (t / n), n)
    new_merchant = rng.binomial(1, 0.22, n)
    hour = (t * 72 / n % 24).astype(int)
    international = rng.binomial(1, 0.1, n)
    ratio = np.round(np.clip(rng.lognormal(0, 0.8, n), 0.1, 20), 2)
    failed = rng.poisson(0.25, n)
    hidden_risk = rng.normal(0, 0.85, n)
    log_odds = (-7.2 + 1.7 * new_device + 0.8 * new_merchant + 0.45 * velocity
                + 0.9 * international + 0.8 * failed + 0.65 * (age < 30)
                + 0.45 * (ratio > 3) + 0.5 * ((hour < 5) & (new_device == 1))
                + 1.5 * ((velocity >= 4) & (new_device == 1))
                + 0.7 * ((amount > 500) & (international == 1)) + hidden_risk
                + 0.35 * (t / n))
    fraud = rng.binomial(1, 1 / (1 + np.exp(-log_odds)))
    df = pd.DataFrame(dict(zip(FEATURES, [amount, age, velocity, distance, new_device,
                 new_merchant, hour, international, ratio, failed])))
    df.insert(0, "Time", t * 72 * 3600 / n)
    df["Class"] = fraud
    return df


def chronological_split(df):
    """Whole equal-time groups stay on one side of each boundary."""
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    times = df.Time.to_numpy()
    bounds = [0] + [int(np.searchsorted(times, times[int(len(df) * q)], side="left"))
                    for q in (0.6, 0.7, 0.8)] + [len(df)]
    parts = [df.iloc[a:b].copy() for a, b in zip(bounds, bounds[1:])]
    if any(len(p) == 0 or p.Class.nunique() < 2 for p in parts):
        raise ValueError("Every temporal partition needs both classes; use more data.")
    return parts


def risk_logit(scores):
    p = np.clip(scores, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def choose_threshold(scores, capacity):
    """Validation quantile; ties may make observed review rate exceed the target."""
    return float(np.quantile(scores, 1 - capacity, method="higher"))


def policy_metrics(labels, scores, amounts, threshold):
    flagged = scores >= threshold
    fraud = labels == 1
    tp = int(np.sum(flagged & fraud))
    fp = int(np.sum(flagged & ~fraud))
    fn = int(np.sum(~flagged & fraud))
    tn = int(np.sum(~flagged & ~fraud))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "reviewed": tp + fp,
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "review_rate": float(flagged.mean()),
            "flagged_fraud_amount": float(amounts[flagged & fraud].sum()),
            "missed_fraud_amount": float(amounts[~flagged & fraud].sum())}


def export_estimator(model, calibration, features, median):
    result = {"features": features, "median": median.tolist(),
              "calibration": {"coefficient": float(calibration.coef_[0, 0]),
                              "intercept": float(calibration.intercept_[0])}}
    if hasattr(model, "named_steps"):
        scaler = model.named_steps["standardscaler"]
        lr = model.named_steps["logisticregression"]
        result.update(kind="logistic", mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                      coefficients=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0]))
    else:
        trees = []
        for estimator in model.estimators_[:, 0]:
            tree = estimator.tree_
            trees.append({"left": tree.children_left.tolist(), "right": tree.children_right.tolist(),
                          "feature": tree.feature.tolist(), "threshold": tree.threshold.tolist(),
                          "value": tree.value[:, 0, 0].tolist()})
        result.update(kind="boosting", learning_rate=model.learning_rate,
                      intercept=float(model._raw_predict_init(np.zeros((1, len(features))))[0, 0]),
                      trees=trees)
    return result


def run(source="synthetic", csv_path=None, rows=60000):
    started = perf_counter()
    if source == "synthetic":
        df = simulate_transactions(rows)
        features = FEATURES
        amount_col = "amount"
    else:
        if not csv_path:
            raise ValueError("--csv is required for the ULB benchmark")
        df = pd.read_csv(csv_path)
        required = {"Time", "Amount", "Class", *[f"V{i}" for i in range(1, 29)]}
        if not required.issubset(df.columns):
            raise ValueError("Expected the original ULB creditcard.csv schema")
        features = [f"V{i}" for i in range(1, 29)] + ["LogAmount"]
        df["LogAmount"] = np.log1p(df.Amount)
        amount_col = "Amount"
    if df[features + ["Time", "Class", amount_col]].isna().any().any():
        raise ValueError("Missing data needs an explicit handling strategy")
    if not set(df.Class.unique()).issubset({0, 1}):
        raise ValueError("Class must be binary")
    raw_rows = len(df)
    # Original PCA features were produced upstream: their fitting protocol is unknown.
    # Remove exact duplicate observables + label before chronological partitioning.
    df = df.drop_duplicates(subset=["Time", *features, "Class"]).copy()
    train, calibration_data, validation, test = chronological_split(df)
    candidates = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(
            class_weight="balanced", max_iter=1000, C=1, random_state=SEED)),
        "boosting": GradientBoostingClassifier(n_estimators=70, max_depth=3,
            learning_rate=0.07, min_samples_leaf=80, subsample=0.8, random_state=SEED),
    }
    models = {}
    median = train[features].median().to_numpy()
    print(f"Source: {source}; partitions: {[len(p) for p in (train, calibration_data, validation, test)]}", flush=True)
    for name, estimator in candidates.items():
        fit_start = perf_counter()
        if name == "boosting":
            estimator.fit(train[features].to_numpy(), train.Class,
                          sample_weight=compute_sample_weight("balanced", train.Class))
        else:
            estimator.fit(train[features].to_numpy(), train.Class)
        # The calibrator never sees the training or validation labels.
        calibrator = LogisticRegression(C=100, max_iter=1000, random_state=SEED)
        calibrator.fit(risk_logit(estimator.predict_proba(calibration_data[features].to_numpy())[:, 1]),
                       calibration_data.Class)
        def score(part):
            return calibrator.predict_proba(risk_logit(estimator.predict_proba(part[features].to_numpy())[:, 1]))[:, 1]
        val_scores = score(validation)
        threshold = choose_threshold(val_scores, 0.01)
        validation_policy = policy_metrics(validation.Class.to_numpy(), val_scores,
                                           validation[amount_col].to_numpy(), threshold)
        models[name] = {"estimator": estimator, "calibrator": calibrator,
                       "validation_scores": val_scores, "threshold": threshold,
                       "validation_policy": validation_policy,
                       "validation_ap": float(average_precision_score(validation.Class, val_scores)),
                       "training_seconds": round(perf_counter() - fit_start, 3)}
        print(name, "validation AP:", models[name]["validation_ap"], flush=True)
    # Lock selection before computing any held-out test metrics.
    selected = max(models, key=lambda name: (models[name]["validation_policy"]["recall"], models[name]["validation_ap"]))
    report = {"source": source, "seed": SEED, "rows": len(df), "raw_rows": raw_rows,
              "duplicates_removed": raw_rows - len(df), "fraud_count": int(df.Class.sum()),
              "fraud_rate": float(df.Class.mean()), "features": features, "selected_model": selected,
              "selection_rule": "Highest validation recall at a validation-derived 1% review target; AP breaks ties.",
              "split": {name: {"rows": len(part), "fraud": int(part.Class.sum()),
                         "time_start": float(part.Time.min()), "time_end": float(part.Time.max())}
                        for name, part in zip(("train", "calibration", "validation", "test"), (train, calibration_data, validation, test))},
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
              "data_sha256": hashlib.sha256(df.to_csv(index=False).encode()).hexdigest(), "models": {}}
    exported = {}
    test_scores = {}
    for name, item in models.items():
        scores = item["calibrator"].predict_proba(risk_logit(item["estimator"].predict_proba(test[features].to_numpy())[:, 1]))[:, 1]
        test_scores[name] = scores
        policy = policy_metrics(test.Class.to_numpy(), scores, test[amount_col].to_numpy(), item["threshold"])
        rng = np.random.default_rng(SEED)
        fraud_flags = (scores[test.Class.to_numpy() == 1] >= item["threshold"]).astype(float)
        boot = rng.choice(fraud_flags, (1000, len(fraud_flags)), replace=True).mean(axis=1)
        report["models"][name] = {"validation_ap": item["validation_ap"],
            "validation_policy": item["validation_policy"], "test_ap": float(average_precision_score(test.Class, scores)),
            "test_roc_auc": float(roc_auc_score(test.Class, scores)), "test_brier": float(brier_score_loss(test.Class, scores)),
            "threshold": item["threshold"], "test_policy": policy,
            "test_recall_interval": np.quantile(boot, [0.025, 0.975]).tolist(),
            "training_seconds": item["training_seconds"]}
        exported[name] = export_estimator(item["estimator"], item["calibrator"], features, median)
    report["total_seconds"] = round(perf_counter() - started, 3)
    test_rows = [{"id": f"TX-{i + 1:05}", "time": round(float(row.Time), 1),
                  "amount": round(float(row[amount_col]), 2), "label": int(row.Class),
                  "features": [round(float(row[f]), 6) for f in features],
                  "scores": {name: float(test_scores[name][i]) for name in models}}
                 for i, (_, row) in enumerate(test.iterrows())]
    payload = {"report": report, "models": exported, "test": test_rows,
               "validation_scores": {name: item["validation_scores"].tolist() for name, item in models.items()}}
    for directory in (ROOT / "reports", ROOT / "dist"):
        directory.mkdir(exist_ok=True)
    (ROOT / "reports/metrics.json").write_text(json.dumps(report, indent=2))
    (ROOT / "dist/experiment.json").write_text(json.dumps(payload, separators=(",", ":")))
    # Full precision vectors test the exported browser scorer against sklearn.
    parity_x = np.vstack([test[features].iloc[:15].to_numpy(), median.reshape(1, -1)])
    parity = {"features": parity_x.tolist(), "expected": {
        name: item["calibrator"].predict_proba(risk_logit(item["estimator"].predict_proba(parity_x)[:, 1]))[:, 1].tolist()
        for name, item in models.items()}}
    (ROOT / "reports/parity.json").write_text(json.dumps(parity))
    print(json.dumps({"selected_model": selected, "test_results": report["models"], "seconds": report["total_seconds"]}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["synthetic", "ulb"], default="synthetic")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--rows", type=int, default=60000)
    args = parser.parse_args()
    run(args.source, args.csv, args.rows)
