"""
Analysis script: Static (Model A) vs Adaptive (Model B) rain prediction comparison.

Reads predictions_log.csv with columns:
    timestamp, actual_rain, prediction_A, prediction_B

Outputs:
    - Console: metrics, confusion matrices, McNemar's test, bootstrap CI, weekly trend
    - results_summary.csv: table you can paste/adapt into your paper
    - weekly_accuracy_trend.png: plot testing the "does B improve over time" hypothesis
    - confusion_matrices.png: side-by-side confusion matrices for A and B

Usage:
    python data_analysis.py predictions_log.csv
"""

import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

N_BOOTSTRAP = 10000
RANDOM_SEED = 42
MIN_HOURS_PER_WEEK = 100


def load_data(path):
    df = pd.read_csv(path)
    required = {"timestamp", "actual_rain", "prediction_A", "prediction_B"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def basic_metrics(y_true, y_pred, label):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"\n{label}")
    print(f"Accuracy:  {acc:.2%}")
    print(f"Precision: {prec:.2%}")
    print(f"Recall:    {rec:.2%}")
    print(f"F1 score:  {f1:.3f}")
    print("Confusion matrix [no rain, rain]:")
    print(cm)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "cm": cm}


def majority_baseline(y_true):
    majority_class = 0 if y_true.mean() < 0.5 else 1
    baseline_preds = np.full(len(y_true), majority_class)
    acc = accuracy_score(y_true, baseline_preds)
    print(f"\nMajority-class baseline")
    print(f"Always predicting '{majority_class}': {acc:.2%} accuracy")
    print("Both models need to clear this bar to be considered useful.")
    return acc


def mcnemar_test(y_true, pred_A, pred_B):
    a_correct = (pred_A == y_true)
    b_correct = (pred_B == y_true)

    b = np.sum(a_correct & ~b_correct)
    c = np.sum(~a_correct & b_correct)
    n = b + c

    print(f"\nMcNemar's test")
    print(f"A correct, B wrong: {b}")
    print(f"A wrong, B correct: {c}")

    if n == 0:
        print("No disagreements to test.")
        return None

    result = stats.binomtest(c, n, p=0.5, alternative="two-sided")
    p_value = result.pvalue
    print(f"p-value: {p_value:.4f}")
    if p_value < 0.05:
        winner = "B" if c > b else "A"
        print(f"Significant difference, favoring {winner}.")
    else:
        print("Not statistically significant at the 0.05 level.")
    return p_value


def bootstrap_accuracy_diff(y_true, pred_A, pred_B, n_boot=N_BOOTSTRAP, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_true = np.asarray(y_true)
    pred_A = np.asarray(pred_A)
    pred_B = np.asarray(pred_B)

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        acc_a = np.mean(pred_A[idx] == y_true[idx])
        acc_b = np.mean(pred_B[idx] == y_true[idx])
        diffs[i] = acc_b - acc_a

    lower, upper = np.percentile(diffs, [2.5, 97.5])
    point_estimate = np.mean(pred_B == y_true) - np.mean(pred_A == y_true)

    print(f"\nBootstrap 95% CI on accuracy difference (B - A)")
    print(f"Point estimate: {point_estimate:+.2%}")
    print(f"95% CI: [{lower:+.2%}, {upper:+.2%}]")
    if lower > 0:
        print("CI is entirely above 0: B reliably outperforms A.")
    elif upper < 0:
        print("CI is entirely below 0: A reliably outperforms B.")
    else:
        print("CI straddles 0: can't rule out no real difference.")
    return point_estimate, lower, upper


def weekly_trend(df):
    start = df["timestamp"].min()
    df = df.copy()
    df["week"] = ((df["timestamp"] - start).dt.days // 7) + 1

    weekly = df.groupby("week").apply(
        lambda g: pd.Series({
            "n_hours": len(g),
            "acc_A": accuracy_score(g["actual_rain"], g["prediction_A"]),
            "acc_B": accuracy_score(g["actual_rain"], g["prediction_B"]),
        }),
        include_groups=False
    ).reset_index()
    weekly["B_minus_A"] = weekly["acc_B"] - weekly["acc_A"]

    incomplete = weekly[weekly["n_hours"] < MIN_HOURS_PER_WEEK]
    if len(incomplete) > 0:
        print(f"\nDropping {len(incomplete)} incomplete week(s) (< {MIN_HOURS_PER_WEEK} hours):")
        print(incomplete.to_string(index=False))
        weekly = weekly[weekly["n_hours"] >= MIN_HOURS_PER_WEEK].reset_index(drop=True)

    print("\nWeekly accuracy")
    print(weekly.to_string(index=False))

    if len(weekly) >= 3:
        slope, intercept, r, p, se = stats.linregress(weekly["week"], weekly["B_minus_A"])
        print(f"\nTrend of (B - A) gap over weeks: slope={slope:+.4f}/week, p={p:.4f}")
        if p < 0.05 and slope > 0:
            print("B's advantage over A is growing over time.")
        elif p < 0.05 and slope < 0:
            print("B's advantage over A is shrinking over time.")
        else:
            print("No significant trend in the B-A gap.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(weekly["week"], weekly["acc_A"], marker="o", label="Model A (static)")
    ax.plot(weekly["week"], weekly["acc_B"], marker="o", label="Model B (adaptive)")
    ax.set_xlabel("Week")
    ax.set_ylabel("Accuracy")
    ax.set_title("Weekly Prediction Accuracy: Static vs Adaptive Model")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("weekly_accuracy_trend.png", dpi=150)
    print("\nSaved weekly_accuracy_trend.png")

    return weekly


def paired_weekly_ttest(weekly):
    a = weekly["acc_A"].values
    b = weekly["acc_B"].values
    diffs = b - a

    print(f"\nPaired t-test on weekly accuracy (n={len(weekly)} weeks)")
    print(f"Mean weekly accuracy A: {a.mean():.2%}")
    print(f"Mean weekly accuracy B: {b.mean():.2%}")
    print(f"Mean weekly difference (B-A): {diffs.mean():+.2%}")

    if len(weekly) < 3:
        print("Too few weeks for a meaningful t-test.")
        return None

    t_stat, t_p = stats.ttest_rel(b, a)
    print(f"t={t_stat:.3f}, p={t_p:.4f}")
    if t_p < 0.05:
        winner = "B" if diffs.mean() > 0 else "A"
        print(f"Significant difference, favoring {winner}.")
    else:
        print("Not statistically significant at the 0.05 level.")

    return t_p


def plot_confusion_matrices(cm_a, cm_b):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, cm, title in zip(axes, [cm_a, cm_b], ["Model A (static)", "Model B (adaptive)"]):
        ax.imshow(cm, cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["No rain", "Rain"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["No rain", "Rain"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig("confusion_matrices.png", dpi=150)
    print("Saved confusion_matrices.png")


def main(path):
    df = load_data(path)
    y_true = df["actual_rain"].astype(int)
    pred_A = df["prediction_A"].astype(int)
    pred_B = df["prediction_B"].astype(int)

    print(f"Loaded {len(df)} hourly records from {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Rain hours: {y_true.sum()} ({y_true.mean():.1%} of total)")

    base_acc = majority_baseline(y_true)
    metrics_a = basic_metrics(y_true, pred_A, "Model A (Static)")
    metrics_b = basic_metrics(y_true, pred_B, "Model B (Adaptive)")

    mcnemar_p = mcnemar_test(y_true, pred_A, pred_B)
    point_est, ci_lo, ci_hi = bootstrap_accuracy_diff(y_true, pred_A, pred_B)
    weekly = weekly_trend(df)
    ttest_p = paired_weekly_ttest(weekly)
    plot_confusion_matrices(metrics_a["cm"], metrics_b["cm"])

    summary = pd.DataFrame([
        {"Model": "Majority Baseline", "Accuracy": base_acc, "Precision": None, "Recall": None, "F1": None},
        {"Model": "A (Static)", "Accuracy": metrics_a["accuracy"], "Precision": metrics_a["precision"],
         "Recall": metrics_a["recall"], "F1": metrics_a["f1"]},
        {"Model": "B (Adaptive)", "Accuracy": metrics_b["accuracy"], "Precision": metrics_b["precision"],
         "Recall": metrics_b["recall"], "F1": metrics_b["f1"]},
    ])
    summary.to_csv("results_summary.csv", index=False)
    weekly.to_csv("weekly_results.csv", index=False)

    print("\nDone.")
    print("Saved: results_summary.csv, weekly_results.csv, weekly_accuracy_trend.png, confusion_matrices.png")
    print(f"\nHeadline numbers")
    print(f"Model A accuracy: {metrics_a['accuracy']:.1%}")
    print(f"Model B accuracy: {metrics_b['accuracy']:.1%}")
    print(f"Difference (B-A): {point_est:+.1%}, 95% CI [{ci_lo:+.1%}, {ci_hi:+.1%}]")
    if mcnemar_p is not None:
        print(f"McNemar's test p-value: {mcnemar_p:.4f}")
    if ttest_p is not None:
        print(f"Paired t-test (weekly) p-value: {ttest_p:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python data_analysis.py predictions_log.csv")
        sys.exit(1)
    main(sys.argv[1])