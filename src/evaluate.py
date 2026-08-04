"""
Operational evaluation of the readmission models.

AUC is the wrong number to make a staffing decision with. A care-management
team has capacity to call some fixed number of patients a day, so the questions
that matter are:

    if we work the top K% of the list, how many readmissions do we catch?
    how many people do we call to prevent one?
    are the predicted probabilities trustworthy enough to set a threshold?
    is intervening better than the two trivial policies - everyone, nobody?

Those are lift, number-needed-to-screen, calibration, and net benefit. All four
are reported here; AUC is a footnote.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402
import pandas as pd                    # noqa: E402
from sklearn.calibration import calibration_curve   # noqa: E402
from sklearn.isotonic import IsotonicRegression     # noqa: E402
from sklearn.metrics import (           # noqa: E402
    average_precision_score, brier_score_loss, roc_auc_score, roc_curve,
    precision_recall_curve,
)

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "data" / "interim" / "holdout_predictions.parquet"
OUT = ROOT / "outputs"
FIGS = OUT / "figures"

INK, ACCENT, WARN, MUTED = "#12201c", "#0e7490", "#b45309", "#6b7c76"
GREY = "#94a3b8"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "600",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c6d4ce", "grid.color": "#e6ede9",
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": "white",
})

TOP_K = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
MODELS = {
    "p_gbm": "Gradient boosting",
    "p_logistic": "Logistic regression",
    "rule_prior_inpatient": "Rule: prior inpatient count",
}


def lift_table(y: np.ndarray, score: np.ndarray, base: float) -> pd.DataFrame:
    """Recall, precision, lift and number-needed-to-screen at each list depth."""
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order]
    n, total_pos = len(y), int(y.sum())

    rows = []
    for k in TOP_K:
        cut = max(int(round(k * n)), 1)
        caught = int(y_sorted[:cut].sum())
        precision = caught / cut
        rows.append({
            "top_k_pct": round(k * 100, 1),
            "patients_flagged": cut,
            "readmissions_caught": caught,
            "recall_pct": round(100 * caught / total_pos, 1),
            "precision_pct": round(100 * precision, 1),
            "lift": round(precision / base, 2),
            "number_needed_to_screen": round(1 / precision, 1) if precision else None,
        })
    return pd.DataFrame(rows)


def calibration_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Brier score plus the calibration intercept and slope.

    A perfectly calibrated model has intercept 0 and slope 1 on the logit
    scale. Slope < 1 means the predictions are too extreme; a large negative
    intercept means they are systematically too high.
    """
    eps = 1e-6
    p_c = np.clip(p, eps, 1 - eps)
    logit = np.log(p_c / (1 - p_c))

    from sklearn.linear_model import LogisticRegression
    slope_fit = LogisticRegression(max_iter=1000).fit(logit.reshape(-1, 1), y)
    slope = float(slope_fit.coef_[0][0])
    intercept_fit = LogisticRegression(max_iter=1000, fit_intercept=True)
    intercept_fit.fit(np.zeros((len(y), 1)), y)   # unused; kept explicit below

    # intercept with slope fixed at 1 (offset model), estimated by a 1-D search
    from scipy.optimize import minimize_scalar
    def nll(a: float) -> float:
        q = 1 / (1 + np.exp(-(a + logit)))
        q = np.clip(q, eps, 1 - eps)
        return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).sum())
    intercept = float(minimize_scalar(nll, bounds=(-5, 5), method="bounded").x)

    return {
        "brier": round(float(brier_score_loss(y, p)), 5),
        "calibration_slope": round(slope, 3),
        "calibration_intercept": round(intercept, 3),
        "mean_predicted": round(float(p.mean()), 4),
        "observed_rate": round(float(y.mean()), 4),
    }


def net_benefit(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Decision-curve net benefit.

        NB = TP/n - FP/n * (pt / (1 - pt))

    The threshold probability pt encodes how many false positives a team will
    tolerate to catch one true positive. A model is only worth deploying where
    its curve sits above both "treat everyone" and "treat nobody".
    """
    n = len(y)
    out = []
    for pt in thresholds:
        flag = p >= pt
        tp = int((flag & (y == 1)).sum())
        fp = int((flag & (y == 0)).sum())
        out.append(tp / n - (fp / n) * (pt / (1 - pt)))
    return np.array(out)


def main() -> int:
    if not PRED.exists():
        print(f"missing {PRED}\nrun: python src/model.py", file=sys.stderr)
        return 1

    df = pd.read_parquet(PRED)
    y = df["y_true"].to_numpy()
    base = float(y.mean())
    FIGS.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "holdout_encounters": int(len(df)),
        "holdout_patients": int(df["patient_nbr"].nunique()),
        "base_rate": round(base, 4),
        "models": {},
    }

    print(f"\n{'='*74}\nHOLD-OUT: {len(df):,} encounters, {df['patient_nbr'].nunique():,} "
          f"patients, base rate {base:.2%}\n{'='*74}")

    for col, label in MODELS.items():
        s = df[col].to_numpy(float)
        auc = roc_auc_score(y, s)
        ap = average_precision_score(y, s)
        entry = {
            "label": label,
            "roc_auc": round(float(auc), 4),
            "pr_auc": round(float(ap), 4),
            "pr_auc_over_base": round(float(ap) / base, 2),
        }
        if col.startswith("p_"):
            entry["calibration"] = calibration_metrics(y, s)
        lt = lift_table(y, s, base)
        entry["lift_table"] = lt.to_dict(orient="records")
        report["models"][col] = entry

        print(f"\n  {label}")
        print(f"    ROC AUC {auc:.4f}   PR AUC {ap:.4f} "
              f"({ap/base:.2f}x the {base:.1%} base rate)")
        if "calibration" in entry:
            c = entry["calibration"]
            print(f"    Brier {c['brier']:.5f}   calibration slope {c['calibration_slope']:.2f} "
                  f"intercept {c['calibration_intercept']:+.2f}   "
                  f"mean predicted {c['mean_predicted']:.3f} vs observed {c['observed_rate']:.3f}")
        row20 = lt[lt.top_k_pct == 20.0].iloc[0]
        print(f"    top 20% of the list -> catches {row20.recall_pct:.1f}% of readmissions, "
              f"lift {row20.lift:.2f}x, NNS {row20.number_needed_to_screen:.1f}")

    # ---- recalibrate the best model -------------------------------------
    # class_weight='balanced' and boosting both distort probabilities. A risk
    # board needs numbers a clinician can act on, so isotonic recalibration is
    # fitted on half the hold-out and evaluated on the other half.
    rng = np.random.default_rng(20260803)
    patients = df["patient_nbr"].unique()
    rng.shuffle(patients)
    cal_ids = set(patients[: len(patients) // 2])
    in_cal = df["patient_nbr"].isin(cal_ids).to_numpy()

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(df.loc[in_cal, "p_gbm"], y[in_cal])
    p_cal = iso.predict(df.loc[~in_cal, "p_gbm"])
    y_eval = y[~in_cal]

    before = calibration_metrics(y_eval, df.loc[~in_cal, "p_gbm"].to_numpy())
    after = calibration_metrics(y_eval, p_cal)
    report["recalibration"] = {
        "method": "isotonic, fitted on half the hold-out patients",
        "before": before, "after": after,
        "auc_before": round(float(roc_auc_score(y_eval, df.loc[~in_cal, "p_gbm"])), 4),
        "auc_after": round(float(roc_auc_score(y_eval, p_cal)), 4),
    }
    print(f"\n{'='*74}\nRECALIBRATION (isotonic)\n{'='*74}")
    print(f"  Brier      {before['brier']:.5f} -> {after['brier']:.5f}")
    print(f"  slope      {before['calibration_slope']:.2f} -> {after['calibration_slope']:.2f}")
    print(f"  intercept  {before['calibration_intercept']:+.2f} -> {after['calibration_intercept']:+.2f}")
    print(f"  AUC        {report['recalibration']['auc_before']:.4f} -> "
          f"{report['recalibration']['auc_after']:.4f}  (ranking is unchanged, as expected)")

    # ---- figures ---------------------------------------------------------
    _fig_lift(df, y, base)
    _fig_calibration(y, df["p_gbm"].to_numpy(), y_eval, p_cal)
    _fig_roc_pr(df, y, base)
    _fig_decision_curve(df, y, base)

    lift_table(y, df["p_gbm"].to_numpy(), base).to_csv(OUT / "lift_table.csv", index=False)
    (OUT / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(f"\n  wrote {(OUT / 'evaluation.json').relative_to(ROOT)} and "
          f"{len(list(FIGS.glob('*.png')))} figures")
    return 0


def _fig_lift(df: pd.DataFrame, y: np.ndarray, base: float) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    depth = np.linspace(0.01, 1.0, 120)
    for col, label, colour in [("p_gbm", "Gradient boosting", ACCENT),
                               ("p_logistic", "Logistic regression", "#7dd3fc"),
                               ("rule_prior_inpatient", "Rule: prior inpatient count", WARN)]:
        s = df[col].to_numpy(float)
        order = np.argsort(-s, kind="mergesort")
        ys = y[order]
        recall = [ys[:max(int(d * len(y)), 1)].sum() / y.sum() for d in depth]
        ax.plot(depth * 100, np.array(recall) * 100, lw=2, color=colour, label=label)
    ax.plot([0, 100], [0, 100], ls=":", color=GREY, lw=1.4, label="Calling at random")
    ax.axvline(20, color=INK, lw=1, ls="--", alpha=.5)
    ax.set_xlabel("share of patients contacted, highest risk first (%)")
    ax.set_ylabel("readmissions caught (%)")
    ax.set_title("What the model buys a care-management team", loc="left")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=.7)
    fig.savefig(FIGS / "lift_curve.png")
    plt.close(fig)


def _fig_calibration(y, p_raw, y_eval, p_cal) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for yy, pp, label, colour in [(y, p_raw, "Raw model", WARN),
                                  (y_eval, p_cal, "After isotonic recalibration", ACCENT)]:
        frac, mean_pred = calibration_curve(yy, pp, n_bins=10, strategy="quantile")
        ax.plot(mean_pred * 100, frac * 100, "o-", color=colour, lw=1.8, ms=4.5, label=label)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], ls=":", color=GREY, lw=1.4, label="Perfect calibration")
    ax.set_xlabel("predicted risk (%)")
    ax.set_ylabel("observed readmission rate (%)")
    ax.set_title("Can a clinician trust the number?", loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=.7)
    fig.savefig(FIGS / "calibration.png")
    plt.close(fig)


def _fig_roc_pr(df, y, base) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for col, label, colour in [("p_gbm", "Gradient boosting", ACCENT),
                               ("p_logistic", "Logistic", "#7dd3fc"),
                               ("rule_prior_inpatient", "Prior-inpatient rule", WARN)]:
        s = df[col].to_numpy(float)
        fpr, tpr, _ = roc_curve(y, s)
        a1.plot(fpr, tpr, lw=1.9, color=colour,
                label=f"{label} ({roc_auc_score(y, s):.3f})")
        pr, rc, _ = precision_recall_curve(y, s)
        a2.plot(rc, pr, lw=1.9, color=colour,
                label=f"{label} ({average_precision_score(y, s):.3f})")
    a1.plot([0, 1], [0, 1], ls=":", color=GREY, lw=1.3)
    a1.set_xlabel("false positive rate"); a1.set_ylabel("true positive rate")
    a1.set_title("ROC", loc="left"); a1.legend(frameon=False, fontsize=8); a1.grid(alpha=.7)

    a2.axhline(base, ls=":", color=GREY, lw=1.3, label=f"base rate ({base:.1%})")
    a2.set_xlabel("recall"); a2.set_ylabel("precision")
    a2.set_title("Precision–recall — the honest view at an 11% base rate", loc="left")
    a2.legend(frameon=False, fontsize=8); a2.grid(alpha=.7)
    fig.savefig(FIGS / "roc_pr.png")
    plt.close(fig)


def _fig_decision_curve(df, y, base) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    th = np.linspace(0.02, 0.45, 90)
    ax.plot(th * 100, net_benefit(y, df["p_gbm"].to_numpy(float), th),
            lw=2, color=ACCENT, label="Gradient boosting")
    ax.plot(th * 100, net_benefit(y, np.ones(len(y)), th), lw=1.5,
            color=GREY, ls="--", label="Contact everyone")
    ax.axhline(0, color=INK, lw=1.2, label="Contact nobody")
    ax.set_xlabel("threshold probability (%) — how many false alarms you will accept")
    ax.set_ylabel("net benefit")
    ax.set_title("Decision curve: the band where the model is worth using", loc="left")
    ax.legend(frameon=False)
    ax.grid(alpha=.7)
    ax.set_ylim(-0.02, None)
    fig.savefig(FIGS / "decision_curve.png")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
