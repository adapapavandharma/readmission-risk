"""
Train and cross-validate 30-day readmission models.

Contains the experiment that motivates the whole project: the same model,
scored two ways, once with a naive random split and once with a split grouped
on patient. The gap between them is the amount of performance a careless
evaluation invents out of nothing.

    python src/model.py                # leakage experiment + train + hold-out
    python src/model.py --skip-leakage # just train
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "interim" / "encounters.parquet"
INTERIM = ROOT / "data" / "interim"
OUT = ROOT / "outputs"

SEED = 20260803
N_SPLITS = 5
TARGET = "readmitted_30d"
GROUP = "patient_nbr"

NUMERIC = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    "age_midpoint", "prior_visits_total", "n_drugs_changed", "n_drugs_active",
    "had_prior_inpatient", "had_prior_emergency", "a1c_measured", "glucose_measured",
]

CATEGORICAL = [
    "race", "gender", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "medical_specialty", "payer_code",
    "A1Cresult", "max_glu_serum", "change", "diabetesMed",
    "diag_1_group", "diag_2_group", "diag_3_group",
    "metformin", "insulin", "glipizide", "glyburide", "pioglitazone",
    "rosiglitazone", "glimepiride", "repaglinide", "nateglinide",
]

# medical_specialty alone has ~70 levels with a long tail of single-digit
# counts. Rare levels become "Rare" so the encoder does not manufacture
# hundreds of columns that each fire on a handful of rows.
RARE_THRESHOLD = 100


def load() -> pd.DataFrame:
    df = pd.read_parquet(DATA)
    for c in CATEGORICAL:
        df[c] = df[c].astype(str)
        counts = df[c].value_counts()
        rare = counts[counts < RARE_THRESHOLD].index
        df[c] = df[c].where(~df[c].isin(rare), "Rare")
    return df


def make_pipeline(kind: str) -> Pipeline:
    if kind == "logistic":
        pre = ColumnTransformer([
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), CATEGORICAL),
        ])
        # class_weight balanced: at an 11% base rate an unweighted fit puts
        # almost all its effort on the majority class.
        clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced",
                                 solver="lbfgs", random_state=SEED)
    elif kind == "gbm":
        pre = ColumnTransformer([
            ("num", "passthrough", NUMERIC),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1), CATEGORICAL),
        ])
        clf = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=60, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.12,
            categorical_features=list(range(len(NUMERIC), len(NUMERIC) + len(CATEGORICAL))),
            random_state=SEED,
        )
    else:
        raise ValueError(kind)
    return Pipeline([("pre", pre), ("clf", clf)])


def cross_val_auc(df: pd.DataFrame, kind: str, grouped: bool) -> tuple[float, float, list[float]]:
    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET].to_numpy()
    groups = df[GROUP].to_numpy()

    splitter = (StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
                if grouped else
                StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED))
    split_args = (X, y, groups) if grouped else (X, y)

    scores = []
    for train_idx, test_idx in splitter.split(*split_args):
        pipe = make_pipeline(kind)
        pipe.fit(X.iloc[train_idx], y[train_idx])
        p = pipe.predict_proba(X.iloc[test_idx])[:, 1]
        scores.append(roc_auc_score(y[test_idx], p))
    return float(np.mean(scores)), float(np.std(scores)), scores


def leakage_experiment(df: pd.DataFrame) -> dict:
    print(f"\n{'='*72}\nLEAKAGE EXPERIMENT — does the split method change the answer?\n{'='*72}")
    print(f"  {df[GROUP].nunique():,} patients across {len(df):,} encounters; "
          f"{(df[GROUP].value_counts() > 1).sum():,} patients appear more than once\n")

    results = {}
    for kind in ("logistic", "gbm"):
        naive_m, naive_s, _ = cross_val_auc(df, kind, grouped=False)
        group_m, group_s, _ = cross_val_auc(df, kind, grouped=True)
        inflation = naive_m - group_m
        results[kind] = {
            "auc_random_split": round(naive_m, 4), "auc_random_split_sd": round(naive_s, 4),
            "auc_patient_grouped": round(group_m, 4), "auc_patient_grouped_sd": round(group_s, 4),
            "inflation": round(inflation, 4),
        }
        print(f"  {kind:<9}  random split {naive_m:.4f} ±{naive_s:.4f}   "
              f"patient-grouped {group_m:.4f} ±{group_s:.4f}   "
              f"inflation {inflation:+.4f}")
    return results


def fit_final(df: pd.DataFrame) -> dict:
    """Fit on a patient-grouped train split and score a held-out set once."""
    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET].to_numpy()
    groups = df[GROUP].to_numpy()

    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    overlap = set(groups[train_idx]) & set(groups[test_idx])
    assert not overlap, f"patient leaked across the split: {len(overlap)} patients"
    print(f"\n{'='*72}\nHOLD-OUT EVALUATION\n{'='*72}")
    print(f"  train {len(train_idx):,} encounters / {len(set(groups[train_idx])):,} patients")
    print(f"  test  {len(test_idx):,} encounters / {len(set(groups[test_idx])):,} patients")
    print(f"  patient overlap between train and test: {len(overlap)}")

    preds = {}
    for kind in ("logistic", "gbm"):
        pipe = make_pipeline(kind)
        pipe.fit(X.iloc[train_idx], y[train_idx])
        p = pipe.predict_proba(X.iloc[test_idx])[:, 1]
        preds[kind] = p
        print(f"  {kind:<9}  hold-out AUC {roc_auc_score(y[test_idx], p):.4f}")

    # Carry the two crude "rules a nurse could apply from the chart" through to
    # evaluation. If a gradient-boosted model over 23 categorical features
    # cannot beat counting prior admissions, that is the finding, and it should
    # be impossible to avoid noticing it.
    held = pd.DataFrame({
        "patient_nbr": groups[test_idx],
        "y_true": y[test_idx],
        "p_logistic": preds["logistic"],
        "p_gbm": preds["gbm"],
        "rule_prior_inpatient": df["number_inpatient"].to_numpy()[test_idx],
        "rule_prior_visits": df["prior_visits_total"].to_numpy()[test_idx],
    })
    for rule in ("rule_prior_inpatient", "rule_prior_visits"):
        print(f"  {rule:<21}  hold-out AUC "
              f"{roc_auc_score(held['y_true'], held[rule]):.4f}   (single variable)")
    INTERIM.mkdir(parents=True, exist_ok=True)
    held.to_parquet(INTERIM / "holdout_predictions.parquet", index=False)
    print(f"\n  wrote {(INTERIM / 'holdout_predictions.parquet').relative_to(ROOT)}")

    return {kind: round(float(roc_auc_score(y[test_idx], p)), 4)
            for kind, p in preds.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-leakage", action="store_true")
    args = ap.parse_args()

    if not DATA.exists():
        print(f"missing {DATA}\nrun: python src/build_dataset.py", file=sys.stderr)
        return 1

    df = load()
    report: dict = {
        "encounters": int(len(df)),
        "patients": int(df[GROUP].nunique()),
        "base_rate": round(float(df[TARGET].mean()), 4),
        "seed": SEED,
        "cv_folds": N_SPLITS,
    }
    if not args.skip_leakage:
        report["leakage_experiment"] = leakage_experiment(df)
    report["holdout_auc"] = fit_final(df)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_report.json").write_text(json.dumps(report, indent=2))
    print(f"  wrote {(OUT / 'model_report.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
