"""Tests for src/model.py -- feature wiring and the leakage experiment.

The two things that can go wrong here silently are (a) the gradient booster's
categorical_features indices drifting out of step with the ColumnTransformer's
column order, which would tell the model that a numeric column is categorical
and vice versa, and (b) the grouped split quietly degrading into a random one.
Neither raises; both are checked below.

All fits are on small synthetic frames with fixed seeds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

import model as md


# --------------------------------------------------------------------------
# Feature lists
# --------------------------------------------------------------------------

def test_the_two_feature_lists_are_disjoint_and_free_of_duplicates():
    """A column in both lists would be fed to the model twice, once scaled and
    once encoded."""
    assert len(set(md.NUMERIC)) == len(md.NUMERIC)
    assert len(set(md.CATEGORICAL)) == len(md.CATEGORICAL)
    assert not set(md.NUMERIC) & set(md.CATEGORICAL)
    assert md.TARGET not in md.NUMERIC + md.CATEGORICAL
    assert md.GROUP not in md.NUMERIC + md.CATEGORICAL, \
        "patient_nbr as a feature would be identity leakage by construction"


def test_engineered_features_from_the_cleaning_stage_are_actually_used():
    """The derived columns build_dataset.py creates have to reach the model."""
    for col in ("prior_visits_total", "had_prior_inpatient",
                "had_prior_emergency", "n_drugs_changed", "n_drugs_active",
                "a1c_measured", "glucose_measured", "age_midpoint"):
        assert col in md.NUMERIC, col
    for col in ("diag_1_group", "diag_2_group", "diag_3_group"):
        assert col in md.CATEGORICAL, col


def test_an_unknown_pipeline_kind_is_rejected():
    with pytest.raises(ValueError):
        md.make_pipeline("random-forest")


# --------------------------------------------------------------------------
# Pipeline construction
# --------------------------------------------------------------------------

def test_the_logistic_model_keeps_the_settings_the_readme_reasons_about():
    """Finding 3 is specifically about class_weight='balanced'. If someone
    drops it the calibration finding evaporates and the README goes stale."""
    clf = md.make_pipeline("logistic").named_steps["clf"]
    assert isinstance(clf, LogisticRegression)
    assert clf.class_weight == "balanced"
    assert clf.C == 0.1
    assert clf.random_state == md.SEED

    steps = dict((name, trans) for name, trans, _ in
                 md.make_pipeline("logistic").named_steps["pre"].transformers)
    assert isinstance(steps["num"], StandardScaler)
    assert isinstance(steps["cat"], OneHotEncoder)
    assert steps["cat"].handle_unknown == "ignore"


def test_the_gbm_is_seeded_and_uses_early_stopping():
    clf = md.make_pipeline("gbm").named_steps["clf"]
    assert isinstance(clf, HistGradientBoostingClassifier)
    assert clf.random_state == md.SEED
    assert clf.early_stopping is True


def test_the_gbm_categorical_indices_point_at_the_encoded_categorical_columns():
    """The indices are computed as range(len(NUMERIC), len(NUMERIC)+len(CAT)).

    That arithmetic is only correct while the ColumnTransformer emits the
    numeric block first. Rather than restate the arithmetic, this fits the
    preprocessor on real values and checks the matrix: the declared
    categorical positions must hold small consecutive integers produced by the
    OrdinalEncoder, and the positions before them must hold the untouched
    numeric values.
    """
    n = 40
    rng = np.random.default_rng(0)
    X = pd.DataFrame({c: rng.uniform(10, 20, n) for c in md.NUMERIC})
    for c in md.CATEGORICAL:
        X[c] = rng.choice(["alpha", "beta", "gamma"], n)

    pipe = md.make_pipeline("gbm")
    pre = pipe.named_steps["pre"]
    matrix = np.asarray(pre.fit_transform(X), dtype=float)
    declared = list(pipe.named_steps["clf"].categorical_features)

    assert matrix.shape[1] == len(md.NUMERIC) + len(md.CATEGORICAL)
    assert declared == list(range(len(md.NUMERIC), matrix.shape[1]))

    numeric_block = matrix[:, :len(md.NUMERIC)]
    categorical_block = matrix[:, declared]
    # numeric values are passed through untouched (all >= 10 by construction)
    assert np.allclose(numeric_block, X[md.NUMERIC].to_numpy())
    # every declared categorical column is an ordinal code in {0, 1, 2}
    assert set(np.unique(categorical_block)) <= {0.0, 1.0, 2.0}
    assert isinstance(
        dict((name, t) for name, t, _ in pre.transformers)["cat"],
        OrdinalEncoder)


def test_a_category_unseen_in_training_is_encoded_as_minus_one_not_a_crash():
    """handle_unknown='use_encoded_value' with unknown_value=-1. A hold-out
    encounter with a rare specialty must score, not raise."""
    n = 30
    rng = np.random.default_rng(1)
    X = pd.DataFrame({c: rng.uniform(0, 5, n) for c in md.NUMERIC})
    for c in md.CATEGORICAL:
        X[c] = "seen"
    pre = md.make_pipeline("gbm").named_steps["pre"].fit(X)

    novel = X.iloc[:1].copy()
    novel["medical_specialty"] = "a-specialty-that-was-never-in-training"
    out = np.asarray(pre.transform(novel), dtype=float)
    assert out[0, len(md.NUMERIC) + md.CATEGORICAL.index("medical_specialty")] == -1.0


def test_the_one_hot_encoder_tolerates_an_unseen_level(encounter_frame):
    df = encounter_frame(n=300, seed=2)
    X = df[md.NUMERIC + md.CATEGORICAL]
    pre = md.make_pipeline("logistic").named_steps["pre"].fit(X)
    novel = X.iloc[:1].copy()
    novel["race"] = "never-seen-before"
    assert pre.transform(novel).shape[1] == pre.transform(X.iloc[:1]).shape[1]


def test_a_fitted_pipeline_returns_probabilities_in_the_unit_interval(encounter_frame):
    df = encounter_frame(n=400, seed=4)
    X, y = df[md.NUMERIC + md.CATEGORICAL], df[md.TARGET].to_numpy()
    for kind in ("logistic", "gbm"):
        p = md.make_pipeline(kind).fit(X, y).predict_proba(X)[:, 1]
        assert p.shape == (len(df),)
        assert p.min() >= 0.0 and p.max() <= 1.0
        assert p.std() > 0, f"{kind} collapsed to a constant prediction"


# --------------------------------------------------------------------------
# The leakage experiment -- the point of the project
# --------------------------------------------------------------------------

def test_a_random_split_invents_performance_that_a_grouped_split_does_not(
        patient_fingerprint_frame, monkeypatch):
    """On data whose only signal is patient identity the gap must be maximal.

    Sixteen patients, twenty encounters each, one numeric column per patient
    set to 1 for that patient alone, label constant within patient and
    unpredictable from anything that generalises.

      random split      every test patient was also in training, so the model
                        reads its column straight off  -> AUC 1.0
      patient-grouped   the test patients' columns are all-zero in training,
                        their coefficients never move off zero, every test row
                        scores the intercept -> a total tie -> AUC 0.5

    If StratifiedGroupKFold were ever swapped for StratifiedKFold, or the
    groups argument dropped, the grouped number would jump to 1.0 and this
    test fails.
    """
    monkeypatch.setattr(md, "N_SPLITS", 2)
    df = patient_fingerprint_frame()

    naive_mean, _, naive_folds = md.cross_val_auc(df, "logistic", grouped=False)
    grouped_mean, _, grouped_folds = md.cross_val_auc(df, "logistic", grouped=True)

    assert naive_mean == pytest.approx(1.0, abs=0.01)
    assert grouped_mean == pytest.approx(0.5, abs=0.01)
    assert naive_mean - grouped_mean > 0.4
    assert len(naive_folds) == len(grouped_folds) == 2


def test_the_grouped_splitter_never_puts_a_patient_on_both_sides(
        encounter_frame, monkeypatch):
    """Directly inspect the folds the experiment uses."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    monkeypatch.setattr(md, "N_SPLITS", 3)
    df = encounter_frame(n=600, seed=6, n_patients=90)
    groups = df[md.GROUP].to_numpy()
    y = df[md.TARGET].to_numpy()
    X = df[md.NUMERIC + md.CATEGORICAL]

    grouped = StratifiedGroupKFold(n_splits=3, shuffle=True,
                                   random_state=md.SEED)
    for train_idx, test_idx in grouped.split(X, y, groups):
        assert not set(groups[train_idx]) & set(groups[test_idx])

    # and confirm the naive splitter really does leak on this frame, so the
    # assertion above is testing something
    naive = StratifiedKFold(n_splits=3, shuffle=True, random_state=md.SEED)
    leaks = [len(set(groups[a]) & set(groups[b])) for a, b in naive.split(X, y)]
    assert min(leaks) > 0


# --------------------------------------------------------------------------
# The hand-off to evaluate.py and registry.py
# --------------------------------------------------------------------------

def test_fit_final_writes_the_contract_the_downstream_scripts_read(
        encounter_frame, tmp_path, monkeypatch):
    """fit_final is the only producer of holdout_predictions.parquet.

    Both src/evaluate.py and src/registry.py read that file by column name, so
    the column set is an interface. This also checks that the two carried-over
    baseline rules stay aligned with the rows they came from -- they are
    indexed positionally out of the source frame, which is exactly the kind of
    thing that goes wrong without noticing.
    """
    monkeypatch.setattr(md, "N_SPLITS", 4)
    monkeypatch.setattr(md, "ROOT", tmp_path)
    monkeypatch.setattr(md, "INTERIM", tmp_path / "interim")
    df = encounter_frame(n=800, seed=8, n_patients=200)

    aucs = md.fit_final(df)
    assert set(aucs) == {"logistic", "gbm"}
    assert all(0.0 <= v <= 1.0 for v in aucs.values())

    held = pd.read_parquet(tmp_path / "interim" / "holdout_predictions.parquet")
    assert list(held.columns) == ["patient_nbr", "y_true", "p_logistic",
                                  "p_gbm", "rule_prior_inpatient",
                                  "rule_prior_visits"]
    assert len(held) > 0
    for col in ("p_logistic", "p_gbm"):
        assert held[col].between(0, 1).all()

    # every hold-out row must be a real row of the source frame, with its own
    # label and its own prior-inpatient count still attached
    source = set(map(tuple, df[[md.GROUP, md.TARGET, "number_inpatient",
                                "prior_visits_total"]].to_numpy()))
    written = set(map(tuple, held[["patient_nbr", "y_true",
                                   "rule_prior_inpatient",
                                   "rule_prior_visits"]].to_numpy()))
    assert written <= source

    # the AUC fit_final prints is the AUC of what it wrote
    assert roc_auc_score(held["y_true"], held["p_gbm"]) == \
        pytest.approx(aucs["gbm"], abs=5e-5)
