"""End-to-end test of src/evaluate.py on a synthetic hold-out frame.

This is the script that produces every number in the README, so it is worth
running once in full: the report structure, the recalibration step and the
four figures. The input is built in code by the holdout_frame fixture; nothing
is read from data/.

main() is run once for the whole module (it is the slowest thing in the suite)
and the resulting report is shared by the assertions below.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import evaluate as ev


@pytest.fixture(scope="module")
def evaluated(holdout_frame, tmp_path_factory):
    """Run main() over a hold-out whose gbm probabilities are inflated.

    inflate=1.5 logits reproduces the situation the recalibration step exists
    for, so the step has something to fix and the assertions below are not
    measuring noise.
    """
    tmp_path = tmp_path_factory.mktemp("evaluate")
    df = holdout_frame(n_patients=400, per_patient=5, seed=7, inflate=1.5)
    pred = tmp_path / "holdout_predictions.parquet"
    df.to_parquet(pred, index=False)

    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(ev, "ROOT", tmp_path)
        mp.setattr(ev, "PRED", pred)
        mp.setattr(ev, "OUT", tmp_path / "outputs")
        mp.setattr(ev, "FIGS", tmp_path / "outputs" / "figures")
        (tmp_path / "outputs").mkdir()
        assert ev.main() == 0
    finally:
        mp.undo()

    report = json.loads((tmp_path / "outputs" / "evaluation.json").read_text())
    yield df, report, tmp_path


def test_main_exits_nonzero_without_model_predictions(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ev, "PRED", tmp_path / "absent.parquet")
    assert ev.main() == 1
    assert "src/model.py" in capsys.readouterr().err


def test_the_report_describes_the_cohort_it_was_given(evaluated):
    df, report, _ = evaluated
    assert report["holdout_encounters"] == len(df)
    assert report["holdout_patients"] == df["patient_nbr"].nunique()
    assert report["base_rate"] == pytest.approx(float(df["y_true"].mean()),
                                                abs=5e-5)


def test_every_declared_model_is_scored_and_only_probabilities_are_calibrated(
        evaluated):
    """The prior-inpatient rule is a count, not a probability, so a Brier
    score or calibration slope for it would be meaningless -- and the code
    correctly gates on the 'p_' prefix."""
    _, report, _ = evaluated
    assert set(report["models"]) == set(ev.MODELS)
    for col, entry in report["models"].items():
        assert 0.0 <= entry["roc_auc"] <= 1.0
        assert 0.0 <= entry["pr_auc"] <= 1.0
        assert ("calibration" in entry) == col.startswith("p_")
    assert "calibration" not in report["models"]["rule_prior_inpatient"]


def test_pr_auc_is_reported_relative_to_the_base_rate(evaluated):
    """A PR AUC only means something next to the prevalence it beats."""
    _, report, _ = evaluated
    base = report["base_rate"]
    for entry in report["models"].values():
        assert entry["pr_auc_over_base"] == pytest.approx(
            entry["pr_auc"] / base, abs=0.01)
        assert entry["pr_auc"] > base, "no model beats a coin flip on PR AUC"


def test_the_inflated_gbm_is_flagged_as_miscalibrated_despite_a_normal_auc(
        evaluated):
    """The finding-3 shape: good ranking, unusable probabilities."""
    _, report, _ = evaluated
    gbm = report["models"]["p_gbm"]
    assert gbm["roc_auc"] > 0.6
    assert gbm["calibration"]["calibration_intercept"] < -1.0
    assert gbm["calibration"]["mean_predicted"] > \
        2 * gbm["calibration"]["observed_rate"]


def test_isotonic_recalibration_repairs_the_probability_scale(evaluated):
    """Fitted on half the hold-out patients, scored on the other half.

    With a genuinely inflated input the corrected predictions must land closer
    to the observed rate and the Brier score must fall. (On an already
    calibrated model the same step makes things slightly worse -- that is
    finding 4 in the README, and it is why this test deliberately supplies a
    miscalibrated model rather than asserting recalibration always helps.)
    """
    _, report, _ = evaluated
    rec = report["recalibration"]
    before, after = rec["before"], rec["after"]
    observed = before["observed_rate"]

    assert abs(after["mean_predicted"] - observed) < \
        abs(before["mean_predicted"] - observed)
    assert after["brier"] < before["brier"]
    assert abs(after["calibration_intercept"]) < abs(before["calibration_intercept"])


def test_recalibration_is_monotone_so_the_ranking_barely_moves(evaluated):
    """Isotonic regression is order-preserving; it can only create ties, so
    the AUC may dip a little but must not change materially."""
    _, report, _ = evaluated
    rec = report["recalibration"]
    assert rec["auc_after"] == pytest.approx(rec["auc_before"], abs=0.02)


def test_the_isotonic_fit_and_its_evaluation_never_share_a_patient(
        holdout_frame, tmp_path, monkeypatch):
    """Intercept the isotonic fit and check what it was actually trained on.

    The recalibration split has to be grouped on patient for the same reason
    the model split is: one patient's encounters look alike, so fitting the
    correction on some of them and measuring it on the others would flatter
    the result. Swapping the patient mask for a plain row mask would not raise
    anything -- this test is what would notice.
    """
    fitted_on: dict = {}
    real_isotonic = ev.IsotonicRegression

    class SpyIsotonic(real_isotonic):
        def fit(self, X, y, sample_weight=None):
            fitted_on["index"] = np.asarray(X.index)
            return super().fit(X, y, sample_weight)

    df = holdout_frame(n_patients=120, per_patient=4, seed=3)
    pred = tmp_path / "holdout_predictions.parquet"
    df.to_parquet(pred, index=False)
    monkeypatch.setattr(ev, "IsotonicRegression", SpyIsotonic)
    monkeypatch.setattr(ev, "ROOT", tmp_path)
    monkeypatch.setattr(ev, "PRED", pred)
    monkeypatch.setattr(ev, "OUT", tmp_path / "outputs")
    monkeypatch.setattr(ev, "FIGS", tmp_path / "outputs" / "figures")
    (tmp_path / "outputs").mkdir()
    assert ev.main() == 0

    in_fit = df.index.isin(fitted_on["index"])
    assert 0 < in_fit.sum() < len(df), "the calibration half is empty"
    assert not set(df.loc[in_fit, "patient_nbr"]) & \
        set(df.loc[~in_fit, "patient_nbr"])
    # and it really is about half the patients, not a token holdback
    assert df.loc[in_fit, "patient_nbr"].nunique() == \
        df["patient_nbr"].nunique() // 2


def test_the_lift_table_in_the_report_matches_the_csv_that_is_published(evaluated):
    _, report, tmp_path = evaluated
    csv = pd.read_csv(tmp_path / "outputs" / "lift_table.csv")
    from_report = pd.DataFrame(report["models"]["p_gbm"]["lift_table"])
    pd.testing.assert_frame_equal(csv[from_report.columns], from_report,
                                  check_dtype=False)
    assert csv["top_k_pct"].tolist() == [k * 100 for k in ev.TOP_K]


def test_all_four_figures_are_written(evaluated):
    _, _, tmp_path = evaluated
    figs = {p.name for p in (tmp_path / "outputs" / "figures").glob("*.png")}
    assert figs == {"lift_curve.png", "calibration.png", "roc_pr.png",
                    "decision_curve.png"}
    for name in figs:
        assert (tmp_path / "outputs" / "figures" / name).stat().st_size > 1000


def test_the_top_twenty_percent_row_the_readme_quotes_exists_for_every_model(
        evaluated):
    """README's headline is 'top 20% of the list'; TOP_K must contain it."""
    _, report, _ = evaluated
    assert 0.20 in ev.TOP_K
    for entry in report["models"].values():
        rows = {r["top_k_pct"]: r for r in entry["lift_table"]}
        assert 20.0 in rows
        assert rows[20.0]["patients_flagged"] > 0
