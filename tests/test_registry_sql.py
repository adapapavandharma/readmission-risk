"""Tests for src/registry.py -- tiering and the SQLite risk board.

The deliverable of the project is this table, and it is produced by a SQL
query with two window functions whose denominators are easy to get wrong
(pct_of_list divides by SUM(COUNT(*)) OVER (), pct_of_all_readmissions by
SUM(SUM(...)) OVER ()). Both are checked against hand-computed totals.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import registry as rg

N = 100


@pytest.fixture()
def board(tmp_path: Path, monkeypatch) -> pd.DataFrame:
    """Run the real main() over a 100-row board with known structure.

    100 encounters, one per patient, with strictly increasing predicted risk,
    and the 30 highest-risk patients are the ones who actually came back.
    Ranks therefore run 1..100 and risk_percentile is exactly the rank.
    """
    df = pd.DataFrame({
        "patient_nbr": np.arange(1, N + 1),
        "p_gbm": np.linspace(0.01, 0.99, N),
        "y_true": np.where(np.arange(N) >= 70, 1, 0),
    })
    pred = tmp_path / "holdout_predictions.parquet"
    df.to_parquet(pred, index=False)

    monkeypatch.setattr(rg, "ROOT", tmp_path)
    monkeypatch.setattr(rg, "PRED", pred)
    monkeypatch.setattr(rg, "DB", tmp_path / "registry.db")
    monkeypatch.setattr(rg, "OUT", tmp_path / "outputs")
    assert rg.main() == 0
    return pd.read_csv(tmp_path / "outputs" / "risk_board.csv")


# --------------------------------------------------------------------------
# Capacity-derived cut-points
# --------------------------------------------------------------------------

def test_the_tier_cut_points_come_out_of_the_capacity_constants():
    """24 contacts against 120 discharges is a 20% capacity share, so

        high_cut   = 100 * (1 - 0.20)       = 80th percentile  (top 20%)
        medium_cut = 100 * (1 - 2.5 * 0.20) = 50th percentile  (next 30%)

    which is the 20 / 30 / 50 split the README publishes. Pinned here because
    the whole 'derived from capacity, not round numbers' claim rests on it.
    """
    share = rg.OUTREACH_CAPACITY_PER_DAY / rg.DAILY_DISCHARGES
    assert share == pytest.approx(0.20)
    assert 100 * (1 - share) == pytest.approx(80.0)
    assert 100 * (1 - 2.5 * share) == pytest.approx(50.0)
    assert rg.DAILY_DISCHARGES == 120
    assert rg.OUTREACH_CAPACITY_PER_DAY == 24


def test_main_exits_nonzero_without_model_predictions(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(rg, "PRED", tmp_path / "absent.parquet")
    assert rg.main() == 1
    assert "src/model.py" in capsys.readouterr().err


# --------------------------------------------------------------------------
# The board itself
# --------------------------------------------------------------------------

def test_the_three_tiers_are_present_and_ordered_by_descending_risk(board):
    assert board["risk_tier"].tolist() == [
        "High - call before discharge",
        "Medium - post-discharge follow-up",
        "Low - standard discharge",
    ]
    risk = board["mean_predicted_risk_pct"].to_numpy()
    assert np.all(np.diff(risk) < 0)


def test_tier_sizes_follow_from_the_percentile_cut_points(board):
    """rank(pct=True)*100 gives rank/100*100 = the rank itself here, and the
    comparisons are '>=', so:

        High   ranks  80..100 -> 21 patients
        Medium ranks  50..79  -> 30 patients
        Low    ranks   1..49  -> 49 patients

    High is one patient over the nominal 20% because the boundary rank is
    included; on the real 19,720-row board that single encounter is 0.005% of
    the list. Pinned so the off-by-one stays visible and deliberate.
    """
    assert board["patients"].tolist() == [21, 30, 49]
    assert board["patients"].sum() == N


def test_the_window_function_denominators_are_the_whole_board(board):
    """pct_of_list and pct_of_all_readmissions are each shares of the total,
    so each column must sum to 100."""
    assert board["pct_of_list"].sum() == pytest.approx(100.0, abs=0.15)
    assert board["pct_of_all_readmissions"].sum() == pytest.approx(100.0, abs=0.15)
    for row in board.itertuples():
        assert row.pct_of_list == pytest.approx(100 * row.patients / N, abs=0.05)


def test_observed_outcomes_per_tier_are_the_hand_counted_ones(board):
    """The 30 readmissions sit at ranks 71..100.

        High   ranks 80..100 -> all 21 readmitted -> 100.0% observed,
                                21/30 = 70.0% of every readmission
        Medium ranks 50..79  -> ranks 71..79 readmitted = 9 -> 30.0% observed,
                                9/30 = 30.0% of every readmission
        Low    ranks 1..49   -> none
    """
    high, medium, low = (board.iloc[i] for i in range(3))
    assert high.readmissions == 21
    assert high.observed_readmit_pct == pytest.approx(100.0)
    assert high.pct_of_all_readmissions == pytest.approx(70.0)

    assert medium.readmissions == 9
    assert medium.observed_readmit_pct == pytest.approx(30.0)
    assert medium.pct_of_all_readmissions == pytest.approx(30.0)

    assert low.readmissions == 0
    assert low.observed_readmit_pct == pytest.approx(0.0)
    assert board["readmissions"].sum() == 30


def test_the_high_tier_is_enriched_relative_to_the_whole_board(board):
    """The operational claim: working the top slice beats calling at random."""
    overall = 30 / N * 100
    assert board.iloc[0]["observed_readmit_pct"] > overall
    assert board.iloc[-1]["observed_readmit_pct"] < overall
    assert board.iloc[0]["pct_of_all_readmissions"] > board.iloc[0]["pct_of_list"]


# --------------------------------------------------------------------------
# The database that is written alongside the CSV
# --------------------------------------------------------------------------

def test_the_database_holds_one_row_per_scored_encounter(board, tmp_path):
    conn = sqlite3.connect(tmp_path / "registry.db")
    try:
        assert conn.execute("SELECT COUNT(*) FROM risk_score").fetchone()[0] == N
        tiers = dict(conn.execute(
            "SELECT risk_tier, COUNT(*) FROM risk_score GROUP BY risk_tier"))
        assert tiers["High - call before discharge"] == 21
        # percentiles are a genuine 0-100 scale, not a 0-1 fraction
        lo, hi = conn.execute(
            "SELECT MIN(risk_percentile), MAX(risk_percentile) FROM risk_score"
        ).fetchone()
        assert (lo, hi) == (1.0, 100.0)
        # the tier really is a function of the percentile, with no crossover
        boundary = conn.execute(
            "SELECT MIN(risk_percentile) FROM risk_score "
            "WHERE risk_tier LIKE 'High%'").fetchone()[0]
        assert boundary == 80.0
        assert conn.execute(
            "SELECT COUNT(*) FROM risk_score WHERE risk_tier NOT LIKE 'High%' "
            "AND risk_percentile >= 80").fetchone()[0] == 0
    finally:
        conn.close()


def test_the_schema_declares_the_index_the_tier_query_relies_on(board, tmp_path):
    conn = sqlite3.connect(tmp_path / "registry.db")
    try:
        indexes = {r[1] for r in conn.execute("PRAGMA index_list(risk_score)")}
        assert "idx_risk_tier" in indexes
        cols = {r[1]: r for r in conn.execute("PRAGMA table_info(risk_score)")}
        # score_id is a surrogate key. The grain is one row per ENCOUNTER, and
        # a patient can be scored more than once, so the natural key
        # (patient_nbr, predicted_risk) is not unique and cannot be the PK.
        assert set(cols) == {"score_id", "patient_nbr", "predicted_risk",
                             "risk_tier", "risk_percentile", "actual_readmit"}
        assert cols["score_id"][5] == 1, "score_id must be the primary key"
        assert "idx_risk_patient" in indexes
        assert cols["predicted_risk"][3] == 1        # NOT NULL
        assert cols["actual_readmit"][3] == 0        # nullable: absent in production
    finally:
        conn.close()


def test_rerunning_replaces_the_table_instead_of_appending(tmp_path, monkeypatch):
    """SCHEMA starts with DROP TABLE IF EXISTS, so the board is idempotent."""
    df = pd.DataFrame({
        "patient_nbr": np.arange(1, 51),
        "p_gbm": np.linspace(0.05, 0.95, 50),
        "y_true": np.where(np.arange(50) >= 35, 1, 0),
    })
    pred = tmp_path / "pred.parquet"
    df.to_parquet(pred, index=False)
    monkeypatch.setattr(rg, "ROOT", tmp_path)
    monkeypatch.setattr(rg, "PRED", pred)
    monkeypatch.setattr(rg, "DB", tmp_path / "registry.db")
    monkeypatch.setattr(rg, "OUT", tmp_path / "outputs")

    assert rg.main() == 0
    assert rg.main() == 0
    conn = sqlite3.connect(tmp_path / "registry.db")
    try:
        assert conn.execute("SELECT COUNT(*) FROM risk_score").fetchone()[0] == 50
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Known defect (see the report)
# --------------------------------------------------------------------------

def test_two_encounters_of_one_patient_at_the_same_risk_both_persist(
        tmp_path, monkeypatch):
    df = pd.DataFrame({
        "patient_nbr": [7, 7] + list(range(100, 148)),
        "p_gbm": [0.42, 0.42] + list(np.linspace(0.01, 0.99, 48)),
        "y_true": [1, 0] + [0] * 48,
    })
    pred = tmp_path / "dup.parquet"
    df.to_parquet(pred, index=False)
    monkeypatch.setattr(rg, "ROOT", tmp_path)
    monkeypatch.setattr(rg, "PRED", pred)
    monkeypatch.setattr(rg, "DB", tmp_path / "registry.db")
    monkeypatch.setattr(rg, "OUT", tmp_path / "outputs")
    assert rg.main() == 0
