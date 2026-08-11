"""Tests for evaluate.lift_table() -- the operating table the project leads on.

Every expected number below is derived by hand in the docstring of the test
that asserts it. The fixtures are 100 rows with 20 positives (base rate 0.20)
so that each depth in TOP_K = 5,10,15,20,25,30,40,50 % lands on a whole number
of patients and nothing depends on rounding.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluate import TOP_K, lift_table

N = 100
BASE = 0.20
DESCENDING = np.arange(N, 0, -1, dtype=float)   # score[i] > score[i+1]


@pytest.fixture()
def no_skill():
    """A ranking with no information: one positive every five rows.

    y = [1,0,0,0,0] * 20 with a strictly descending score, so the top-k slice
    always contains exactly k/5 positives -- precision 0.20 at every depth,
    identical to the base rate.
    """
    return np.tile(np.array([1, 0, 0, 0, 0]), 20), DESCENDING


@pytest.fixture()
def perfect():
    """A perfect ranking: all 20 positives occupy the 20 highest scores."""
    return np.array([1] * 20 + [0] * 80), DESCENDING


@pytest.fixture()
def inverted():
    """The worst possible ranking: all 20 positives at the bottom."""
    return np.array([0] * 80 + [1] * 20), DESCENDING


# --------------------------------------------------------------------------
# The no-skill baseline -- lift must be exactly 1.0
# --------------------------------------------------------------------------

def test_a_random_order_score_has_lift_one_at_every_depth(no_skill):
    """Lift is precision / base rate, so an uninformative ranking gives 1.0.

    This is the reference line the whole README table is read against: if a
    change to lift_table made a no-information ranking look like lift 1.3, the
    headline '1.89x enrichment' would be meaningless.
    """
    y, score = no_skill
    lt = lift_table(y, score, base=BASE)
    assert lt["lift"].tolist() == [1.0] * len(TOP_K)
    assert lt["precision_pct"].tolist() == [20.0] * len(TOP_K)


def test_number_needed_to_screen_at_the_base_rate_is_one_over_the_base_rate(no_skill):
    """1 / 0.20 = 5.0 patients contacted per readmission caught."""
    y, score = no_skill
    assert lift_table(y, score, base=BASE)["number_needed_to_screen"].tolist() \
        == [5.0] * len(TOP_K)


def test_with_no_skill_recall_equals_list_depth(no_skill):
    """Contact 20% of the list at random, catch 20% of the readmissions."""
    y, score = no_skill
    lt = lift_table(y, score, base=BASE)
    assert lt["recall_pct"].tolist() == pytest.approx(lt["top_k_pct"].tolist())


# --------------------------------------------------------------------------
# A perfect ranking -- hand-computed row by row
# --------------------------------------------------------------------------

def test_perfect_ranking_row_values(perfect):
    """With 20 positives at the top of a 100-row list:

      top 5%  -> 5 flagged,  5 caught, recall  5/20 = 25%,  precision 100%
      top 10% -> 10 flagged, 10 caught, recall 10/20 = 50%, precision 100%
      top 20% -> 20 flagged, 20 caught, recall        100%, precision 100%
      top 25% -> 25 flagged, 20 caught, recall        100%, precision  80%
      top 50% -> 50 flagged, 20 caught, recall        100%, precision  40%

    lift = precision / 0.20, so 100% precision is lift 5.0 -- the ceiling at
    this base rate, since you cannot do better than an all-positive list.
    """
    y, score = perfect
    lt = lift_table(y, score, base=BASE).set_index("top_k_pct")

    assert lt.loc[5.0, "patients_flagged"] == 5
    assert lt.loc[5.0, "readmissions_caught"] == 5
    assert lt.loc[5.0, "recall_pct"] == 25.0
    assert lt.loc[5.0, "precision_pct"] == 100.0
    assert lt.loc[5.0, "lift"] == 5.0
    assert lt.loc[5.0, "number_needed_to_screen"] == 1.0

    assert lt.loc[20.0, "recall_pct"] == 100.0
    assert lt.loc[20.0, "lift"] == 5.0

    assert lt.loc[25.0, "readmissions_caught"] == 20    # cannot exceed total
    assert lt.loc[25.0, "precision_pct"] == 80.0
    assert lt.loc[25.0, "lift"] == 4.0

    assert lt.loc[50.0, "precision_pct"] == 40.0
    assert lt.loc[50.0, "lift"] == 2.0


def test_lift_cannot_exceed_one_over_the_base_rate(perfect):
    """Ceiling check: 1 / 0.20 = 5.0."""
    y, score = perfect
    assert lift_table(y, score, base=BASE)["lift"].max() == 5.0


# --------------------------------------------------------------------------
# Degenerate slice
# --------------------------------------------------------------------------

def test_an_empty_slice_reports_no_nns_rather_than_dividing_by_zero(inverted):
    """Nothing caught -> precision 0, lift 0, and NNS is None, not inf."""
    y, score = inverted
    lt = lift_table(y, score, base=BASE)
    assert lt["readmissions_caught"].tolist() == [0] * len(TOP_K)
    assert lt["recall_pct"].tolist() == [0.0] * len(TOP_K)
    assert lt["lift"].tolist() == [0.0] * len(TOP_K)
    assert lt["number_needed_to_screen"].isna().all()


# --------------------------------------------------------------------------
# Properties that must hold for any y / score
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_recall_is_monotonically_non_decreasing_in_list_depth(seed):
    """Working further down the list can never catch fewer readmissions.

    Random scores, five seeds -- the property holds regardless of how good the
    ranking is, so this is a real invariant and not a restatement of the
    fixture.
    """
    rng = np.random.default_rng(seed)
    y = (rng.random(N) < 0.2).astype(int)
    if y.sum() == 0:                       # keep the fixture well posed
        y[0] = 1
    lt = lift_table(y, rng.random(N), base=float(y.mean()))
    recall = lt["recall_pct"].to_numpy()
    assert np.all(np.diff(recall) >= 0)
    assert np.all(np.diff(lt["readmissions_caught"].to_numpy()) >= 0)
    assert np.all(np.diff(lt["patients_flagged"].to_numpy()) > 0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_recall_and_precision_agree_with_the_counts_in_the_same_row(seed):
    """recall = caught/positives and precision = caught/flagged, per row."""
    rng = np.random.default_rng(seed)
    y = (rng.random(N) < 0.3).astype(int)
    total_pos = int(y.sum())
    lt = lift_table(y, rng.random(N), base=float(y.mean()))
    for row in lt.itertuples():
        # the table rounds both percentages to one decimal place
        assert row.recall_pct == round(
            100 * row.readmissions_caught / total_pos, 1)
        assert row.precision_pct == round(
            100 * row.readmissions_caught / row.patients_flagged, 1)
        assert row.readmissions_caught <= row.patients_flagged
        assert row.readmissions_caught <= total_pos


def test_flagged_count_is_the_requested_share_of_the_list(no_skill):
    y, score = no_skill
    lt = lift_table(y, score, base=BASE)
    assert lt["patients_flagged"].tolist() == [5, 10, 15, 20, 25, 30, 40, 50]
    assert lt["top_k_pct"].tolist() == [k * 100 for k in TOP_K]


def test_at_least_one_patient_is_flagged_on_a_tiny_list():
    """round(0.05 * 4) is 0; the max(..., 1) floor keeps precision defined."""
    y = np.array([1, 0, 0, 0])
    lt = lift_table(y, np.array([4.0, 3.0, 2.0, 1.0]), base=0.25)
    assert lt["patients_flagged"].min() == 1
    assert lt.iloc[0]["precision_pct"] == 100.0     # the single flagged row is the positive


# --------------------------------------------------------------------------
# Tie handling
# --------------------------------------------------------------------------

def test_ties_are_broken_stably_by_original_order():
    """kind='mergesort' is a stable sort, so equal scores keep input order.

    With an all-constant score the table must be exactly what you get by
    walking the list front to back -- deterministic, and reproducible across
    runs and machines. An unstable sort would make the risk board wobble.
    """
    y = np.array([1] * 20 + [0] * 80)
    constant = lift_table(y, np.zeros(N), base=BASE)
    front_to_back = lift_table(y, DESCENDING, base=BASE)
    assert constant.equals(front_to_back)


def test_the_table_is_deterministic_across_repeated_calls():
    rng = np.random.default_rng(11)
    y = (rng.random(N) < 0.2).astype(int)
    score = rng.random(N).round(2)          # forces many ties
    first = lift_table(y, score, base=float(y.mean()))
    assert first.equals(lift_table(y, score, base=float(y.mean())))


def test_lift_is_precision_relative_to_the_base_rate_you_pass_in():
    """Doubling the reference base rate halves every lift value."""
    y = np.array([1] * 20 + [0] * 80)
    at_20 = lift_table(y, DESCENDING, base=0.20)["lift"].to_numpy()
    at_40 = lift_table(y, DESCENDING, base=0.40)["lift"].to_numpy()
    # abs tolerance because both sides are rounded to two decimals first
    assert at_40 == pytest.approx(at_20 / 2, abs=0.01)
