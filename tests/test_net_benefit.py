"""Tests for evaluate.net_benefit() -- the decision curve.

    NB(pt) = TP/n - (FP/n) * (pt / (1 - pt))

pt is the threshold probability: the risk at which a team judges one
intervention worth it. The weight pt/(1-pt) is the exchange rate between a
false positive and a true positive at that threshold.

The two reference policies a model has to beat are 'treat everyone' (flag all)
and 'treat nobody' (flag none), and both have closed-form curves, which is
what makes this function cheap to test exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluate import net_benefit

# 10 rows, 3 positive -> prevalence 0.30
Y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
PREVALENCE = 0.30


# --------------------------------------------------------------------------
# The two trivial policies
# --------------------------------------------------------------------------

def test_treat_all_tends_to_the_prevalence_as_the_threshold_goes_to_zero():
    """Flag everyone: TP/n = prevalence, FP/n = 1 - prevalence, so

        NB(pt) = prev - (1 - prev) * pt/(1 - pt)  ->  prev  as pt -> 0.

    At pt = 1e-9 the correction term is 7e-10, so the curve is the prevalence
    to nine decimals.
    """
    nb = net_benefit(Y, np.ones(len(Y)), np.array([1e-9]))
    assert nb[0] == pytest.approx(PREVALENCE, abs=1e-8)


def test_treat_all_matches_its_closed_form_at_every_threshold():
    thresholds = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    expected = PREVALENCE - (1 - PREVALENCE) * thresholds / (1 - thresholds)
    assert net_benefit(Y, np.ones(len(Y)), thresholds) == pytest.approx(expected)


def test_treat_all_crosses_zero_exactly_at_the_prevalence():
    """prev - (1-prev) * pt/(1-pt) = 0 has the single root pt = prev.

    Here: 0.3 - 0.7 * (0.3/0.7) = 0. Above that threshold, contacting
    everyone is worse than contacting nobody.
    """
    assert net_benefit(Y, np.ones(len(Y)), np.array([PREVALENCE]))[0] \
        == pytest.approx(0.0, abs=1e-12)
    assert net_benefit(Y, np.ones(len(Y)), np.array([0.5]))[0] < 0


def test_net_benefit_is_zero_when_nobody_is_flagged():
    """No TP and no FP -> 0 - 0 = 0, the 'treat nobody' reference line."""
    below_threshold = np.full(len(Y), 0.1)
    nb = net_benefit(Y, below_threshold, np.array([0.2, 0.3, 0.5, 0.9]))
    assert nb.tolist() == [0.0, 0.0, 0.0, 0.0]


def test_a_zero_risk_model_is_flat_at_zero_for_any_positive_threshold():
    nb = net_benefit(Y, np.zeros(len(Y)), np.linspace(0.02, 0.45, 20))
    assert np.array_equal(nb, np.zeros(20))


# --------------------------------------------------------------------------
# Hand-computed cases
# --------------------------------------------------------------------------

def test_a_mixed_flag_set_matches_the_arithmetic():
    """p = [.9 .8 .3 | .7 .6 .1 .1 .1 .1 .1], the first three rows positive.

    At pt = 0.2 the flagged set is {.9,.8,.3,.7,.6}: TP=3, FP=2, n=10.
        NB = 3/10 - (2/10)*(0.2/0.8) = 0.30 - 0.05 = 0.25
    At pt = 0.5 the flagged set is {.9,.8,.7,.6}: TP=2, FP=2.
        NB = 2/10 - (2/10)*(0.5/0.5) = 0.20 - 0.20 = 0.00
    """
    p = np.array([.9, .8, .3, .7, .6, .1, .1, .1, .1, .1])
    assert net_benefit(Y, p, np.array([0.2]))[0] == pytest.approx(0.25)
    assert net_benefit(Y, p, np.array([0.5]))[0] == pytest.approx(0.00)


def test_a_perfect_classifier_reaches_the_prevalence_at_every_threshold():
    """TP = all positives, FP = 0, so NB = prevalence with no penalty term.

    This is the ceiling of the decision curve: no policy can do better than
    catching every case while contacting no one else.
    """
    perfect = Y.astype(float)
    nb = net_benefit(Y, perfect, np.array([0.05, 0.2, 0.4, 0.5]))
    assert nb == pytest.approx(np.full(4, PREVALENCE))


def test_the_false_positive_penalty_grows_with_the_threshold():
    """One TP and one FP, so NB = 1/n - (1/n)*pt/(1-pt).

    n = 10, pt = 0.25 -> 0.1 - 0.1*(1/3) = 0.06667
             pt = 0.50 -> 0.1 - 0.1*1     = 0.0
             pt = 0.75 -> 0.1 - 0.1*3     = -0.2
    """
    y = np.array([1, 0] + [0] * 8)
    p = np.array([1.0, 1.0] + [0.0] * 8)
    nb = net_benefit(y, p, np.array([0.25, 0.50, 0.75]))
    assert nb == pytest.approx([1 / 15, 0.0, -0.2])


# --------------------------------------------------------------------------
# Properties
# --------------------------------------------------------------------------

def test_the_threshold_comparison_is_inclusive():
    """flag = p >= pt. A patient sitting exactly on the threshold is called.

    y = [1,1,0] with p = [.5,.5,.4] at pt = .5 gives TP=2, FP=0 -> 2/3.
    A strict '>' would flag nobody and return 0, so this discriminates.
    """
    y = np.array([1, 1, 0])
    p = np.array([0.5, 0.5, 0.4])
    assert net_benefit(y, p, np.array([0.5]))[0] == pytest.approx(2 / 3)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_net_benefit_is_non_increasing_in_the_threshold(seed):
    """Raising pt can only drop patients from the flagged set and can only
    increase the penalty per false positive, so the curve never rises."""
    rng = np.random.default_rng(seed)
    y = (rng.random(400) < 0.15).astype(int)
    p = rng.random(400)
    nb = net_benefit(y, p, np.linspace(0.02, 0.45, 60))
    assert np.all(np.diff(nb) <= 1e-12)


def test_net_benefit_never_exceeds_the_prevalence():
    """The TP term is capped at prevalence and the FP term is non-negative."""
    rng = np.random.default_rng(5)
    y = (rng.random(500) < 0.2).astype(int)
    p = rng.random(500)
    nb = net_benefit(y, p, np.linspace(0.02, 0.45, 40))
    assert nb.max() <= float(y.mean()) + 1e-12


def test_the_output_lines_up_one_to_one_with_the_thresholds():
    thresholds = np.linspace(0.02, 0.45, 90)
    nb = net_benefit(Y, np.linspace(0, 1, len(Y)), thresholds)
    assert isinstance(nb, np.ndarray)
    assert nb.shape == thresholds.shape
