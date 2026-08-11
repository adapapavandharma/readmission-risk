"""Tests for evaluate.calibration_metrics().

Calibration is the property that caught the class_weight='balanced' problem
reported as finding 3 in the README: a model whose ranking is fine but whose
probabilities are unusable. AUC cannot see that; the calibration intercept
can. These tests pin the behaviour that makes the finding detectable.

Definitions used throughout:
  logit(p) = log(p / (1 - p))
  the calibration *slope* is the coefficient of logit(p) in a logistic
  regression of y on logit(p): 1.0 when the predictions are exactly right
  the calibration *intercept* is the shift a on the logit scale that best
  explains y under the offset model  y ~ Bernoulli(sigmoid(a + logit(p))):
  0.0 when the predictions are exactly right, negative when they are too high
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import sigmoid
from evaluate import calibration_metrics

N = 20_000


def _calibrated(seed: int = 0, n: int = N, mean_logit: float = -2.1):
    """Draw a perfectly calibrated set: y ~ Bernoulli(p) with p reported as p.

    mean_logit -2.1 puts the base rate near 12%, which is the neighbourhood of
    this project's real 11.1%.
    """
    rng = np.random.default_rng(seed)
    lin = rng.normal(mean_logit, 1.0, n)
    p = sigmoid(lin)
    y = (rng.random(n) < p).astype(int)
    return y, p, lin


# --------------------------------------------------------------------------
# The reference case
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_perfectly_calibrated_set_gives_slope_one_and_intercept_zero(seed):
    """Ground truth by construction: the labels were drawn from the reported p.

    Tolerances are wide relative to the Monte-Carlo error at n=20,000 (the
    observed spread over these seeds is slope 0.98-1.03, intercept +/-0.02)
    but far tighter than the miscalibrated cases below, which come out at
    slope 0.51 and intercept -2.00.
    """
    y, p, _ = _calibrated(seed)
    m = calibration_metrics(y, p)
    assert m["calibration_slope"] == pytest.approx(1.0, abs=0.08)
    assert m["calibration_intercept"] == pytest.approx(0.0, abs=0.08)
    assert m["mean_predicted"] == pytest.approx(m["observed_rate"], abs=0.01)


# --------------------------------------------------------------------------
# Systematic inflation -- the class_weight='balanced' failure mode
# --------------------------------------------------------------------------

@pytest.mark.parametrize("shift", [1.0, 2.0, 3.0])
def test_inflated_probabilities_give_a_negative_intercept_of_minus_the_shift(shift):
    """Add a constant on the logit scale; the intercept must recover -shift.

    That is the definition of the offset model: if reported logit = true logit
    + s, then sigmoid(a + reported) matches the truth exactly at a = -s. So a
    model that is too confident by 2 logits must report intercept ~= -2.0.

    Adding a constant on the logit scale is a strictly monotone transform, so
    the ranking -- and therefore the AUC -- is untouched. That is precisely
    why an AUC-only evaluation misses this and why the README says you cannot
    put such a number in front of a clinician.
    """
    y, p_true, lin = _calibrated(seed=0)
    m = calibration_metrics(y, sigmoid(lin + shift))
    assert m["calibration_intercept"] == pytest.approx(-shift, abs=0.10)
    assert m["calibration_intercept"] < -0.5
    # the slope is unaffected by a pure shift
    assert m["calibration_slope"] == pytest.approx(1.0, abs=0.08)


def test_inflation_shows_up_as_mean_predicted_far_above_the_observed_rate():
    """The README's finding 3 in miniature: 48% predicted, 14% observed."""
    y, _, lin = _calibrated(seed=0)
    m = calibration_metrics(y, sigmoid(lin + 2.0))
    assert m["mean_predicted"] > 3 * m["observed_rate"]
    assert m["observed_rate"] == pytest.approx(float(y.mean()), abs=1e-4)


def test_inflation_is_punished_by_the_brier_score_but_not_by_the_ranking():
    from sklearn.metrics import roc_auc_score

    y, p_true, lin = _calibrated(seed=0)
    p_bad = sigmoid(lin + 2.0)
    good, bad = calibration_metrics(y, p_true), calibration_metrics(y, p_bad)
    assert bad["brier"] > 2 * good["brier"]
    # identical ordering -> identical AUC, to within float noise
    assert roc_auc_score(y, p_bad) == pytest.approx(roc_auc_score(y, p_true),
                                                    abs=1e-9)


# --------------------------------------------------------------------------
# Over- and under-confidence -- the slope
# --------------------------------------------------------------------------

def test_doubling_the_logit_halves_the_calibration_slope():
    """Predicted logit = 2 x true logit means the recalibration slope is 0.5.

    Derivation: the slope is the b that best fits y ~ sigmoid(a + b*reported).
    With reported = 2*true, the truth is recovered at b = 0.5. Slope < 1 is
    the standard signature of predictions that are too extreme.
    """
    y, _, lin = _calibrated(seed=0)
    m = calibration_metrics(y, sigmoid(2.0 * lin))
    assert m["calibration_slope"] == pytest.approx(0.5, abs=0.06)


def test_halving_the_logit_doubles_the_calibration_slope():
    """The mirror image: predictions shrunk toward the middle give slope ~2."""
    y, _, lin = _calibrated(seed=0)
    m = calibration_metrics(y, sigmoid(0.5 * lin))
    assert m["calibration_slope"] == pytest.approx(2.0, abs=0.20)


# --------------------------------------------------------------------------
# Exactly hand-computable cases
# --------------------------------------------------------------------------

def test_brier_and_intercept_for_a_constant_prediction_at_the_base_rate():
    """10 rows, 3 positive, every prediction 0.3.

    Brier = mean (p - y)^2 = (3*(1-0.3)^2 + 7*(0-0.3)^2) / 10
          = (3*0.49 + 7*0.09) / 10 = (1.47 + 0.63) / 10 = 0.21

    The offset model's likelihood is maximised where the fitted probability
    equals the observed rate, and the prediction already is the observed rate,
    so the intercept is exactly 0. (The slope is undefined here -- logit(p) is
    a zero-variance feature -- so it is not asserted.)
    """
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    p = np.full(10, 0.3)
    m = calibration_metrics(y, p)
    assert m["brier"] == 0.21
    assert m["calibration_intercept"] == pytest.approx(0.0, abs=1e-3)
    assert m["mean_predicted"] == 0.3
    assert m["observed_rate"] == 0.3


def test_a_constant_prediction_above_the_base_rate_gives_a_negative_intercept():
    """Predict 0.5 everywhere when only 30% are positive.

    The correcting shift is logit(0.3) - logit(0.5)
                          = log(0.3/0.7) - 0 = -0.8473.
    """
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    m = calibration_metrics(y, np.full(10, 0.5))
    assert m["calibration_intercept"] == pytest.approx(np.log(0.3 / 0.7),
                                                       abs=1e-3)
    assert m["brier"] == 0.25          # (p - y)^2 = 0.25 for every row


def test_a_perfect_prediction_has_a_zero_brier_score():
    y = np.array([1, 0, 1, 0, 1])
    m = calibration_metrics(y, y.astype(float))
    assert m["brier"] == 0.0
    assert m["observed_rate"] == 0.6


def test_probabilities_of_exactly_zero_and_one_are_clipped_not_crashed():
    """logit(0) is -inf; the eps clip is what keeps the fit finite."""
    y = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    m = calibration_metrics(y, np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]))
    assert np.isfinite(m["calibration_slope"])
    assert np.isfinite(m["calibration_intercept"])
    assert m["brier"] == 0.0


def test_the_reported_keys_are_the_ones_the_report_and_figures_consume():
    y, p, _ = _calibrated(seed=0, n=500)
    assert set(calibration_metrics(y, p)) == {
        "brier", "calibration_slope", "calibration_intercept",
        "mean_predicted", "observed_rate"}
