"""Tests for the cleaning stage: the three documented traps and the features.

build_dataset.main() reads a CSV and writes a parquet, so it is driven here by
pointing its module-level paths at tmp_path and feeding it the miniature raw
file from conftest. No part of data/external or data/interim is touched.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

import build_dataset as bd
from conftest import RAW_ROWS


# --------------------------------------------------------------------------
# Documented constants
# --------------------------------------------------------------------------

def test_the_unreadmittable_disposition_sets_are_exactly_as_documented():
    """Trap 1: dispositions that make readmission impossible.

    11/19/20/21 are the four 'expired' codes in the UCI mapping and 13/14 are
    the two hospice codes. Pinned because adding or losing one silently
    changes the cohort and every headline number with it.
    """
    assert bd.DECEASED_DISPOSITIONS == {11, 19, 20, 21}
    assert bd.HOSPICE_DISPOSITIONS == {13, 14}
    assert not (bd.DECEASED_DISPOSITIONS & bd.HOSPICE_DISPOSITIONS)
    # discharge home (1) and transfer (3, 6) must never be in there
    assert not ({1, 2, 3, 6} & (bd.DECEASED_DISPOSITIONS | bd.HOSPICE_DISPOSITIONS))


def test_age_midpoints_are_the_arithmetic_centre_of_their_bracket():
    """Derived from the bracket label itself, not copied from the source."""
    assert len(bd.AGE_MIDPOINT) == 10
    for bracket, midpoint in bd.AGE_MIDPOINT.items():
        lo, hi = map(int, re.match(r"\[(\d+)-(\d+)\)", bracket).groups())
        assert hi - lo == 10, bracket
        assert midpoint == (lo + hi) / 2
    # the ten UCI brackets tile 0-100 with no gap and no overlap
    lows = sorted(int(re.match(r"\[(\d+)", b).group(1)) for b in bd.AGE_MIDPOINT)
    assert lows == list(range(0, 100, 10))


def test_dropped_columns_carry_a_reason_and_include_the_zero_variance_drugs():
    assert set(bd.DROP_COLUMNS) == {
        "weight", "encounter_id", "examide", "citoglipton",
        "metformin-rosiglitazone", "glimepiride-pioglitazone",
    }
    assert all(isinstance(v, str) and v for v in bd.DROP_COLUMNS.values())
    # a dropped drug must not also be counted as an active regimen drug
    assert not (set(bd.DROP_COLUMNS) & set(bd.DRUG_COLUMNS))


# --------------------------------------------------------------------------
# End-to-end on the miniature raw file
# --------------------------------------------------------------------------

@pytest.fixture()
def cleaned(raw_csv: Path, tmp_path: Path, monkeypatch) -> pd.DataFrame:
    monkeypatch.setattr(bd, "ROOT", tmp_path)
    monkeypatch.setattr(bd, "RAW", raw_csv)
    monkeypatch.setattr(bd, "INTERIM", tmp_path / "interim")
    monkeypatch.setattr(bd, "OUT", tmp_path / "interim" / "encounters.parquet")
    assert bd.main() == 0
    return pd.read_parquet(tmp_path / "interim" / "encounters.parquet")


def test_main_exits_nonzero_when_the_raw_file_is_absent(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(bd, "RAW", tmp_path / "nope" / "diabetic_data.csv")
    assert bd.main() == 1
    assert "src/download.py" in capsys.readouterr().err


def test_deceased_and_hospice_encounters_are_removed(cleaned):
    """Trap 1. Four of the eight input rows are unreadmittable by construction.

    Rows with discharge_disposition_id 11, 13, 19 and 14 (patients 2, 3, 4, 5)
    must be gone; the survivors are patients 1 (twice), 6 and 7.
    """
    assert len(cleaned) == 4
    assert sorted(cleaned["patient_nbr"]) == [1, 1, 6, 7]
    assert not set(cleaned["discharge_disposition_id"]) & {11, 13, 14, 19}


def test_the_thirty_day_target_collapses_only_the_right_level(cleaned):
    """Trap 2. '<30' is the event; '>30' joins the negatives, it is not dropped."""
    got = dict(zip(cleaned["readmitted"], cleaned["readmitted_30d"]))
    assert got == {"<30": 1, ">30": 0, "NO": 0}
    # the >30 encounter survived rather than being discarded
    assert (cleaned["readmitted"] == ">30").sum() == 1
    assert cleaned["readmitted_30d"].tolist() == [1, 0, 0, 1]


def test_the_same_patient_still_appears_more_than_once(cleaned):
    """Trap 3. Cleaning must not deduplicate patients -- that is what the
    grouped split in src/model.py exists to handle."""
    assert cleaned["patient_nbr"].nunique() == 3
    assert len(cleaned) > cleaned["patient_nbr"].nunique()


def test_age_is_replaced_by_its_bracket_midpoint(cleaned):
    # [70-80) -> 75, [70-80) -> 75, [0-10) -> 5, [90-100) -> 95
    assert cleaned["age_midpoint"].tolist() == [75, 75, 5, 95]


def test_diagnosis_columns_are_grouped_and_missing_is_preserved(cleaned):
    assert cleaned["diag_1_group"].tolist() == [
        "Diabetes",       # 250.83
        "Respiratory",    # 486
        "Digestive",      # 787.91
        "Genitourinary",  # 599.0
    ]
    assert cleaned["diag_2_group"].tolist() == [
        "Circulatory",    # 428
        "Missing",        # '?' -> NaN -> Missing
        "Injury",         # 820
        "Neoplasms",      # 185
    ]
    assert cleaned["diag_3_group"].tolist() == [
        "Other",          # V57
        "Diabetes",       # 250
        "Musculoskeletal",  # 715
        "Other",          # E909
    ]
    # the original code columns survive alongside the grouped ones
    assert {"diag_1", "diag_2", "diag_3"} <= set(cleaned.columns)


def test_prior_utilisation_features(cleaned):
    """outpatient + emergency + inpatient, plus the two 'any' flags.

    Kept rows carry (outpatient, emergency, inpatient) =
    (2,0,3), (0,1,0), (0,0,0), (1,2,4)  ->  totals 5, 1, 0, 7.
    """
    assert cleaned["prior_visits_total"].tolist() == [5, 1, 0, 7]
    assert cleaned["had_prior_inpatient"].tolist() == [1, 0, 0, 1]
    assert cleaned["had_prior_emergency"].tolist() == [0, 1, 0, 1]


def test_drug_activity_counts_distinguish_changed_from_active(cleaned):
    """n_drugs_changed counts Up/Down; n_drugs_active adds Steady.

    The three drug columns in the fixture are metformin / insulin / glipizide:
      row 0  Up,     Steady, No      -> changed 1, active 2
      row 1  Down,   Down,   Steady  -> changed 2, active 3
      row 6  No,     No,     No      -> changed 0, active 0
      row 7  No,     Up,     Down    -> changed 2, active 2
    """
    assert cleaned["n_drugs_changed"].tolist() == [1, 2, 0, 2]
    assert cleaned["n_drugs_active"].tolist() == [2, 3, 0, 2]
    # changed is always a subset of active
    assert (cleaned["n_drugs_changed"] <= cleaned["n_drugs_active"]).all()


def test_lab_measured_flags_are_computed_before_the_missing_fill(cleaned):
    """The ordering here is load-bearing and easy to break.

    In the raw file 'None' means the test was not run, and read_csv turns that
    spelling into a real NaN. a1c_measured/glucose_measured must be taken from
    that NaN pattern *before* the columns are filled with the literal string
    'Missing' -- otherwise notna() is true everywhere and both flags become
    constant 1, i.e. two dead features that nothing would flag.

      row 0  A1C '>8',  glu 'None'  -> 1, 0
      row 1  A1C 'None', glu '>200' -> 0, 1
      row 6  A1C 'None', glu 'None' -> 0, 0
      row 7  A1C 'Norm', glu 'Norm' -> 1, 1
    """
    assert cleaned["a1c_measured"].tolist() == [1, 0, 0, 1]
    assert cleaned["glucose_measured"].tolist() == [0, 1, 0, 1]
    assert cleaned["a1c_measured"].nunique() == 2, "flag collapsed to a constant"
    assert cleaned["A1Cresult"].tolist() == [">8", "Missing", "Missing", "Norm"]


def test_missingness_is_kept_as_an_explicit_level_not_imputed(cleaned):
    """'?' is the dataset's missing marker and becomes the level 'Missing'."""
    assert cleaned["race"].tolist() == ["Caucasian", "Missing",
                                        "AfricanAmerican", "Missing"]
    assert cleaned["payer_code"].tolist() == ["MC", "HM", "Missing", "MC"]
    assert cleaned["medical_specialty"].tolist() == [
        "InternalMedicine", "Missing", "Family/GeneralPractice", "Nephrology"]
    for col in ("race", "medical_specialty", "payer_code", "A1Cresult",
                "max_glu_serum"):
        assert cleaned[col].notna().all(), col
    # no literal '?' survives anywhere in the frame
    assert not (cleaned.astype(str) == "?").to_numpy().any()


def test_unknown_gender_is_folded_into_missing(cleaned):
    assert cleaned["gender"].tolist() == ["Female", "Male", "Missing", "Female"]


def test_zero_variance_and_identifier_columns_are_dropped(cleaned):
    for col in bd.DROP_COLUMNS:
        assert col not in cleaned.columns, col
    # and nothing else was lost: every other raw column is still present
    survivors = set(RAW_ROWS) - set(bd.DROP_COLUMNS)
    assert survivors <= set(cleaned.columns)
