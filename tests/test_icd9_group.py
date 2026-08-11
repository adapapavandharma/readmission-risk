"""Tests for build_dataset.icd9_group().

The diagnosis grouping is the only clinical judgement encoded in this
repository, it feeds three model features (diag_1/2/3_group), and it is
silent when it is wrong: a misgrouped code produces a plausible category, not
an error. So it gets an independent oracle rather than spot checks.

Reference for the expected mapping: Strack et al., "Impact of HbA1c
Measurement on Hospital Readmission Rates", BioMed Research International,
2014, Table 2 -- transcribed below in STRACK_TABLE without looking at the
implementation, so the two can disagree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from build_dataset import icd9_group

LABELS = {"Diabetes", "Circulatory", "Respiratory", "Digestive", "Injury",
          "Musculoskeletal", "Genitourinary", "Neoplasms", "Other", "Missing"}

# Strack et al. Table 2, as ranges over the integer part of the ICD-9 code.
STRACK_TABLE = [
    ("Circulatory", [(390, 459), (785, 785)]),
    ("Respiratory", [(460, 519), (786, 786)]),
    ("Digestive", [(520, 579), (787, 787)]),
    ("Diabetes", [(250, 250)]),
    ("Injury", [(800, 999)]),
    ("Musculoskeletal", [(710, 739)]),
    ("Genitourinary", [(580, 629), (788, 788)]),
    ("Neoplasms", [(140, 239)]),
]


def expected_group(integer_code: int) -> str:
    """Independent oracle for whole-number ICD-9 codes."""
    for label, ranges in STRACK_TABLE:
        if any(lo <= integer_code <= hi for lo, hi in ranges):
            return label
    return "Other"


# --------------------------------------------------------------------------
# The whole integer code space, against the independent oracle
# --------------------------------------------------------------------------

def test_every_whole_icd9_code_matches_the_strack_table():
    """Sweep 001-999 and compare against STRACK_TABLE, not against the code.

    This catches a shifted range boundary anywhere in the function, which spot
    checks of a handful of codes would not.
    """
    disagreements = [
        (code, icd9_group(str(code)), expected_group(code))
        for code in range(1, 1000)
        if icd9_group(str(code)) != expected_group(code)
    ]
    assert disagreements == []


def test_the_grouping_is_a_partition_of_the_code_space():
    """Every whole code lands in exactly one documented label, never a typo."""
    produced = {icd9_group(str(code)) for code in range(1, 1000)}
    assert produced <= LABELS
    # all eight clinical categories are actually reachable, i.e. no branch is
    # dead because an earlier branch shadows it
    assert produced == LABELS - {"Missing"}


# --------------------------------------------------------------------------
# Diabetes: the 250.xx decision, and the value-set guard
# --------------------------------------------------------------------------

@pytest.mark.parametrize("code", ["250", "250.0", "250.00", "250.13", "250.6",
                                  "250.83", "250.92", "250.99"])
def test_all_250_subcodes_are_diabetes(code):
    """250.xx is diabetes mellitus in full; the subcode is the complication."""
    assert icd9_group(code) == "Diabetes"


@pytest.mark.parametrize("code", ["249", "249.0", "249.91", "251", "251.0",
                                  "252", "648.0", "648.00", "775.1"])
def test_only_250_is_diabetes(code):
    """The half-open interval [250, 251) is deliberate, so pin both edges.

    249.xx (secondary diabetes), 251.x (other pancreatic disorders), 648.0x
    (diabetes complicating pregnancy) and 775.1 (neonatal diabetes) are all
    diabetes-adjacent and all deliberately excluded, because Strack et al.
    define the Diabetes group as 250.xx only.
    """
    assert icd9_group(code) != "Diabetes"


@pytest.mark.parametrize("code", ["790.2", "790.21", "790.22", "790.29",
                                  "277.7", "648.8", "648.80"])
def test_prediabetes_is_not_in_the_diabetes_group(code):
    """Regression guard on a documented decision: prediabetes is NOT diabetes.

    790.21 (impaired fasting glucose), 790.22 (impaired glucose tolerance) and
    790.29 (other abnormal glucose) are the ICD-9 expression of prediabetes --
    the same concept as SNOMED 15777000 -- and 648.8x is abnormal glucose
    tolerance in pregnancy. Folding any of them into Diabetes would silently
    change the meaning of the diag_*_group features and inflate the diabetic
    share of the cohort. This test exists so that a future edit cannot
    reintroduce them without a red build.
    """
    assert icd9_group(code) != "Diabetes"


# --------------------------------------------------------------------------
# Every clinical branch, at both endpoints
# --------------------------------------------------------------------------

@pytest.mark.parametrize("code,group", [
    # lower edge, an interior code, upper edge -- for each range
    ("390", "Circulatory"), ("428", "Circulatory"), ("459", "Circulatory"),
    ("460", "Respiratory"), ("486", "Respiratory"), ("519", "Respiratory"),
    ("520", "Digestive"), ("550", "Digestive"), ("579", "Digestive"),
    ("580", "Genitourinary"), ("599", "Genitourinary"), ("629", "Genitourinary"),
    ("710", "Musculoskeletal"), ("724", "Musculoskeletal"),
    ("739", "Musculoskeletal"),
    ("800", "Injury"), ("820", "Injury"), ("999", "Injury"),
    ("140", "Neoplasms"), ("185", "Neoplasms"), ("239", "Neoplasms"),
])
def test_range_endpoints_are_inclusive(code, group):
    assert icd9_group(code) == group


@pytest.mark.parametrize("code", [
    "389", "240",      # just below Circulatory / just above Diabetes+thyroid
    "139", "709",      # just below Neoplasms / just below Musculoskeletal
    "740", "799",      # just above Musculoskeletal / top of the symptom chapter
    "630",             # just above Genitourinary
])
def test_codes_just_outside_a_range_fall_through_to_other(code):
    assert icd9_group(code) == "Other"


@pytest.mark.parametrize("code,group", [
    ("459.0", "Circulatory"),     # .0 keeps the float <= the 459 upper bound
    ("998.59", "Injury"),         # 998 < 999, so the decimal is harmless here
    ("428.32", "Circulatory"),
    ("486.0", "Respiratory"),
    ("820.09", "Injury"),
])
def test_sub_decimal_codes_inside_a_range_keep_their_chapter(code, group):
    """Decimals are fine anywhere except at a range's upper bound (see the
    xfail at the bottom of this file)."""
    assert icd9_group(code) == group


@pytest.mark.parametrize("code,group", [
    ("785", "Circulatory"),     # symptoms involving the cardiovascular system
    ("785.4", "Circulatory"),
    ("785.52", "Circulatory"),
    ("786", "Respiratory"),     # symptoms involving the respiratory system
    ("786.05", "Respiratory"),
    ("786.50", "Respiratory"),
    ("787", "Digestive"),       # symptoms involving the digestive system
    ("787.91", "Digestive"),
    ("788", "Genitourinary"),   # symptoms involving the urinary system
    ("788.30", "Genitourinary"),
])
def test_symptom_codes_are_routed_to_their_organ_system(code, group):
    """785-788 sit outside every range and are pulled in by integer truncation.

    These are the only codes where the implementation truncates the decimal,
    which is why the sub-decimal forms are worth pinning explicitly.
    """
    assert icd9_group(code) == group


@pytest.mark.parametrize("code", ["780", "781", "782", "783", "784", "789",
                                  "790", "796", "799", "784.0", "789.00"])
def test_neighbouring_symptom_codes_stay_in_other(code):
    """Only 785/786/787/788 are pulled out of the 780-799 symptom chapter."""
    assert icd9_group(code) == "Other"


# --------------------------------------------------------------------------
# V-codes, E-codes, missing values, junk
# --------------------------------------------------------------------------

@pytest.mark.parametrize("code", ["V57", "V58", "V45.81", "V27.0", "E909",
                                  "E878.8", "E888"])
def test_v_and_e_codes_go_to_other(code):
    """Supplementary and external-cause codes have no numeric chapter."""
    assert icd9_group(code) == "Other"


@pytest.mark.parametrize("code", ["v57", "e909"])
def test_lowercase_v_and_e_codes_also_land_in_other(code):
    """Not via the V/E prefix branch -- via float() failing. Same answer."""
    assert icd9_group(code) == "Other"


@pytest.mark.parametrize("code", [np.nan, None, pd.NA, float("nan")])
def test_missing_values_are_their_own_category(code):
    """Missing is a distinct level, not silently merged into Other."""
    assert icd9_group(code) == "Missing"


@pytest.mark.parametrize("code", ["?", "", "   ", "abc", "Missing", "250.83.1",
                                  "--", "N/A"])
def test_unparseable_strings_go_to_other(code):
    assert icd9_group(code) == "Other"


def test_surrounding_whitespace_is_stripped():
    assert icd9_group("  250.83  ") == "Diabetes"
    assert icd9_group("\t428\n") == "Circulatory"
    assert icd9_group(" V57 ") == "Other"


@pytest.mark.parametrize("code,group", [
    (428, "Circulatory"),        # numpy/python ints, not strings
    (250.83, "Diabetes"),
    (np.int64(486), "Respiratory"),
    (np.float64(820.0), "Injury"),
])
def test_non_string_inputs_are_coerced(code, group):
    """pandas hands this function whatever dtype the CSV parser inferred."""
    assert icd9_group(code) == group


def test_it_maps_cleanly_over_a_pandas_series():
    """The production call site is Series.map, including a NaN."""
    s = pd.Series(["250.83", "428", "V57", np.nan, "?"])
    assert s.map(icd9_group).tolist() == [
        "Diabetes", "Circulatory", "Other", "Missing", "Other"]


# --------------------------------------------------------------------------
# Known defects (see the report)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("code,group", [
    ("459.9", "Circulatory"),      # unspecified circulatory system disorder
    ("459.89", "Circulatory"),
    ("519.9", "Respiratory"),
    ("579.9", "Digestive"),
    ("629.9", "Genitourinary"),
    ("739.9", "Musculoskeletal"),
    ("239.9", "Neoplasms"),
    ("999.9", "Injury"),           # complication of medical care, unspecified
    ("999.31", "Injury"),          # infection due to a central venous catheter
])
def test_subdecimal_codes_at_the_top_of_a_range_group_correctly(code, group):
    assert icd9_group(code) == group


@pytest.mark.parametrize("code", ["nan", "NaN", "inf", "-inf"])
def test_float_like_junk_returns_a_label_instead_of_raising(code):
    assert icd9_group(code) in LABELS
