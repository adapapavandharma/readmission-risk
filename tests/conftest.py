"""Shared test configuration and hermetic fixtures.

The modules under ``src/`` import each other by bare name (``import model``),
so ``src/`` has to be on ``sys.path`` before any test module is collected.

Everything in this suite is hermetic: no network, no Synthea, and nothing read
from ``data/external/`` or ``data/interim/`` (both gitignored and absent on
CI). Every fixture builds its data in code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


# --------------------------------------------------------------------------
# A miniature version of the raw UCI file, with one row per situation that
# build_dataset.main() is supposed to handle.
# --------------------------------------------------------------------------

# row index -> what it exercises
#  0  kept    patient 1, readmitted '<30'  -> label 1
#  1  kept    patient 1 again (same patient twice), readmitted '>30' -> label 0
#  2  DROP    discharge 11 = expired
#  3  DROP    discharge 13 = hospice / home
#  4  DROP    discharge 19 = expired in a medical facility
#  5  DROP    discharge 14 = hospice / medical facility
#  6  kept    patient 6, readmitted 'NO' -> label 0
#  7  kept    patient 7, readmitted '<30' -> label 1
RAW_ROWS = {
    "encounter_id":              [101, 102, 103, 104, 105, 106, 107, 108],
    "patient_nbr":               [1, 1, 2, 3, 4, 5, 6, 7],
    "race":                      ["Caucasian", "?", "Other", "Asian", "Hispanic",
                                  "Other", "AfricanAmerican", "?"],
    "gender":                    ["Female", "Male", "Female", "Male", "Female",
                                  "Male", "Unknown/Invalid", "Female"],
    "age":                       ["[70-80)", "[70-80)", "[60-70)", "[50-60)",
                                  "[80-90)", "[40-50)", "[0-10)", "[90-100)"],
    "weight":                    ["?"] * 8,
    "admission_type_id":         [1, 1, 1, 2, 3, 1, 2, 1],
    "discharge_disposition_id":  [1, 3, 11, 13, 19, 14, 1, 6],
    "admission_source_id":       [7, 7, 1, 7, 7, 7, 1, 7],
    "time_in_hospital":          [3, 5, 2, 9, 1, 4, 2, 8],
    "payer_code":                ["MC", "HM", "?", "MC", "?", "SP", "?", "MC"],
    "medical_specialty":         ["InternalMedicine", "?", "Surgery", "?",
                                  "Cardiology", "?", "Family/GeneralPractice",
                                  "Nephrology"],
    "num_lab_procedures":        [41, 59, 11, 70, 33, 22, 18, 64],
    "num_procedures":            [0, 1, 0, 3, 0, 1, 0, 2],
    "num_medications":           [12, 18, 7, 25, 9, 11, 6, 21],
    "number_outpatient":         [2, 0, 0, 1, 0, 0, 0, 1],
    "number_emergency":          [0, 1, 0, 0, 2, 0, 0, 2],
    "number_inpatient":          [3, 0, 1, 2, 0, 0, 0, 4],
    "diag_1":                    ["250.83", "486", "410.71", "780.6", "584.9",
                                  "V57", "787.91", "599.0"],
    "diag_2":                    ["428", "?", "250.02", "276.8", "428.0",
                                  "401.9", "820", "185"],
    "diag_3":                    ["V57", "250", "401.9", "?", "250.6",
                                  "272.4", "715", "E909"],
    "number_diagnoses":          [9, 6, 5, 9, 7, 4, 3, 9],
    "max_glu_serum":             ["None", ">200", "None", "None", "None",
                                  "Norm", "None", "Norm"],
    "A1Cresult":                 [">8", "None", "None", "Norm", "None",
                                  "None", "None", "Norm"],
    "metformin":                 ["Up", "Down", "No", "No", "Steady", "No",
                                  "No", "No"],
    "insulin":                   ["Steady", "Down", "No", "Up", "No", "Steady",
                                  "No", "Up"],
    "glipizide":                 ["No", "Steady", "No", "No", "No", "No",
                                  "No", "Down"],
    "examide":                   ["No"] * 8,
    "citoglipton":               ["No"] * 8,
    "metformin-rosiglitazone":   ["No"] * 8,
    "glimepiride-pioglitazone":  ["No"] * 8,
    "change":                    ["Ch", "Ch", "No", "Ch", "No", "No", "No", "Ch"],
    "diabetesMed":               ["Yes", "Yes", "No", "Yes", "Yes", "Yes",
                                  "No", "Yes"],
    "readmitted":                ["<30", ">30", "NO", "<30", "<30", "NO",
                                  "NO", "<30"],
}

@pytest.fixture()
def raw_csv(tmp_path: Path) -> Path:
    """Write the miniature raw file in the layout build_dataset expects."""
    target = tmp_path / "external" / "dataset_diabetes" / "diabetic_data.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(RAW_ROWS).to_csv(target, index=False)
    return target


# --------------------------------------------------------------------------
# Hold-out prediction frames, shaped exactly like the parquet model.fit_final
# writes, so evaluate.py and registry.py can be driven without a real fit.
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def holdout_frame():
    """Factory for a synthetic hold-out prediction frame.

    ``inflate`` shifts the reported gbm probability by that many logits, which
    reproduces the class_weight='balanced' failure mode: the ranking is
    untouched, the probability scale is destroyed.
    """
    def _make(n_patients: int = 400, per_patient: int = 5, seed: int = 7,
              inflate: float = 0.0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        n = n_patients * per_patient
        patient = np.repeat(np.arange(1000, 1000 + n_patients), per_patient)

        prior_inpatient = rng.poisson(0.6, n).astype(float)
        lin = -2.4 + 0.55 * prior_inpatient + rng.normal(0, 0.8, n)
        p_true = sigmoid(lin)
        y = (rng.random(n) < p_true).astype(int)

        return pd.DataFrame({
            "patient_nbr": patient,
            "y_true": y,
            "p_logistic": sigmoid(lin + rng.normal(0, 0.25, n)),
            "p_gbm": sigmoid(lin + inflate),
            "rule_prior_inpatient": prior_inpatient,
            "rule_prior_visits": prior_inpatient + rng.poisson(0.4, n),
        })
    return _make


# --------------------------------------------------------------------------
# Modelling frames for src/model.py
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def encounter_frame():
    """Factory for a frame with every column src/model.py expects."""
    def _make(n: int = 600, seed: int = 3, n_patients: int = 150) -> pd.DataFrame:
        import model as md

        rng = np.random.default_rng(seed)
        data = {c: rng.integers(0, 8, n).astype(float) for c in md.NUMERIC}
        data.update({c: rng.choice(["a", "b", "c"], n) for c in md.CATEGORICAL})
        df = pd.DataFrame(data)
        df["patient_nbr"] = rng.integers(0, n_patients, n)
        df["prior_visits_total"] = (df["number_outpatient"]
                                    + df["number_emergency"]
                                    + df["number_inpatient"])
        lin = -1.8 + 0.4 * df["number_inpatient"].to_numpy()
        df[md.TARGET] = (rng.random(n) < sigmoid(lin)).astype(int)
        return df
    return _make


@pytest.fixture(scope="session")
def patient_fingerprint_frame():
    """A frame whose *only* signal is patient identity.

    Each of the 16 patients owns one NUMERIC column, set to 1 for that
    patient's encounters and 0 for everyone else, and the label is constant
    within a patient. Nothing here generalises to a new patient, so:

      * a random (encounter-level) split sees every patient in training and
        can score the hold-out perfectly  -> AUC 1.0
      * a patient-grouped split never sees the test patients' columns, those
        coefficients stay at zero, every test row gets the same score
        -> AUC 0.5 exactly

    That gap is the leakage src/model.py exists to measure.
    """
    def _make(per_patient: int = 20) -> pd.DataFrame:
        import model as md

        rows = []
        for pid, column in enumerate(md.NUMERIC):
            label = int(pid % 2 == 0)
            for _ in range(per_patient):
                row = {c: 0.0 for c in md.NUMERIC}
                row[column] = 1.0
                row.update({c: "x" for c in md.CATEGORICAL})
                row["patient_nbr"] = 1000 + pid
                row[md.TARGET] = label
                rows.append(row)
        return pd.DataFrame(rows)
    return _make
