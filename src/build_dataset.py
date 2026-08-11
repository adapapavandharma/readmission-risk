"""
Clean the UCI "Diabetes 130-US Hospitals" encounters into a modelling table.

101,766 real inpatient encounters from 130 US hospitals, 1999-2008, with a
genuine 30-day readmission label. Three things about this dataset quietly
break most models built on it, and all three are handled explicitly here.

1. THE SAME PATIENT APPEARS MANY TIMES
   101,766 encounters belong to only 71,518 patients. A random train/test
   split therefore puts the same person on both sides, and the model learns to
   recognise *patients* rather than risk. Every split in this project is
   grouped on patient_nbr. src/model.py quantifies exactly how much AUC the
   naive split invents.

2. SOME PATIENTS CANNOT BE READMITTED
   Discharge dispositions 11, 19, 20 and 21 are deaths; 13 and 14 are hospice.
   Those encounters are guaranteed negatives for reasons that have nothing to
   do with readmission risk, so leaving them in lets the model score easy
   points by learning to detect dying patients. They are removed, which is
   standard for this dataset and is what Strack et al. (2014) do.

3. THE TARGET HAS THREE LEVELS, NOT TWO
   `readmitted` is '<30', '>30' or 'NO'. The clinically and financially
   meaningful event (CMS penalises 30-day readmissions) is '<30'. '>30' is
   collapsed into the negative class rather than dropped: those patients came
   back, just not inside the window, and discarding them would inflate
   apparent performance by removing the hardest negatives.

Reference: Strack et al., "Impact of HbA1c Measurement on Hospital Readmission
Rates", BioMed Research International, 2014.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "external" / "dataset_diabetes" / "diabetic_data.csv"
INTERIM = ROOT / "data" / "interim"
OUT = INTERIM / "encounters.parquet"

DECEASED_DISPOSITIONS = {11, 19, 20, 21}
HOSPICE_DISPOSITIONS = {13, 14}

# Columns dropped outright, with the reason.
DROP_COLUMNS = {
    "weight": "96.9% missing",
    "encounter_id": "identifier, not a feature",
    # These four drugs are constant across all 101,766 rows — zero variance.
    "examide": "single-valued",
    "citoglipton": "single-valued",
    "metformin-rosiglitazone": "near-constant",
    "glimepiride-pioglitazone": "near-constant",
}

# 24 diabetes drug columns: 'No' / 'Steady' / 'Up' / 'Down'
DRUG_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "metformin-pioglitazone",
]


def icd9_group(code: object) -> str:
    """Collapse an ICD-9 code into the clinical categories used by Strack et al.

    V- and E-codes (supplementary classification and external causes) have no
    numeric range and go to Other.
    """
    if pd.isna(code):
        return "Missing"
    s = str(code).strip()
    if s.startswith(("V", "E")):
        return "Other"
    try:
        v = float(s)
    except ValueError:
        return "Other"
    # 'nan' and 'inf' parse as floats but are not codes; int() on either raises.
    if not math.isfinite(v):
        return "Other"

    # Compare on the CHAPTER, i.e. the integer part, not the decimal code.
    # ICD-9 chapters are stated as whole-number ranges but the codes carry
    # subdivisions: 459.9 is circulatory, yet `390 <= 459.9 <= 459` is False.
    # Testing the float sent every x.9 code at the top of a chapter to Other —
    # 459.9, 519.9, 579.9, 629.9, 739.9, 239.9 and the entire 999.x block
    # (complications of medical care, which is exactly the kind of diagnosis a
    # readmission model should be able to see).
    chapter = int(v)

    if chapter == 250:
        return "Diabetes"
    if 390 <= chapter <= 459 or chapter == 785:
        return "Circulatory"
    if 460 <= chapter <= 519 or chapter == 786:
        return "Respiratory"
    if 520 <= chapter <= 579 or chapter == 787:
        return "Digestive"
    if 800 <= chapter <= 999:
        return "Injury"
    if 710 <= chapter <= 739:
        return "Musculoskeletal"
    if 580 <= chapter <= 629 or chapter == 788:
        return "Genitourinary"
    if 140 <= chapter <= 239:
        return "Neoplasms"
    return "Other"


AGE_MIDPOINT = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45,
    "[50-60)": 55, "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95,
}


def main() -> int:
    if not RAW.exists():
        print(f"missing {RAW}\nrun: python src/download.py", file=sys.stderr)
        return 1

    df = pd.read_csv(RAW, low_memory=False)
    start_rows, start_patients = len(df), df["patient_nbr"].nunique()
    print(f"  loaded                       {start_rows:>8,} encounters, "
          f"{start_patients:,} patients")

    # '?' is this dataset's missing marker
    df = df.replace("?", np.nan)

    # --- trap 2: remove encounters where readmission is impossible ---------
    dead = df["discharge_disposition_id"].isin(DECEASED_DISPOSITIONS)
    hospice = df["discharge_disposition_id"].isin(HOSPICE_DISPOSITIONS)
    print(f"  dropped deceased             {dead.sum():>8,}")
    print(f"  dropped hospice              {hospice.sum():>8,}")
    df = df[~(dead | hospice)].copy()

    # --- trap 3: binary target --------------------------------------------
    df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)
    rate = df["readmitted_30d"].mean()
    print(f"  remaining                    {len(df):>8,} encounters, "
          f"{df['patient_nbr'].nunique():,} patients")
    print(f"  30-day readmission rate      {rate:>8.2%}")

    # --- features ----------------------------------------------------------
    df["age_midpoint"] = df["age"].map(AGE_MIDPOINT)

    for col in ("diag_1", "diag_2", "diag_3"):
        df[f"{col}_group"] = df[col].map(icd9_group)

    # Prior utilisation is the strongest signal in this dataset and the most
    # clinically sensible: someone in hospital three times last year is the
    # person to worry about.
    df["prior_visits_total"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )
    df["had_prior_inpatient"] = (df["number_inpatient"] > 0).astype(int)
    df["had_prior_emergency"] = (df["number_emergency"] > 0).astype(int)

    # How actively the diabetes regimen is being manipulated this stay.
    present = [c for c in DRUG_COLUMNS if c in df.columns]
    df["n_drugs_changed"] = df[present].isin(["Up", "Down"]).sum(axis=1)
    df["n_drugs_active"] = df[present].isin(["Up", "Down", "Steady"]).sum(axis=1)

    df["a1c_measured"] = (df["A1Cresult"].notna()).astype(int)
    df["glucose_measured"] = (df["max_glu_serum"].notna()).astype(int)

    # Explicit "Missing" category rather than imputation: for medical_specialty
    # and payer_code, whether the field was recorded at all is itself
    # informative about the admitting pathway.
    for col in ("race", "medical_specialty", "payer_code", "A1Cresult", "max_glu_serum"):
        df[col] = df[col].fillna("Missing")

    df["gender"] = df["gender"].replace("Unknown/Invalid", "Missing")

    drop = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=drop)
    print(f"  dropped columns              {', '.join(drop)}")

    INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\n  wrote {OUT.relative_to(ROOT)}  ({len(df):,} x {len(df.columns)})")

    enc_per_patient = len(df) / df["patient_nbr"].nunique()
    print(f"  encounters per patient       {enc_per_patient:>8.2f}  "
          f"(why splits must be grouped)")
    multi = df["patient_nbr"].value_counts()
    print(f"  patients with >1 encounter   {(multi > 1).sum():>8,} "
          f"({(multi > 1).mean():.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
