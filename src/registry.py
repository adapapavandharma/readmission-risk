"""
Turn model scores into the artefact a care-management team actually uses:
a daily risk board, tiered, sized to the outreach capacity that exists.

A probability column is not a deliverable. What a team needs is "here are the
41 people to call today, and here is what we expect that to be worth", with the
tier boundaries derived from capacity rather than from round numbers.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "data" / "interim" / "holdout_predictions.parquet"
DB = ROOT / "data" / "registry.db"
OUT = ROOT / "outputs"

# What one care manager can realistically work in a day, and the daily
# discharge volume of a mid-sized hospital. Tier cut-points fall out of these
# rather than being picked to look tidy.
DAILY_DISCHARGES = 120
OUTREACH_CAPACITY_PER_DAY = 24          # 20% of discharges

SCHEMA = """
DROP TABLE IF EXISTS risk_score;
CREATE TABLE risk_score (
    -- One row per ENCOUNTER, not per patient. A patient can be scored several
    -- times, and two of their encounters can land on the same predicted risk:
    -- a tree ensemble's output is piecewise constant, and one patient's
    -- encounters share most of their features, so collisions are likelier than
    -- chance. Keying on (patient_nbr, predicted_risk) therefore aborts the
    -- whole load with an IntegrityError as soon as that happens.
    score_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_nbr      INTEGER NOT NULL,
    predicted_risk   REAL    NOT NULL,
    risk_tier        TEXT    NOT NULL,
    risk_percentile  REAL    NOT NULL,
    actual_readmit   INTEGER            -- present only in back-test data
);
CREATE INDEX idx_risk_tier    ON risk_score(risk_tier);
CREATE INDEX idx_risk_patient ON risk_score(patient_nbr);
"""

TIER_QUERY = """
SELECT
    risk_tier,
    COUNT(*)                                                  AS patients,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)        AS pct_of_list,
    ROUND(AVG(predicted_risk) * 100, 1)                       AS mean_predicted_risk_pct,
    ROUND(AVG(actual_readmit) * 100, 1)                       AS observed_readmit_pct,
    SUM(actual_readmit)                                       AS readmissions,
    ROUND(100.0 * SUM(actual_readmit)
          / SUM(SUM(actual_readmit)) OVER (), 1)              AS pct_of_all_readmissions
FROM risk_score
GROUP BY risk_tier
ORDER BY mean_predicted_risk_pct DESC;
"""


def main() -> int:
    if not PRED.exists():
        print(f"missing {PRED}\nrun: python src/model.py", file=sys.stderr)
        return 1

    df = pd.read_parquet(PRED).rename(columns={"p_gbm": "predicted_risk"})
    df = df[["patient_nbr", "predicted_risk", "y_true"]].copy()

    # Percentile within the scored population, highest risk = 100th.
    df["risk_percentile"] = df["predicted_risk"].rank(pct=True) * 100

    capacity_share = OUTREACH_CAPACITY_PER_DAY / DAILY_DISCHARGES
    high_cut = 100 * (1 - capacity_share)                 # top 20%
    medium_cut = 100 * (1 - 2.5 * capacity_share)         # next 30%

    df["risk_tier"] = np.select(
        [df["risk_percentile"] >= high_cut, df["risk_percentile"] >= medium_cut],
        ["High - call before discharge", "Medium - post-discharge follow-up"],
        default="Low - standard discharge",
    )
    df = df.rename(columns={"y_true": "actual_readmit"})

    DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)
    try:
        conn.executescript(SCHEMA)
        df[["patient_nbr", "predicted_risk", "risk_tier",
            "risk_percentile", "actual_readmit"]].to_sql(
            "risk_score", conn, if_exists="append", index=False)
        conn.commit()
        board = pd.read_sql_query(TIER_QUERY, conn)
    finally:
        conn.close()

    print(f"\n{'='*76}\nRISK BOARD — back-tested on {len(df):,} held-out encounters\n{'='*76}")
    print(f"  assumes {DAILY_DISCHARGES} discharges/day and capacity to contact "
          f"{OUTREACH_CAPACITY_PER_DAY}\n")
    print(board.to_string(index=False))

    high = board[board.risk_tier.str.startswith("High")].iloc[0]
    scale = DAILY_DISCHARGES / len(df)
    print(f"\n  Working the High tier means contacting "
          f"{OUTREACH_CAPACITY_PER_DAY} patients a day and reaching "
          f"{high.pct_of_all_readmissions:.0f}% of everyone who will come back.")
    print(f"  Observed readmission rate in that tier is {high.observed_readmit_pct:.1f}% "
          f"against {df['actual_readmit'].mean()*100:.1f}% overall — "
          f"{high.observed_readmit_pct / (df['actual_readmit'].mean()*100):.2f}x enrichment.")
    print(f"\n  Note this is enrichment, not prevention. The model finds who is at "
          f"risk;\n  whether calling them changes the outcome is a question only a "
          f"trial can answer.")

    OUT.mkdir(parents=True, exist_ok=True)
    board.to_csv(OUT / "risk_board.csv", index=False)
    print(f"\n  wrote {(OUT / 'risk_board.csv').relative_to(ROOT)} and "
          f"{DB.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
