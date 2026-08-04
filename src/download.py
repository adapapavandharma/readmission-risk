"""Fetch the UCI Diabetes 130-US Hospitals dataset (not committed to git)."""
from __future__ import annotations
import shutil, sys, urllib.request, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"
CSV = EXTERNAL / "dataset_diabetes" / "diabetic_data.csv"
EXPECTED_ROWS = 101_766


def main() -> int:
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    zip_path = EXTERNAL / "dataset_diabetes.zip"
    if not zip_path.exists():
        print("  fetching dataset_diabetes.zip ...", end="", flush=True)
        with urllib.request.urlopen(URL, timeout=240) as r, zip_path.open("wb") as f:
            shutil.copyfileobj(r, f)
        print(f" {zip_path.stat().st_size:,} bytes")
    else:
        print("  cached   dataset_diabetes.zip")

    if not CSV.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(EXTERNAL)

    rows = sum(1 for _ in CSV.open()) - 1
    ok = rows == EXPECTED_ROWS
    print(f"  {CSV.relative_to(ROOT)}  {rows:,} rows  "
          f"{'OK' if ok else f'UNEXPECTED (wanted {EXPECTED_ROWS:,})'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
