"""Run the whole pipeline: download -> clean -> model -> evaluate -> registry."""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("fetch the UCI dataset", "src/download.py"),
    ("clean encounters and handle the three traps", "src/build_dataset.py"),
    ("leakage experiment, grouped CV, hold-out fit", "src/model.py"),
    ("lift, calibration, decision curve", "src/evaluate.py"),
    ("tiered risk board", "src/registry.py"),
]


def main() -> int:
    for i, (label, script) in enumerate(STEPS, start=1):
        print(f"\n{'-' * 72}\n[{i}/{len(STEPS)}] {label}\n{'-' * 72}")
        r = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT)
        if r.returncode != 0:
            print(f"\nfailed at step {i}: {script}", file=sys.stderr)
            return r.returncode
    print(f"\n{'-' * 72}\ndone — see outputs/\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
