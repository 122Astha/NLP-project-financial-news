"""Run Task 1 and Task 2 scripts."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

scripts = [
    ROOT / "src" / "task1_eda.py",
    ROOT / "src" / "task2_preprocessing.py",
]

for script in scripts:
    print(f"\nRunning {script.name} ...")
    result = subprocess.run([sys.executable, str(script)], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"Script failed: {script.name}")

print("\nTask 1 and Task 2 completed successfully.")
