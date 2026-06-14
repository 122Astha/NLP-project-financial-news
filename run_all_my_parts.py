"""
Run all prepared parts: Task 1, Task 2, and Task 4.

Before running:
1. Put Sentences_50Agree.txt into data/
2. Run: pip install -r requirements.txt
3. Run: python run_all_my_parts.py
"""
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
scripts = [ROOT/'src'/'task1_eda.py', ROOT/'src'/'task2_preprocessing.py', ROOT/'src'/'task4_similarity.py']
for script in scripts:
    if not script.exists():
        print(f'Skipping missing script: {script.name}')
        continue
    print('
' + '='*80)
    print(f'Running {script.name}')
    print('='*80)
    result = subprocess.run([sys.executable, str(script)], cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(f'{script.name} failed. Please check the error above.')
print('
All prepared parts finished successfully.')
