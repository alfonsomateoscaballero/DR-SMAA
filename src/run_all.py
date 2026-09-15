from pathlib import Path
import subprocess, sys
ROOT=Path(__file__).resolve().parents[1]
for script in ['01_synthetic_benchmark.py','02_contamination_analysis.py','03_ai_governance_publicdata.py']:
    print(f'Running {script}...')
    subprocess.run([sys.executable,str(ROOT/'src'/script)],check=True)
print('All analyses completed.')
