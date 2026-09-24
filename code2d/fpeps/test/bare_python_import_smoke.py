"""Dependency/import check only; do not execute adapters or read checkpoints."""
import ast
import json
import os
from pathlib import Path
import numpy
from pip._vendor import tomli

assert os.environ['SLURM_JOB_PARTITION'] == 'preempt'
root = Path(__file__).resolve().parents[1]
files = sorted((root / 'benchmark/independent_bivumps').glob('bare*.py'))
assert files
for path in files:
    ast.parse(path.read_text(), filename=str(path))
assert tomli.loads('passed=true')['passed']
print(json.dumps(dict(passed=True, adapters_parsed=len(files),
                     numpy_version=numpy.__version__, numpy_path=numpy.__file__,
                     tomli_path=tomli.__file__, partition=os.environ['SLURM_JOB_PARTITION'],
                     scope='imports and syntax only; no numerical validation')))
