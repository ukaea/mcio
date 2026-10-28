"""Pylint-style score out of 10 from ruff findings.

score = 10 - 10 * (5 * errors + other findings) / statements, clamped to [0, 10]

"errors" are the pyflakes (F) and syntax (E9) findings, i.e. likely bugs such
as undefined names; everything else counts once.

usage: python scripts/ruff_score.py [paths...]   (default: mcio)

Used by the lint workflow to publish the score badge; ruff must be on PATH.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

ERROR_PREFIXES = ("F", "E9")

paths = sys.argv[1:] or ["mcio"]

result = subprocess.run(
    ["ruff", "check", "--output-format=json", "--exit-zero", *paths],
    capture_output=True, text=True, check=True,
)
findings = json.loads(result.stdout)
errors = sum(1 for f in findings if (f["code"] or "").startswith(ERROR_PREFIXES))
others = len(findings) - errors

statements = 0
for path in paths:
    files = [Path(path)] if path.endswith(".py") else Path(path).rglob("*.py")
    for file in files:
        tree = ast.parse(file.read_text(encoding="utf-8"))
        statements += sum(isinstance(node, ast.stmt) for node in ast.walk(tree))

score = max(0.0, 10.0 - 10.0 * (5 * errors + others) / max(statements, 1))
print(f"{len(findings)} findings ({errors} errors) in {statements} statements")
print(f"Your code has been rated at {score:.2f}/10")
