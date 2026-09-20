"""Thin wrapper around the official validate_submission.py.

Lets you run the exact scorer gate from Python / one command and get a clean
pass/fail boolean back, so it can be wired into every build step.
"""
from __future__ import annotations

import subprocess
import sys

from . import paths


def run(out_dir=paths.OUT_DIR) -> bool:
    """Run the official validator on out_dir; stream its output; return passed."""
    proc = subprocess.run(
        [sys.executable, str(paths.VALIDATOR), str(out_dir)],
        capture_output=True, text=True,
    )
    sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    # validate_submission.py exits 0 on PASS, 1 on FAIL.
    return proc.returncode == 0


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
