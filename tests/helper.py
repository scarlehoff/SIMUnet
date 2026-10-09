"""Process helpers for the SIMUnet regression tests."""

import os
from pathlib import Path
import subprocess as sp
import sys

ROOT = Path(__file__).resolve().parents[1]


def run_simunet(runcard, *, cwd, timeout=1200):
    """Run a runcard in an isolated directory and save its output to run.log."""
    cwd.mkdir(parents=True, exist_ok=True)
    (cwd / "tests").symlink_to(ROOT / "tests", target_is_directory=True)
    env = dict(os.environ, MPLBACKEND="Agg", BROWSER="true")
    result = sp.run(
        [sys.executable, "-m", "simunet.app", str(runcard)],
        cwd=cwd,
        env=env,
        text=True,
        stdout=sp.PIPE,
        stderr=sp.STDOUT,
        timeout=timeout,
    )
    (cwd / "run.log").write_text(result.stdout)
    result.check_returncode()
    return result
