"""The self-test the release build runs on the packaged exe, run here on the source.

`BOXCOORDINATES_SELFTEST=1 python BoxCoordinates.py` builds SAM with random weights (no checkpoint,
no network), runs the image encoder once, draws a matplotlib figure and starts Qt offscreen, then
exits with 0. It runs in a separate process because it uses the real dependencies, not the fakes
from conftest.py.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_self_test_exits_cleanly():
    pytest.importorskip("segment_anything")
    env = {**os.environ, "BOXCOORDINATES_SELFTEST": "1", "QT_QPA_PLATFORM": "offscreen"}
    result = subprocess.run([sys.executable, str(ROOT / "BoxCoordinates.py")], env=env, cwd=ROOT,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
