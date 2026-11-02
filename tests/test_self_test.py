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


def test_self_test_exits_cleanly(tmp_path):
    pytest.importorskip("segment_anything")
    log = tmp_path / "selftest.log"
    env = {**os.environ, "BOXCOORDINATES_SELFTEST": "1", "BOXCOORDINATES_SELFTEST_LOG": str(log),
           "QT_QPA_PLATFORM": "offscreen"}
    result = subprocess.run([sys.executable, str(ROOT / "BoxCoordinates.py")], env=env, cwd=ROOT,
                            capture_output=True, text=True, timeout=300)
    written = log.read_text(encoding="utf-8") if log.exists() else ""
    assert result.returncode == 0, written + result.stderr[-2000:]
    assert "self-test passed" in written


def test_self_test_logs_an_error_instead_of_hanging(tmp_path):
    # An error must end the process with code 3 and a traceback in the log; in the windowed exe it
    # would otherwise open a dialog and the build would wait for it.
    log = tmp_path / "selftest.log"
    script = ("import os, runpy, sys; sys.argv = ['BoxCoordinates.py']; "
              "import builtins; real = builtins.__import__; "
              "builtins.__import__ = lambda n, *a, **k: (_ for _ in ()).throw(ImportError('no ' + n)) "
              "if n == 'segment_anything' else real(n, *a, **k); "
              f"runpy.run_path({str(ROOT / 'BoxCoordinates.py')!r}, run_name='__main__')")
    env = {**os.environ, "BOXCOORDINATES_SELFTEST": "1", "BOXCOORDINATES_SELFTEST_LOG": str(log),
           "QT_QPA_PLATFORM": "offscreen"}
    result = subprocess.run([sys.executable, "-c", script], env=env, cwd=ROOT,
                            capture_output=True, text=True, timeout=300)
    written = log.read_text(encoding="utf-8") if log.exists() else ""
    assert result.returncode == 3, written + result.stderr[-2000:]
    assert "ImportError: no segment_anything" in written
