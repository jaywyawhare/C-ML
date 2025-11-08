import subprocess
from pathlib import Path
import numpy as np
import pytest


def _run_opcheck() -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    exe = repo_root / "build" / "examples" / "opcheck"
    if not exe.exists():
        raise FileNotFoundError(
            f"Missing {exe}. Run 'make -j' in project root before pytest."
        )
    p = subprocess.run(
        [str(exe)], capture_output=True, text=True, check=True, cwd=str(repo_root)
    )
    results = {}
    for line in p.stdout.strip().splitlines():
        parts = line.strip().split(",")
        key = parts[0]
        vals = parts[1:]
        if key in {"MSE", "MAE", "BCE", "SUM", "MEAN"}:
            results[key] = float(vals[0])
        else:
            results[key] = np.array([float(v) for v in vals], dtype=np.float32)
    return results


@pytest.fixture(scope="session")
def opout():
    return _run_opcheck()
