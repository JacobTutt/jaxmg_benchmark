from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


def get_gpu_name() -> str:
    """Return the first GPU name from nvidia-smi, sanitized for filesystem paths."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            encoding="utf-8",
        )
        raw_name = output.strip().split("\n")[0]
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Error querying GPU name: {exc}")

    return "_".join(raw_name.split())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_save_dir(
    *,
    script_file: str | Path,
    data_dir: str,
    gpu_name: str,
    dtype_name: str,
    ndev: int,
) -> Path:
    """Build the save directory used by the benchmark scripts."""
    base = Path(script_file).parent
    return base / data_dir / gpu_name / dtype_name / f"ndev_{ndev}"


def npy_path(save_dir: Path, file_stem: str) -> Path:
    return save_dir / f"{file_stem}.npy"


def maybe_load_npy(file_path: Path) -> np.ndarray | None:
    if file_path.exists():
        return np.load(file_path)
    return None


def save_npy(file_path: Path, array: np.ndarray) -> None:
    ensure_dir(file_path.parent)
    np.save(file_path, array)


def _block_one(x: Any) -> None:
    # JAX arrays / pytrees often have .block_until_ready()
    fn = getattr(x, "block_until_ready", None)
    if callable(fn):
        fn()


def block_until_ready_tree(obj: Any) -> None:
    """Best-effort .block_until_ready() for scalars/arrays/tuples/lists."""
    if isinstance(obj, (tuple, list)):
        for item in obj:
            block_until_ready_tree(item)
        return

    _block_one(obj)


def time_runs(
    *,
    run_once: Callable[[], Any],
    n_runs: int,
    print_alloc_msg: str = "Data allocated",
    print_elapsed: bool = True,
) -> np.ndarray:
    """Run `run_once` n_runs times (plus one warmup), timing each after warmup."""
    times: list[float] = []

    for run in range(n_runs + 1):
        if print_alloc_msg:
            print(print_alloc_msg)

        start = time.time()
        out = run_once()
        block_until_ready_tree(out)
        end = time.time()

        if run > 0:  # skip jitted run
            times.append(end - start)
            if print_elapsed:
                print(f"Elapsed time {times[-1]} [s]")

    return np.array(times)
