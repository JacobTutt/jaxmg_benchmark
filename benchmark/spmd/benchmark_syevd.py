import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import subprocess
from pathlib import Path

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["JAXMG_CUSOLVER_UTILS_VERBOSE"] = "0"
import time
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from functools import partial
from jax.sharding import PartitionSpec as P, NamedSharding

from jaxmg import syevd
from jaxmg.utils import random_psd

# Allow importing from repo-root when running as a file path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmark.bench_helpers import (
    get_gpu_name,
    make_save_dir,
    maybe_load_npy,
    npy_path,
    save_npy,
    time_runs,
)

dtype = jnp.float64
devices = jax.devices("gpu")
ndev = len(devices)

n_runs = 5


def main_syevd(N, T_A):

    print(f"Available devices: {ndev}")
    gpu_name = get_gpu_name()
    # PARAMETERS
    save_dir = make_save_dir(
        script_file=__file__,
        data_dir="data_syevd",
        gpu_name=gpu_name,
        dtype_name=jnp.dtype(dtype).name,
        ndev=ndev,
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = f"N_{N}_T_A_{T_A}"
    file_path = npy_path(save_dir, file_name)
    existing = maybe_load_npy(file_path)
    if existing is not None:
        print(f"File: {file_path} already found, skipping...")
        return existing
    # INFO
    print(f"GPU name: {gpu_name}")
    print(f"N={N}, T_A={T_A}, dtype={dtype}")
    from jaxmg import calculate_padding

    padding = calculate_padding(N // ndev, T_A)
    print(f"Padding: {padding}")
    print(jnp.dtype(dtype).itemsize)
    print(f"Memory allocated: {N*N*jnp.dtype(dtype).itemsize/1e9} GB")
    print(f"Memory allocated tile: {N*T_A*jnp.dtype(dtype).itemsize/1e9} GB")

    # MESH
    mesh = jax.make_mesh((ndev,), ("x",))

    # Build A similar to benchmark_potrs: diagonal matrix sharded on rows
    @jax.jit
    def make_A():
        # _A = jax.lax.with_sharding_constraint(
        #     random_psd(N, dtype=dtype, seed=100),
        #     NamedSharding(mesh, P("x", None)),
        # )
        _A= jax.lax.with_sharding_constraint(
            jnp.diag(jnp.arange(N, dtype=dtype) + 1),
            NamedSharding(mesh, P("x", None)),
        )
        return _A

    myfn = jax.jit(
        partial(syevd, mesh=mesh, in_specs=(P("x", None),)), static_argnums=1
    )

    @jax.jit
    def run_once():
        A = make_A()
        ev, V = myfn(A, T_A)
        return ev, V

    times = time_runs(run_once=run_once, n_runs=n_runs, print_alloc_msg="Data allocated")
    # Keep prior behavior: do not save here
    return times


def main_eigh(N):

    print(f"Available devices: {ndev}")
    gpu_name = get_gpu_name()
    # PARAMETERS
    NRHS = 1
    save_dir = make_save_dir(
        script_file=__file__,
        data_dir="data_syevd",
        gpu_name=gpu_name,
        dtype_name=jnp.dtype(dtype).name,
        ndev=ndev,
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_name = f"N_{N}_jax_native"
    file_path = npy_path(save_dir, file_name)
    existing = maybe_load_npy(file_path)
    if existing is not None:
        print(f"File: {file_path} already found, skipping...")
        return existing
    # INFO
    print(f"GPU name: {gpu_name}")
    print(f"N={N}, dtype={dtype}")
    print(jnp.dtype(dtype).itemsize)
    print(f"Memory allocated: {N*N*jnp.dtype(dtype).itemsize/1e9} GB")

    # MESH
    make_diag = jax.jit(lambda: jnp.diag(jnp.arange(N, dtype=dtype) + 1))

    @partial(jax.jit, donate_argnums=0)
    def eigh(A):
        return jnp.linalg.eigh(A)

    @jax.jit
    def run_once():
        A = make_diag()
        return A

    def run_once_with_eigh():
        A = run_once()
        A.block_until_ready()
        return eigh(A)

    times = time_runs(run_once=run_once_with_eigh, n_runs=n_runs, print_alloc_msg="Data allocated")
    save_npy(file_path, times)
    return times


if __name__ == "__main__":

    for N in (
        [2**i for i in range(9, 16)]
        + [2**15 + 2**14]
        + [2**16]
        + [2**16 + 2**15]
        + [2**17]
        + [2**17 + 2**16]
        + [2**18] + [2**18 + 2**16]
        + [2**18 +2**17]
        + [2**19]
    ):
        if ndev == 1:
            main_eigh(N)
        else:
            for T_A in [2**i for i in range(8, 11)]:
                print(f"N={N}, T_A={T_A}")
                main_syevd(N, T_A=T_A)
