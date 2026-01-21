import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import subprocess
from pathlib import Path

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["JAXMG_CUSOLVER_UTILS_VERBOSE"] = "0"
if len(sys.argv) > 1:
    coord_addr = sys.argv[1]
    proc_id = int(sys.argv[2])
    num_procs = int(sys.argv[3])
    import jax

    # Initialize the GPU machines.
    jax.distributed.initialize(
        coordinator_address=coord_addr,
        num_processes=num_procs,
        process_id=proc_id,
        local_device_ids=proc_id,
        coordinator_bind_address=coord_addr,
    )
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import jax
print("process id =", jax.process_index())
print("global devices =", jax.devices())
print("local devices =", jax.local_devices())
print("visible devices", os.environ["CUDA_VISIBLE_DEVICES"])

import time
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from functools import partial
from jax.sharding import PartitionSpec as P, NamedSharding
from jax.experimental.multihost_utils import process_allgather

from jaxmg import potri
from jaxmg import calculate_padding

dtype = jnp.float64
devices = jax.devices("gpu")
ndev = len(devices)

n_runs = 5


def main_potri(N, T_A):
    pid = jax.process_index()
    print(f"Available devices: {ndev}")
    gpu_name = ""
    gpu_name = ""
    if pid == 0:
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                encoding="utf-8",
            )
            gpu_name = output.strip().split("\n")[0]
        except Exception as e:
            raise ValueError(f"Error querying GPU name: {e}")
    gpu_name = "_".join(gpu_name.split(" "))
    # PATHS
    save_path = f"{Path(__file__).parent}/data_potri/{gpu_name}/{jnp.dtype(dtype).name}/ndev_{ndev}/"
    if not os.path.exists(save_path) and pid == 0:
        os.makedirs(save_path)
    file_name = f"N_{N}_T_A_{T_A}"
    returning = False
    if os.path.exists(f"{save_path}{file_name}.npy"):
        print(f"File: {save_path}{file_name}.npy already found, skipping...")
        returning = True

    gathered_return = process_allgather(jnp.array(returning, dtype=jnp.bool))
    print(f"Are we returning? {gathered_return}")
    if bool(jnp.any(gathered_return)):
        print(f"pid {pid} is returning")
        return

    # INFO
    print(f"GPU name: {gpu_name}")
    print(f"N={N}, T_A={T_A}, dtype={dtype}")
    padding = calculate_padding(N // max(1, ndev), T_A)
    print(f"Padding: {padding}")
    print(jnp.dtype(dtype).itemsize)
    print(f"Memory allocated: {N*N*jnp.dtype(dtype).itemsize/1e9} GB")
    print(f"Memory allocated tile: {N*T_A*jnp.dtype(dtype).itemsize/1e9} GB")

    # MESH
    mesh = jax.make_mesh((ndev,), ("x",))

    # Build A: diagonal SPD, sharded on rows
    @jax.jit
    def make_A():
        return jax.lax.with_sharding_constraint(
            jnp.diag(jnp.arange(N, dtype=dtype) + 1),
            NamedSharding(mesh, P("x", None)),
        )

    myfn = jax.jit(
        partial(potri, mesh=mesh, in_specs=(P("x", None),), pad=True),
        static_argnums=1,
    )

    @jax.jit
    def run_once():
        A = make_A()
        out = myfn(A, T_A)
        return out

    times = []
    for run in range(n_runs + 1):
        print("Data allocated")

        start = time.time()
        out = run_once()
        out.block_until_ready()
        end = time.time()
        if run > 0:  # skip jitted run
            times.append(end - start)
            print(f"Elapsed time {times[-1]} [s]")

    if pid == 0:
        np.save(f"{save_path}/{file_name}", np.array(times))
    return np.array(times)


def main_inverse(N):

    print(f"Available devices: {ndev}")
    gpu_name = ""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            encoding="utf-8",
        )
        gpu_name = output.strip().split("\n")[0]
    except Exception as e:
        raise ValueError(f"Error querying GPU name: {e}")
    gpu_name = "_".join(gpu_name.split(" "))
    # PATHS
    save_path = f"{Path(__file__).parent}/data_potri/{gpu_name}/{jnp.dtype(dtype).name}/ndev_{ndev}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_name = f"N_{N}_jax_native"
    if os.path.exists(f"{save_path}{file_name}.npy"):
        print(f"File: {save_path}{file_name}.npy already found, skipping...")
        return np.load(f"{save_path}{file_name}.npy")

    # INFO
    print(f"GPU name: {gpu_name}")
    print(f"N={N}, dtype={dtype}")
    print(jnp.dtype(dtype).itemsize)
    print(f"Memory allocated: {N*N*jnp.dtype(dtype).itemsize/1e9} GB")

    @partial(jax.jit, donate_argnums=0)
    def inverse(A):
        return jnp.linalg.inv(A)

    @jax.jit
    def run_once():
        A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
        out = inverse(A)
        return out

    times = []
    for run in range(n_runs + 1):
        print("Data allocated")

        start = time.time()
        out = run_once()
        out.block_until_ready()
        end = time.time()
        if run > 0:  # skip jitted run
            times.append(end - start)
            print(f"Elapsed time {times[-1]} [s]")

    np.save(f"{save_path}/{file_name}", np.array(times))
    return np.array(times)


if __name__ == "__main__":

    for N in (
        [2**i for i in range(9, 16)]
        + [2**15 + 2**14]
        + [2**16]
        + [2**16 + 2**15]
        + [2**17 + 2**16]
        + [2**18]
    ):
        if ndev == 1:
            main_inverse(N)
        else:
            for T_A in [2**i for i in range(8, 13)]:
                print(f"N={N}, T_A={T_A}")
                main_potri(N, T_A=T_A)
