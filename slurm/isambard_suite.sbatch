#!/usr/bin/env bash
set -euo pipefail

# submit.py supplies the allocation, solver, dtype, grid, and output root.
: "${JAXMG_BENCHMARK_VENV:?Set JAXMG_BENCHMARK_VENV to the JAXMg virtual environment}"
: "${JAXMG_BENCHMARK_CONFIG:?Missing benchmark configuration}"
: "${JAXMG_BENCHMARK_ROUTINE:?Missing routine}"
: "${JAXMG_BENCHMARK_DTYPE:?Missing dtype}"
: "${JAXMG_BENCHMARK_GRID:?Missing process grid}"
: "${JAXMG_BENCHMARK_OUTPUT_ROOT:?Missing output directory}"

module load cray-python/3.11.7 cuda/12.6 gcc-native/14.2
source "${JAXMG_BENCHMARK_VENV}/bin/activate"

# Some Isambard nodes do not provision Slurm's default /local/user/<uid>
# directory. JAX invokes ptxas while compiling the input factory, so force all
# CUDA compiler temporary files to the standard writable node-local location.
export TMPDIR=/tmp
export TMP=/tmp
export TEMP=/tmp

# Use an editable checkout when requested; otherwise use the installed wheel.
if [[ -n "${JAXMG_SOURCE_ROOT:-}" ]]; then
    export PYTHONPATH="${JAXMG_SOURCE_ROOT}/src:${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
else
    export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
fi

# Add CUDA wheel libraries and Isambard's NCCL/UCC libraries to the loader path.
NVIDIA_LIBRARY_PATH="$(${JAXMG_BENCHMARK_VENV}/bin/python - <<'PY'
from pathlib import Path
import site

paths = []
for root in site.getsitepackages():
    nvidia = Path(root) / "nvidia"
    if nvidia.is_dir():
        paths.extend(str(path) for path in sorted(nvidia.glob("*/lib")))
print(":".join(paths))
PY
)"
export LD_LIBRARY_PATH="${NVIDIA_LIBRARY_PATH}:/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/comm_libs/12.6/hpcx/hpcx-2.20/ucc/lib:/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/comm_libs/12.6/nccl/lib:${LD_LIBRARY_PATH:-}"

export XLA_PYTHON_CLIENT_ALLOCATOR=vmm
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.99
# Keep the NCCL setting that has proved reliable on Isambard. It can still be
# overridden by the submission environment when testing another runtime.
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export OMP_NUM_THREADS=1

extra_args=()
if [[ -n "${JAXMG_BENCHMARK_TILE_SIZE:-}" ]]; then
    extra_args+=(--tile-size "${JAXMG_BENCHMARK_TILE_SIZE}")
fi
if [[ -n "${JAXMG_BENCHMARK_MATRIX_SIZES:-}" ]]; then
    # Explicit sizes are used for a small memory-limit sweep. Each still gets
    # its own fresh srun process group.
    read -r -a matrix_sizes <<< "${JAXMG_BENCHMARK_MATRIX_SIZES}"
    for matrix_size in "${matrix_sizes[@]}"; do
        extra_args+=(--matrix-size "${matrix_size}")
    done
elif [[ -n "${JAXMG_BENCHMARK_MATRIX_SIZE:-}" ]]; then
    extra_args+=(--matrix-size "${JAXMG_BENCHMARK_MATRIX_SIZE}")
fi

python -m benchmark.cusolvermp.run_suite \
    --config "${JAXMG_BENCHMARK_CONFIG}" \
    --routine "${JAXMG_BENCHMARK_ROUTINE}" \
    --dtype "${JAXMG_BENCHMARK_DTYPE}" \
    --grid "${JAXMG_BENCHMARK_GRID}" \
    --output-root "${JAXMG_BENCHMARK_OUTPUT_ROOT}" \
    --retry-failures \
    "${extra_args[@]}"
