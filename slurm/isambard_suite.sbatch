#!/usr/bin/env bash
set -euo pipefail

# Resource shape, wall time, and output paths are supplied by submit.py.
: "${JAXMG_BENCHMARK_VENV:?Set JAXMG_BENCHMARK_VENV to the JAXMg virtual environment}"
: "${JAXMG_BENCHMARK_CONFIG:?Missing benchmark configuration}"
: "${JAXMG_BENCHMARK_ROUTINE:?Missing routine}"
: "${JAXMG_BENCHMARK_DTYPE:?Missing dtype}"
: "${JAXMG_BENCHMARK_GRID:?Missing process grid}"
: "${JAXMG_BENCHMARK_OUTPUT_ROOT:?Missing output directory}"

module load cray-python/3.11.7 cuda/12.6 gcc-native/14.2
source "${JAXMG_BENCHMARK_VENV}/bin/activate"

# An editable source checkout can override the wheel installed in the venv.
if [[ -n "${JAXMG_SOURCE_ROOT:-}" ]]; then
    export PYTHONPATH="${JAXMG_SOURCE_ROOT}/src:${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
else
    export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"
fi

# Include CUDA-wheel libraries and Isambard's NCCL/UCC installations.
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
# Isambard supports cuMem, but retain the proven conservative NCCL setting by
# default. Users can override it explicitly when testing another runtime.
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export OMP_NUM_THREADS=1

extra_args=()
if [[ -n "${JAXMG_BENCHMARK_TILE_SIZE:-}" ]]; then
    extra_args+=(--tile-size "${JAXMG_BENCHMARK_TILE_SIZE}")
fi
if [[ -n "${JAXMG_BENCHMARK_MATRIX_SIZES:-}" ]]; then
    # Space-separated explicit sizes are useful for a tightly bracketed
    # memory-limit probe.  run_suite still launches each case in a fresh srun.
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
