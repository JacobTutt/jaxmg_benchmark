# JAXMg Benchmark

This repository benchmarks the distributed `jaxmg.potrs` and
`jaxmg.lu_solve` APIs backed by cuSOLVERMp. The suite uses one Python process
per GPU and supports 1-4 Isambard-AI nodes (4-16 GPUs).

## Installation

Install the JAXMg wheel built from the target branch, then install the Python
benchmark dependencies that match its pinned CUDA/JAX release:

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate

# JAXMg's cuSOLVERMp backend currently targets CUDA 12 / JAX 0.10.1.
pip install -r requirements_cusolvermp_cuda12.txt
```

## cuSOLVERMp suite

The default Isambard configuration covers:

- `potrs` and `lu_solve`;
- `float32`, `float64`, `complex64`, and `complex128`;
- tile sizes 256, 512, 1024, 2048, and 4096;
- 4x1, 2x2, 8x1, 4x2, 12x1, 6x2, 4x3, 16x1, 8x2, and 4x4 process grids;
- one cold solve and three warm solves per case;
- baseline matrix sizes plus a denser sweep from 85% to 100% of the raw
  distributed input-matrix capacity.

Every `(routine, dtype, grid, N, T_A)` case is launched as a fresh `srun`
process group. A failed or out-of-memory case therefore cannot contaminate
later cases. Within that process group, all four solves use one compiled JAX
configuration, so the three warm measurements are genuine cache-reuse runs.
Input creation and result validation are excluded from solver timings.

The planner only emits dimensions satisfying

```text
N % (T_A * lcm(process_rows, process_cols)) == 0
```

so the large input matrix requires no tile-alignment padding. The single RHS
still has the small internal capacity needed to route an `N x 1` input across
the selected process grid.

### Isambard setup

The Slurm wrapper expects a JAXMg environment and optionally an editable
JAXMg source checkout:

```bash
export JAXMG_BENCHMARK_VENV=/projects/u6my/users/$USER/JAXMG/venvs/jaxmg-cu12
export JAXMG_SOURCE_ROOT=/projects/u6my/users/$USER/JAXMG/jaxmg
```

For a new environment, install the target branch's Jenkins wheel (or its Bazel
build) and then install `requirements_cusolvermp_cuda12.txt`.

It sets the validated allocator configuration before Python imports JAX:

```bash
export XLA_PYTHON_CLIENT_ALLOCATOR=vmm
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.99
```

The hardware model uses the 97,871 MiB visible on each Isambard GH200 GPU,
giving a 94.622 GiB VMM budget per process. Memory-frontier points are targets,
not guaranteed successes: cuSOLVERMp workspace is opaque and depends on the
routine, dtype, grid, matrix size, and tile size.

### Review and submit

First print the complete submission manifest without changing Slurm state:

```bash
python -m benchmark.cusolvermp.submit
```

Filters can select a small checkpoint:

```bash
python -m benchmark.cusolvermp.submit \
  --routine potrs --dtype float32 --grid 4x1
```

After reviewing the generated command and case count, submit that selection:

```bash
python -m benchmark.cusolvermp.submit \
  --routine potrs --dtype float32 --grid 4x1 --submit
```

Each logical Slurm job covers every configured tile and matrix size for one
routine, dtype, and process grid. Completed cases are skipped on restart;
recorded failures are retried by the supplied Slurm wrapper.

### Results

Each case writes one atomic JSON record under `results/cases` and a separate
combined stdout/stderr log under `results/logs`. Records contain cold and warm
timings, the warm median, validation error, solver status, topology, allocator
configuration, and analytical known-buffer sizes. No `nvidia-smi` polling is
performed during a benchmark.

Create convenient summary tables with:

```bash
python -m benchmark.cusolvermp.collect
```

This produces `results/summary.csv` and `results/summary.jsonl`.

Plot one completed configuration with:

```bash
python -m benchmark.cusolvermp.plot \
  --routine potrs --dtype float32 --grid 4x1
```

### Planner tests

The CPU-only tests verify alignment and analytical memory accounting:

```bash
python -m unittest discover -s tests -v
```
