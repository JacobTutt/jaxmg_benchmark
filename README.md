# JAXMg Benchmarks

This repository measures the distributed `jaxmg.potrs` and `jaxmg.lu_solve`
solvers backed by cuSOLVERMp. It is written for the supported execution model:
one Python process per GPU.

Each benchmark case is defined by:

- the solver (`potrs` or `lu_solve`);
- the data type;
- a two-dimensional GPU process grid;
- the matrix dimension `N`;
- the tile width `T_A`.

The benchmark is configured from TOML rather than for one fixed cluster. A
profile records visible memory per GPU, GPUs per node, CPUs per GPU, node
counts, tile widths, and datatypes. Matrix sizes near the memory frontier and
the supported process grids are then generated from that hardware profile.

`configs/isambard_gh200.toml` is the profile used for the included Isambard
results. `configs/example_h200_8gpu.toml` demonstrates an eight-H200 node: one
node automatically produces `8x1` and `4x2` grids and its larger HBM capacity
produces correspondingly larger near-limit matrices.

## What A Case Measures

For every case, the runner:

1. builds a distributed diagonal system outside the timed region;
2. runs one cold solve, including compilation;
3. runs one warm solve using the same compiled configuration;
4. validates the solution and native status after every solve;
5. writes one JSON result and one combined log.

Each `N` is chosen so the input matrix is already aligned to the selected grid
and tile size. The benchmark therefore does not exercise JAXMg's padding path.
The rule is:

```text
N % (T_A * lcm(process_rows, process_cols)) == 0
```

The benchmark times only the solver calls. Matrix construction and validation
are deliberately outside those timings.

Every case is launched in its own `srun` process group. This is important near
the memory limit: a CUDA or NCCL out-of-memory error cannot leave state behind
for the next case. The standard configuration gives a complete case one hour:
enough for the cold call, warm call, and validation, without allowing a
stalled rank to consume most of the outer allocation. If a rank exits with an
error, `srun --kill-on-bad-exit` stops its peers immediately; the one-hour
guard is only needed for a genuine collective hang.

## Installation

Install a JAXMg CUDA 12 wheel or editable checkout, then install the matching
Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_cusolvermp_cuda12.txt
```

The generic Slurm wrapper can activate a virtual environment and optionally
import an editable JAXMg checkout:

```bash
export JAXMG_BENCHMARK_VENV=/projects/u6my/users/$USER/JAXMG/venvs/jaxmg-cu12
export JAXMG_SOURCE_ROOT=/projects/u6my/users/$USER/JAXMG/jaxmg
```

`JAXMG_SOURCE_ROOT` is optional. When set, it lets the benchmark import that
source checkout instead of the wheel installed in the virtual environment.

Cluster-specific modules and library paths belong in a short setup script. For
Isambard, use the supplied profile:

```bash
export JAXMG_BENCHMARK_SETUP="$PWD/cluster_setup/isambard.sh"
```

The generic wrapper defaults to the following allocator settings; each can be
overridden in the submission environment:

```bash
export XLA_PYTHON_CLIENT_ALLOCATOR=vmm
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.99
export NCCL_CUMEM_ENABLE=0
```

## Plan And Submit

Create or select a hardware profile first. `visible_memory_mib` is the HBM
reported by `nvidia-smi` for one GPU, not the marketing capacity or aggregate
node memory. For automatic grid generation, set:

```toml
[hardware]
visible_memory_mib = 143771
gpus_per_node = 8
cpus_per_gpu = 16

[sweep]
node_counts = [1, 2]

[slurm]
# Optional site-specific sbatch arguments:
submit_args = ["--partition=gpu", "--account=my-project"]
```

For each node count, the planner generates every factor grid with the longer
dimension first. Eight GPUs therefore gives `8x1` and `4x2`; sixteen gives
`16x1`, `8x2`, and `4x4`. To benchmark another orientation or a restricted
topology, replace `node_counts` with an explicit list such as
`grids = ["2x4", "4x4"]`.

First inspect the jobs and their case counts. This does not submit anything:

```bash
python -m benchmark.cusolvermp.submit --config configs/isambard_gh200.toml
```

Restrict the output to one solver, dtype, and grid while checking a smaller
part of the sweep:

```bash
python -m benchmark.cusolvermp.submit \
  --config configs/isambard_gh200.toml \
  --routine potrs --dtype float32 --grid 4x1
```

Add `--submit` after reviewing the commands:

```bash
python -m benchmark.cusolvermp.submit \
  --config configs/isambard_gh200.toml \
  --routine potrs --dtype float32 --grid 4x1 --submit
```

One Slurm job runs every requested tile width and matrix size for that solver,
dtype, and grid. The outer job reserves complete nodes; the suite runner then
uses all four GPUs and their matching CPU cores on each node for every `srun`.

Completed cases are skipped when a suite is rerun. The Slurm wrapper retries
recorded failures, which is useful when a targeted memory-limit sweep is
adjusted after an out-of-memory result.

## Results

Results are written below the chosen output root:

```text
results/
  cases/<solver>/<dtype>/<grid>/*.json
  logs/<solver>/<dtype>/<grid>/*.log
```

Each JSON record includes cold and warm timings, validation error, native
status words, allocator settings, topology, and the known JAXMg allocations.
The estimate includes the local input matrix, RHS capacity, redistribution
scratch, and LU pivots. cuSOLVERMp workspace is not included because it is
allocated internally by that library.

Create a single table from all completed records:

```bash
python -m benchmark.cusolvermp.collect
```

This writes `results/summary.csv` and `results/summary.jsonl`. Plot one solver,
dtype, and grid with:

```bash
python -m benchmark.cusolvermp.plot \
  --routine potrs --dtype float32 --grid 4x1
```

## Tests

The CPU-only tests check the case-selection and memory calculations:

```bash
python -m unittest discover -s tests -v
```
