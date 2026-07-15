# JAXMg Benchmarks

This repository benchmarks the distributed `jaxmg.potrs` and
`jaxmg.lu_solve` routines backed by cuSOLVERMp. Each GPU runs one Python
process.

The benchmark measures one cold solve and one warm solve for combinations of:

- matrix size;
- tile size;
- datatype;
- GPU count and two-dimensional process grid.

Every result is numerically validated. Matrix construction and validation are
outside the timed region.

## Requirements

You need:

- a Slurm cluster with NVIDIA GPUs;
- a working CUDA installation;
- a Python environment containing JAXMg and its CUDA dependencies;
- one or more complete GPU nodes available to a Slurm job.

Install the benchmark dependencies in the environment containing JAXMg:

```bash
git clone https://github.com/JacobTutt/jaxmg_benchmark.git
cd jaxmg_benchmark
python -m pip install -r requirements_cusolvermp_cuda12.txt
```

## 1. Describe Your Cluster

Copy the example profile:

```bash
cp configs/example_h200_8gpu.toml configs/my_cluster.toml
```

Edit the `[hardware]` section:

```toml
[hardware]
gpu_name = "NVIDIA H200"
visible_memory_mib = 143771
allocator_fraction = 0.99
gpus_per_node = 8
cpus_per_gpu = 16
```

The important values are:

- `visible_memory_mib`: memory reported for one GPU by `nvidia-smi`, in MiB;
- `gpus_per_node`: GPUs available on each requested node;
- `cpus_per_gpu`: CPU cores assigned to each GPU process.

Check the first value with:

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

Choose the node counts to benchmark:

```toml
[sweep]
node_counts = [1, 2, 3, 4]
```

The planner creates each non-transposed factor grid automatically. On an
eight-GPU node this gives:

```text
1 node:  8x1, 4x2
2 nodes: 16x1, 8x2, 4x4
3 nodes: 24x1, 12x2, 8x3, 6x4
4 nodes: 32x1, 16x2, 8x4
```

To select grids manually, replace `node_counts` with, for example:

```toml
grids = ["8x1", "4x2"]
```

The profile also contains the routines, datatypes, tile sizes, timing limits,
and Slurm resources. Optional site arguments can be added directly:

```toml
[execution]
cold_runs = 1
warm_runs = 3
case_timeout_seconds = 3600
```

`cold_runs` must remain `1`. Set `warm_runs` to the number of timed calls made
after compilation; the result records every warm duration and reports their
median, minimum, and maximum. `case_timeout_seconds` limits the complete fresh
process-group case, so increase it when adding warm calls or benchmarking very
large matrices.

Slurm resources and optional site arguments are configured separately:

```toml
[slurm]
suite_walltime = "12:00:00"
suite_memory = "512G"
submit_args = ["--partition=gpu", "--account=my-project"]
```

## 2. Set Up The Compute-Node Environment

Point the runner at the Python environment containing JAXMg:

```bash
export JAXMG_BENCHMARK_VENV="$HOME/.venvs/jaxmg"
```

If JAXMg is installed from an editable checkout rather than a wheel, also set:

```bash
export JAXMG_SOURCE_ROOT="$HOME/src/jaxmg"
```

Some clusters require modules or additional library paths on compute nodes.
Put those commands in a small shell script:

```bash
#!/usr/bin/env bash
module load cuda gcc
```

Then point the benchmark at it:

```bash
export JAXMG_BENCHMARK_SETUP="$PWD/cluster_setup/my_cluster.sh"
```

Skip this variable when your Python environment already provides everything.

The runner defaults to:

```bash
XLA_PYTHON_CLIENT_ALLOCATOR=vmm
XLA_PYTHON_CLIENT_MEM_FRACTION=0.99
NCCL_CUMEM_ENABLE=0
```

Export different values before submission if your system requires them.

## 3. Review The Plan

Start with one small part of the sweep. This command only prints the planned
job and does not submit it:

```bash
python -m benchmark.cusolvermp.submit \
  --config configs/my_cluster.toml \
  --routine potrs \
  --dtype float32 \
  --grid 8x1
```

The output shows the number of cases, requested nodes and GPUs, wall time, and
the exact `sbatch` command.

Matrix sizes are derived from the configured GPU memory. More HBM or more GPUs
therefore produces larger near-limit cases automatically. Every selected `N`
also satisfies:

```text
N % (tile_size * lcm(process_rows, process_cols)) == 0
```

Consequently, the standard sweep does not require JAXMg padding.

## 4. Submit

After checking the printed command, add `--submit`:

```bash
python -m benchmark.cusolvermp.submit \
  --config configs/my_cluster.toml \
  --routine potrs \
  --dtype float32 \
  --grid 8x1 \
  --output-root results/my_cluster \
  --submit
```

Remove the routine, datatype, or grid filters to submit every corresponding
entry in the profile. Each solver configuration receives one outer Slurm job;
each matrix-size and tile-size case runs in a fresh `srun` process group.

Completed passing cases are skipped when a suite is submitted again. Recorded
failures are retried.

## Results

Results are stored as:

```text
results/my_cluster/
  cases/<routine>/<dtype>/<grid>/*.json
  logs/<routine>/<dtype>/<grid>/*.log
```

Each successful JSON record contains cold and warm timings, numerical error,
native status values, topology, package versions, and known memory estimates.
The estimate includes the local matrix, RHS capacity, redistribution scratch,
and LU pivots. cuSOLVERMp's internal workspace is not known before execution.

Collect all records into CSV and JSONL tables:

```bash
python -m benchmark.cusolvermp.collect \
  --output-root results/my_cluster \
  --csv results/my_cluster/summary.csv \
  --jsonl results/my_cluster/summary.jsonl
```

Plot one completed configuration:

```bash
python -m benchmark.cusolvermp.plot \
  --output-root results/my_cluster \
  --routine potrs \
  --dtype float32 \
  --grid 8x1
```

## Tests

The CPU-only tests verify configuration parsing, grid generation, alignment,
memory accounting, and command construction:

```bash
python -m unittest discover -s tests -v
```

`configs/isambard_gh200.toml` and `cluster_setup/isambard.sh` are examples of
a completed site profile. They are not required on another system.
