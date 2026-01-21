#!/usr/bin/env bash

# Require arguments: number of processes and benchmark name
if [ -z "${1:-}" ] || [ -z "${2:-}" ]; then
  echo "Usage: $0 <num_processes> <benchmark:{potrs|potri|syevd}>"
  exit 1
fi

num_processes="$1"
benchmark_name="$2"

# Validate benchmark name
case "$benchmark_name" in
  potrs|potri|syevd)
    ;;
  *)
    echo "Invalid benchmark '$benchmark_name'. Allowed: potrs, potri, syevd"
    exit 1
    ;;
esac

script="benchmark_${benchmark_name}.py"

# export CUDA_VISIBLE_DEVICES="0"
range=$(seq 0 $(($num_processes - 1)))
HOSTS=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
echo $HOSTS
MASTER=${HOSTS[0]}
for i in $range; do
  echo "Launching $script for process $i of $num_processes"
  python -u "$script" "$MASTER:10001" $i $num_processes > /tmp/toy_$i.out &
done

wait

for i in $range; do
  echo "=================== process $i output ==================="
  cat /tmp/toy_$i.out
  echo
done