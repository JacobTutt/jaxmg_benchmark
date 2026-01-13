# JAXMG Benchmark

## Installation

Install the requirements that match your GPU's CUDA major version:

- CUDA 12.x GPUs: use [requirements_cuda12.txt](requirements_cuda12.txt)
- CUDA 13.x GPUs: use [requirements_cuda13.txt](requirements_cuda13.txt)

Example commands:

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate

# Install deps for CUDA 12.x GPUs
pip install -r requirements_cuda12.txt

# OR, for CUDA 13.x GPUs
pip install -r requirements_cuda13.txt
```

If you're unsure of your CUDA version, check your NVIDIA driver/CUDA toolkit or run `nvidia-smi` and consult cluster documentation.

