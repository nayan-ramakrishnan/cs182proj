#!/usr/bin/env bash
# One-time setup on the Lambda instance:  bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv ~/icl-venv
source ~/icl-venv/bin/activate
pip install -q -U pip
pip install -q -r requirements.txt

python - <<'EOF'
import jax
print("JAX", jax.__version__, "devices:", jax.devices())
assert jax.devices()[0].platform == "gpu", "JAX does not see the GPUs"
EOF

JAX_PLATFORMS=cpu python tests/test_core.py

grep -q icl-venv ~/.bashrc || echo 'source ~/icl-venv/bin/activate' >> ~/.bashrc
echo
echo "Setup OK. Persistent filesystems on this machine (put RUNS_DIR on one of these):"
df -h | grep -E "nfs|lambda|/home/ubuntu/" || true
echo 'Then: echo "export RUNS_DIR=/path/to/filesystem/cs182runs" >> ~/.bashrc && source ~/.bashrc'
