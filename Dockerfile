# syntax=docker/dockerfile:1

FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /workspace

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        tzdata \
        git \
        wget \
        curl \
        unzip \
        ca-certificates \
        libgl1 \
        libglib2.0-0; \
    ln -snf "/usr/share/zoneinfo/${TZ}" /etc/localtime; \
    echo "${TZ}" > /etc/timezone; \
    rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    git clone --depth 1 https://github.com/facebookresearch/mae.git /workspace/mae

WORKDIR /workspace/mae

RUN set -eux; \
    python -m pip install --no-cache-dir --upgrade pip setuptools wheel; \
    python -m pip install --no-cache-dir \
        "numpy==1.23.5" \
        "timm==0.3.2" \
        "matplotlib<3.8" \
        "pandas<2.1" \
        tensorboard \
        pillow \
        jupyter \
        nbconvert \
        ipykernel \
        requests

# Fix old timm import for modern PyTorch.
# Do NOT import timm before patching.
RUN set -eux; \
    python - <<'PY'
from importlib.metadata import distribution
from pathlib import Path

dist = distribution("timm")
root = Path(dist.locate_file("timm"))

if not root.exists():
    raise RuntimeError(f"timm package directory not found: {root}")

patched = []
for path in root.rglob("*.py"):
    text = path.read_text()
    new_text = text.replace(
        "from torch._six import container_abcs",
        "import collections.abc as container_abcs",
    )
    if new_text != text:
        path.write_text(new_text)
        patched.append(str(path))

print("Patched timm files:")
for item in patched:
    print(" -", item)
PY

# Build-time sanity check.
RUN set -eux; \
    python - <<'PY'
import torch
import torchvision
import timm
import numpy as np

from models_mae import mae_vit_base_patch16
from models_vit import vit_base_patch16

mae_model = mae_vit_base_patch16()
vit_model = vit_base_patch16(num_classes=10)

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("timm:", timm.__version__)
print("numpy:", np.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MAE params:", sum(p.numel() for p in mae_model.parameters()))
print("ViT params:", sum(p.numel() for p in vit_model.parameters()))
PY

CMD ["/bin/bash"]
