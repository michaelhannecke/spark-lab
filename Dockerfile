# syntax=docker/dockerfile:1.7
#
# JupyterLab sandbox for NVIDIA DGX Spark (GB10, Grace-Blackwell, sm_121, aarch64).
#
# Design intent: the *host* never sees a pip install. The image carries the
# CUDA/PyTorch stack and JupyterLab; every notebook project gets its own venv
# under /opt/venvs (a named volume) created with --system-site-packages, so it
# inherits torch/cuDNN/NCCL from the image but layers its own junk on top.
# Blow up an env -> `rmenv <name>`. Blow up the whole container -> `compose down`.
# Either way DGX OS is untouched.

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.10-py3
FROM ${BASE_IMAGE}

ARG DEV_USER=dev
ARG DEV_UID=1000
ARG DEV_GID=1000

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        git-lfs \
        htop \
        less \
        nano \
        rsync \
        tmux \
    && rm -rf /var/lib/apt/lists/*

# JupyterLab + uv live in the IMAGE, never in a project venv. That way a
# project env can be nuked without taking the notebook server with it.
RUN pip install --no-cache-dir \
        "jupyterlab>=4.4,<5" \
        jupyterlab-nvdashboard \
        jupyter-resource-usage \
        ipywidgets \
        uv

# Non-root user whose UID/GID match the host account that owns ./notebooks,
# so bind-mounted files don't come back owned by root.
RUN if ! getent group "${DEV_GID}" >/dev/null; then groupadd -g "${DEV_GID}" "${DEV_USER}"; fi \
 && if ! getent passwd "${DEV_UID}" >/dev/null; then \
        useradd -m -u "${DEV_UID}" -g "${DEV_GID}" -s /bin/bash "${DEV_USER}"; \
    fi

# Every one of these is backed by a named volume in docker-compose.yml.
# Creating them here (owned by dev) makes Docker seed the volumes with the
# right ownership on first run.
RUN mkdir -p /opt/venvs /opt/jupyter-data /opt/jupyter-config \
             /opt/caches/{pip,uv,hf,torch} /workspace \
 && chown -R "${DEV_UID}:${DEV_GID}" \
        /opt/venvs /opt/jupyter-data /opt/jupyter-config /opt/caches /workspace

COPY --chmod=0755 bin/newenv bin/lsenv bin/rmenv /usr/local/bin/

ENV VENV_ROOT=/opt/venvs \
    JUPYTER_DATA_DIR=/opt/jupyter-data \
    JUPYTER_CONFIG_DIR=/opt/jupyter-config \
    PIP_CACHE_DIR=/opt/caches/pip \
    UV_CACHE_DIR=/opt/caches/uv \
    UV_LINK_MODE=copy \
    HF_HOME=/opt/caches/hf \
    TORCH_HOME=/opt/caches/torch \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

USER ${DEV_USER}
WORKDIR /workspace
EXPOSE 8888

# NOTE: the NGC entrypoint (/opt/nvidia/nvidia_entrypoint.sh) is inherited on
# purpose -- it sets up the CUDA env and execs CMD. Do not override ENTRYPOINT.
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--ServerApp.root_dir=/workspace"]
