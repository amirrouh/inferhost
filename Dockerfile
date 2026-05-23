# Test container for inferhost v0.5+.
# Boots on the host's GPU via NVIDIA Container Toolkit (--gpus all). Source is
# baked in at build time so the test artifact is immutable; runtime state
# (HF cache, model registry, llama-server binary) lives on named volumes so
# model downloads survive `docker compose down`.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git tini \
        python3.11 python3.11-venv python3-pip \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

# uv for fast Python env + install. Lives under /root/.local/bin.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/workspace/.venv/bin:$PATH"

WORKDIR /workspace

# Metadata first → layer-cache friendly when source changes but deps don't.
COPY pyproject.toml uv.lock* README.md LICENSE ./
COPY src ./src
RUN uv venv --python 3.11 && uv pip install -e ".[dev]"

COPY tests ./tests
COPY .env.example ./
COPY tests/docker_smoke.sh /usr/local/bin/inferhost-smoke
COPY tests/docker_functional.sh /usr/local/bin/inferhost-functional
RUN chmod +x /usr/local/bin/inferhost-smoke /usr/local/bin/inferhost-functional

# Container paths for volume mounts (set so the user can override via compose).
ENV INFERHOST_CONFIG_DIR=/inferhost/config \
    INFERHOST_DATA_DIR=/inferhost/data \
    INFERHOST_HF_CACHE=/inferhost/hf-cache \
    INFERHOST_GATEWAY_PORT=9001 \
    INFERHOST_SWAP_PORT=9090 \
    PYTHONUNBUFFERED=1

RUN mkdir -p /inferhost/config /inferhost/data /inferhost/hf-cache

EXPOSE 9001

ENTRYPOINT ["tini", "--"]
CMD ["bash"]
