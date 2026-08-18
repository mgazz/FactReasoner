# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1: build the Merlin C++ inference engine from source
# ---------------------------------------------------------------------------
FROM debian:bookworm-slim AS merlin-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        cmake \
        build-essential \
        libboost-program-options-dev \
        libboost-thread-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/radum2275/merlin.git /merlin

RUN cmake -S /merlin -B /merlin/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DMERLIN_BUILD_TESTS=OFF \
        -DMERLIN_BUILD_PYTHON=OFF \
        -DMERLIN_BUILD_DOCS=OFF \
        -DBoost_USE_STATIC_LIBS=ON \
    && cmake --build /merlin/build --target merlin -j"$(nproc)"

# ---------------------------------------------------------------------------
# Stage 2: Python service image
# ---------------------------------------------------------------------------
FROM python:3.11-slim

# System dependencies required by transitive packages (chromadb, sentence-transformers, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast, reproducible installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies including the server extra (fastapi + uvicorn + gunicorn).
# The `rits` and `vllm` extras are omitted: rits needs IBM-internal packages
# and vllm is GPU-only; both are injected at runtime where needed.
RUN uv sync --frozen --no-dev --extra server --no-install-project

# Copy source and static artifacts
COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY README.md ./

# Install the project itself (with the server extra)
RUN uv sync --frozen --no-dev --extra server

# Pre-download the sentence-transformers model used by the NLI similarity gate
# so the container never needs to reach HF Hub at inference time.
RUN /app/.venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Drop the Merlin binary built in stage 1
COPY --from=merlin-builder /merlin/build/merlin ./artifacts/merlin
RUN chmod +x ./artifacts/merlin

# Ensure the virtualenv binaries are on PATH
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# Default entrypoint is the REST server.
# Override ENTRYPOINT at runtime to use the CLI:
#   docker run --entrypoint fact-reasoner <image> --help
ENTRYPOINT ["fact-reasoner-server"]
# --workers 2: Gunicorn spawns two UvicornWorker processes, each with its own
# event loop. A long /assess call can only stall one worker's loop; the other
# remains free to answer /health and keep the liveness probe satisfied.
# Override via the FR_WORKERS env variable or by passing --workers at runtime.
CMD ["--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
