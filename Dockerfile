FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Install Python 3.12 and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev

COPY . .
RUN uv sync --no-dev

ENTRYPOINT ["uv", "run", "scripts/transcribe.py"]
