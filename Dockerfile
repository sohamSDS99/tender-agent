# =============================================================================
# Tender Agent Bridge — Production Dockerfile
# =============================================================================

FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for PDF/DOCX parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition + vendored nexus-sdk (lives in nexus-ams repo
# canonically; vendored here so the bridge image is self-contained — pip
# install resolves the local path, no cross-repo git access needed at build).
COPY pyproject.toml ./
COPY vendor/ ./vendor/

# Install nexus-sdk first (it's a hard runtime import in scripts/nexus_bridge.py),
# then install tender-agent itself in editable mode.
RUN pip install --no-cache-dir ./vendor/nexus-sdk && \
    pip install --no-cache-dir -e "."

# Copy source code
COPY . .

# Don't run as root
RUN useradd --create-home appuser
USER appuser

# Run the bridge
CMD ["python", "scripts/nexus_bridge.py"]
