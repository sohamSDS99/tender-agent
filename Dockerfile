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

# Copy dependency definition
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -e "."

# Copy source code
COPY . .

# Don't run as root
RUN useradd --create-home appuser
USER appuser

# Run the bridge
CMD ["python", "scripts/nexus_bridge.py"]
