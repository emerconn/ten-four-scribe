# Build stage
FROM nvidia/cuda:12.6.3-base-ubuntu24.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-minimal \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM nvidia/cuda:12.6.3-base-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Chicago
ENV PYTHONUNBUFFERED=1

# Install only runtime dependencies - changed python3-minimal to python3
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /work
COPY transcribe.py .
CMD ["python3", "-u", "transcribe.py"]
