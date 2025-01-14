FROM nvidia/cuda:12.6.3-base-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Chicago
ENV PYTHONUNBUFFERED=1

# Combine all apt operations in a single layer and clean up properly
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-minimal \
    python3-pip \
    python3-venv \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /root/.cache/pip/*

# Create and activate virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Set working directory and copy application
WORKDIR /work
COPY transcribe.py .

CMD ["python3", "-u", "transcribe.py"]
