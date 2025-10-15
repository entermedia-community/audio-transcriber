# ===== Stage 1: Build dependencies =====
FROM python:3.10-slim AS builder

# Avoid interactive prompts and improve performance
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install build tools and ffmpeg (required for audio decoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ===== Stage 2: Final image =====
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install ffmpeg for audio file processing
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Copy installed site-packages from builder
COPY --from=builder /usr/local /usr/local

# Copy app code
COPY server.py .

# Expose FastAPI port
EXPOSE 8000

# Default command to run the app
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
