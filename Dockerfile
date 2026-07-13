# Use official Python runtime as base image (slim variant for fast builds)
FROM python:3.11-slim

# Set environment variables for Python optimization
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Fireworks AI Configuration — v2 Multi-Model Ensemble
ENV GEMMA_MODEL="accounts/fireworks/models/deepseek-v4-pro"
ENV MINIMAX_VISION_MODEL="accounts/fireworks/models/minimax-m3"
ENV WHISPER_MODEL="accounts/fireworks/models/whisper-v3-turbo"
ENV GPT_OSS_MODEL="accounts/fireworks/models/gpt-oss-120b"
ENV QWEN_MODEL="accounts/fireworks/models/qwen3p7-plus"

# Set working directory in container
WORKDIR /app

# Install ffmpeg and other system dependencies in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port 8000 for FastAPI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
