# Autonomous AI Trading Agent - Docker Image
# ==========================================

FROM python:3.11-slim

# Security: Create non-root user
RUN useradd -m -u 1000 trader

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Create data directories
RUN mkdir -p data models && chown -R trader:trader /app

# Security: Switch to non-root user
USER trader

# Environment variables (set in Railway dashboard, not here!)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('https://www.okx.com')" || exit 1

# Run the trading bot
CMD ["python", "-m", "src.main"]
