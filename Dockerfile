# Use a supported Python runtime
FROM python:3.11-slim

# Set a working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . /app

# Expose the port Cloud Run / GCP uses
EXPOSE 8080

# # Run Gunicorn for production
# CMD ["gunicorn", "-w", "4", "-b", ":8080", "app:app"]

# CRITICAL CHANGES:
# --preload: Load app before forking workers (saves memory)
# --timeout 120: Give workers more time to start (default is 30s)
# -w 1: Single worker (or 2 max for small instances)
# --max-requests 100: Recycle workers to prevent memory leaks
# --max-requests-jitter 10: Randomize recycling
CMD ["gunicorn", "--preload", "-w", "1", "--timeout", "120", "--max-requests", "100", "--max-requests-jitter", "10", "-b", ":8080", "app:app"]
