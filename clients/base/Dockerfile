FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy client code and shared modules
COPY client.py .
COPY ../shared /app/shared

# Create data directory
RUN mkdir -p data logs

# Environment variables akan di-set dari docker-compose
# Tidak hardcoded di sini

EXPOSE 5001

CMD ["python", "client.py"]