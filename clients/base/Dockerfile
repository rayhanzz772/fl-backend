FROM python:3.9-slim

WORKDIR /app

# Copy requirements dari root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy client code
COPY clients/base/client.py .

# Create data directory
RUN mkdir -p data logs

EXPOSE 5001

CMD ["python", "client.py"]
