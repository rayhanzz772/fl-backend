FROM python:3.9-slim

WORKDIR /app

# Copy requirements dari root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY server/server.py .
COPY server/model.py .

EXPOSE 5000

CMD ["python", "server.py"]
