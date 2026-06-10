FROM python:3.9-slim

WORKDIR /app
ENV PYTHONPATH=/app/server:/app

# Copy requirements dari root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code as a directory so local imports stay in the same module folder
COPY server ./server

EXPOSE 5000

CMD ["python", "server/server.py"]
