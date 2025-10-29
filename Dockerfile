FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies for OpenCV / ONNX Runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . ./

# Prepare the models directory for downloaded assets at runtime
RUN mkdir -p /app/models

EXPOSE 5000

CMD ["python", "app.py"]
