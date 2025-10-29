FROM python:3.11-slim

WORKDIR /app

# System dependencies for OpenCV / ONNXRuntime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

# Create models directory
RUN mkdir -p /app/models

EXPOSE 5000
CMD ["python", "app.py"]
