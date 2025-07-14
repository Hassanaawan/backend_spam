FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (important for numpy, tflite, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy files and install Python packages
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

# Expose the required port for Hugging Face Spaces
EXPOSE 7860

# Run the Flask app with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
