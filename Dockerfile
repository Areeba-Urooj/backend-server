# Use an official Python runtime as a parent image
FROM python:3.13-slim

# Install system dependencies (FFmpeg for librosa)
RUN apt-get update -y && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /opt/render/project/src

# Copy the dependency file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set environment variables for the application
ENV PYTHONUNBUFFERED 1
ENV UPLOAD_DIR /opt/render/project/src/uploads

# Command to run the worker (Render Worker Service)
# This will be overridden by Render's UI setting for the run command, but is good practice.
CMD ["python", "app/analysis_worker.py"]
