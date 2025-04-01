# Use official PyTorch image with CUDA support
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    tar \
    fluidsynth \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pyFluidSynth  # Python bindings for fluidsynth

# Copy all files to container
COPY . .

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV FLUIDSYNTH_PATH=/usr/bin/fluidsynth

# Command to run when container starts
CMD ["bash"]
