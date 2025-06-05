FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt update && apt install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    ffmpeg \
    libsndfile1 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

COPY app/requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    bash -c 'for i in {1..3}; do \
        pip install --no-cache-dir --prefer-binary --requirement /app/requirements.txt && break || sleep 10; done'


COPY app /app

EXPOSE 5000

CMD ["python", "server.py"]

