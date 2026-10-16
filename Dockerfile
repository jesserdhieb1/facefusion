FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Avoid interactive prompt hangs
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PYTHONUNBUFFERED=1

# Install system utilities and Python 3.12 via deadsnakes
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up dedicated Python 3.12 virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade packaging tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install core runtime dependencies (CUDA-enabled ONNX Runtime & FastAPI service)
RUN pip install --no-cache-dir \
    gradio==5.50.0 \
    gradio-rangeslider==0.0.8 \
    numpy==2.4.6 \
    onnx==1.23.1 \
    onnxruntime-gpu==1.24.4 \
    opencv-python-headless==5.0.0.93 \
    tqdm==4.70.1 \
    scipy==1.18.1 \
    fastapi==0.143.1 \
    uvicorn==0.54.0 \
    python-multipart==0.0.32

WORKDIR /facefusion

# Copy project source code
COPY . .

# Ports: 8000 for FastAPI microservice, 7860 for optional Gradio Web UI
EXPOSE 8000 7860

# Default command starts the REST API microservice
CMD ["python", "service.py"]
