# mBER Docker Image
# Runs mBER VHH binder design in CLI mode with CUDA 12.8 support
#
# Build from repo root:
#   docker build -t mber:latest -f docker/Dockerfile .
#
# Build with weights included (~9GB larger, no mounting needed):
#   docker build -t mber:latest -f docker/Dockerfile --build-arg INCLUDE_WEIGHTS=true .

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# Build argument to include weights in the image
ARG INCLUDE_WEIGHTS=false

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    build-essential \
    ca-certificates \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (conda-forge based, no ToS issues)
ENV CONDA_DIR=/opt/conda
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniforge.sh
ENV PATH="${CONDA_DIR}/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy the entire application (needed for -e . install in environment.yml)
COPY . .

# Create conda environment (includes pip install -e . for mber)
RUN conda env create -f environment.yml \
    && conda clean -afy

# Make sure conda environment is activated by default
SHELL ["conda", "run", "-n", "mber", "/bin/bash", "-c"]

# Install mber-protocols in the conda environment
RUN conda run -n mber pip install -e protocols

# Set up entrypoint script
RUN chmod +x /app/docker/docker-entrypoint.sh

# Create directories for volume mounts
RUN mkdir -p /mber_weights /outputs /inputs /root/.mber

# Set environment variables
ENV HOME=/root
ENV MBER_WEIGHTS_DIR=/mber_weights
ENV HF_HOME=/root/.mber/huggingface
ENV HF_HUB_OFFLINE=1

# Optionally download weights at build time
# This makes the image ~9GB larger but eliminates the need to mount weights
RUN if [ "$INCLUDE_WEIGHTS" = "true" ]; then \
    echo "Downloading weights (this will take several minutes)..." && \
    conda run -n mber bash /app/download_weights.sh /root/.mber && \
    echo "Weights included in image at /root/.mber"; \
    else \
    echo "Weights not included. Mount at runtime with -v ~/.mber:/mber_weights:ro"; \
    fi

ENTRYPOINT ["/app/docker/docker-entrypoint.sh"]
