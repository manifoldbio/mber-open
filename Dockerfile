# mber-open Docker image for reproducible AMBER relaxation testing
# Supports both GPU and CPU modes for OpenMM

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (uses conda-forge by default, no ToS acceptance required)
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p $CONDA_DIR \
    && rm /tmp/miniforge.sh \
    && conda clean -afy

# Set up working directory
WORKDIR /workspace/mber-open

# Copy environment file first for better caching
COPY docker/environment.yml ./

# Create conda environment (using conda-forge and bioconda only - no defaults)
RUN conda env create -f environment.yml \
    && conda clean -afy

# Make conda environment accessible
SHELL ["conda", "run", "-n", "mber", "/bin/bash", "-c"]

# Copy source code and build files
COPY src/ ./src/
COPY protocols/ ./protocols/
COPY setup.py requirements.txt README.md ./

# Install pip dependencies and mber-open + protocols
RUN pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple \
    && pip install -e . \
    && pip install -e protocols

# Copy remaining files (scripts, examples, etc.)
COPY download_af_weights.sh ./
COPY docker/ ./docker/

# Download AlphaFold weights to a persistent location
# This will be cached in the image or can be mounted as a volume
ENV MBER_AF_PARAMS_DIR=/root/.mber/af_params
RUN mkdir -p /root/.mber \
    && bash download_af_weights.sh

# Create output directory
RUN mkdir -p /workspace/mber-open/output

# Copy example files
COPY protocols/src/mber_protocols/stable/VHH_binder_design/examples/ ./examples/

# Set environment variables for runtime
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Default entrypoint runs in conda environment
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "mber"]

# Default command: run the VHH design with test settings
CMD ["mber-vhh", "--settings", "./docker/test_settings.yml"]
