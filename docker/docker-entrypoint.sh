#!/bin/bash
set -e

# mBER Docker Entrypoint Script
# Handles weight mounting/symlinking and runs the mber-vhh CLI

MBER_HOME="${HOME}/.mber"
WEIGHTS_MOUNT="/mber_weights"

# Show help if no arguments provided
if [ $# -eq 0 ]; then
    echo "mBER Docker Container"
    echo ""
    echo "Usage: docker run --gpus all [mounts] mber:latest [options]"
    echo ""
    echo "Required options (if not using --settings):"
    echo "  --input-pdb PATH      Target PDB file"
    echo "  --output-dir PATH     Output directory"
    echo "  --chains CHAINS       Target chains (e.g., 'A' or 'A,B')"
    echo ""
    echo "Common options:"
    echo "  --settings PATH       Use YAML settings file"
    echo "  --hotspots RESIDUES   Target residues (e.g., 'A56')"
    echo "  --num-accepted N      Designs to generate (default: 100)"
    echo "  --help                Show full help"
    echo ""
    echo "Example:"
    echo "  docker run --gpus all \\"
    echo "    -v \$(pwd)/output:/outputs \\"
    echo "    -v \$(pwd)/inputs:/inputs:ro \\"
    echo "    mber:latest \\"
    echo "    --input-pdb /inputs/target.pdb \\"
    echo "    --output-dir /outputs/my_run \\"
    echo "    --chains A"
    echo ""
    exit 0
fi

echo "=== mBER Docker Container ==="
echo "Setting up model weights..."

# Create .mber directory if it doesn't exist
mkdir -p "${MBER_HOME}"

# Set HuggingFace cache location
export HF_HOME="${MBER_HOME}/huggingface"
export HF_HUB_OFFLINE=1  # Prevent writes to read-only mounted cache
unset TRANSFORMERS_CACHE  # Avoid conflicts with HF_HOME

# Function to check if weights exist at a path
check_weights_exist() {
    local base_path=$1
    [ -f "${base_path}/af_params/params_model_5_ptm.npz" ] && \
    [ -f "${base_path}/nbb2_weights/nanobody_model_1" ] && \
    [ -d "${base_path}/huggingface/hub/models--facebook--esm2_t33_650M_UR50D" ]
}

# Function to setup symlink for a weights subdirectory
setup_weights_dir() {
    local subdir=$1
    local source_path="${WEIGHTS_MOUNT}/${subdir}"
    local target_path="${MBER_HOME}/${subdir}"
    
    if [ -d "${source_path}" ] && [ "$(ls -A ${source_path} 2>/dev/null)" ]; then
        echo "  ✓ ${subdir}"
        rm -rf "${target_path}"
        ln -sf "${source_path}" "${target_path}"
        return 0
    fi
    return 1
}

# Check for built-in weights first (from --build-arg INCLUDE_WEIGHTS=true)
if check_weights_exist "${MBER_HOME}"; then
    echo "Using built-in weights from ${MBER_HOME}"
    echo "  ✓ AlphaFold2"
    echo "  ✓ NanoBodyBuilder2"
    echo "  ✓ ESM2"
# Otherwise check for mounted weights
elif [ -d "${WEIGHTS_MOUNT}" ] && [ "$(ls -A ${WEIGHTS_MOUNT} 2>/dev/null)" ]; then
    echo "Using mounted weights from ${WEIGHTS_MOUNT}:"
    setup_weights_dir "af_params" || true
    setup_weights_dir "nbb2_weights" || true
    setup_weights_dir "openfold_weights" || true
    setup_weights_dir "jit_cache" || true
    setup_weights_dir "huggingface" && export HF_HOME="${MBER_HOME}/huggingface" || true
    
    echo ""
    echo "Verifying weights..."
    
    # Verify all required weights are present
    missing_weights=""
    [ ! -f "${MBER_HOME}/af_params/params_model_5_ptm.npz" ] && missing_weights="${missing_weights} AlphaFold2"
    [ ! -f "${MBER_HOME}/nbb2_weights/nanobody_model_1" ] && missing_weights="${missing_weights} NanoBodyBuilder2"
    [ ! -d "${HF_HOME}/hub/models--facebook--esm2_t33_650M_UR50D" ] && missing_weights="${missing_weights} ESM2"
    
    if [ -n "$missing_weights" ]; then
        echo "Missing weights:${missing_weights}"
        echo "Downloading missing weights..."
        bash /app/download_weights.sh "${MBER_HOME}"
    else
        echo "  ✓ AlphaFold2"
        echo "  ✓ NanoBodyBuilder2"
        echo "  ✓ ESM2"
    fi
else
    echo "No weights found. Downloading all weights..."
    echo "This will take several minutes on first run."
    echo ""
    bash /app/download_weights.sh "${MBER_HOME}"
fi

echo ""
echo "Starting mBER VHH binder design..."
echo ""

# Execute mber-vhh with all arguments passed to the container
exec conda run --no-capture-output -n mber mber-vhh "$@"
