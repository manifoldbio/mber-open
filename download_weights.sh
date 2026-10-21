#!/bin/bash
# Download all model weights required for mBER
# The AlphaFold2 download steps are adapted from BindCraft's install_bindcraft.sh
# (github.com/martinpacesa/BindCraft), Copyright (c) 2024 Martin Pacesa, MIT License;
# see THIRD_PARTY_NOTICES.md.
# This includes AlphaFold2, NanoBodyBuilder2, and ESM models
#
# Usage: bash download_weights.sh [target_dir]
# Default target: ~/.mber
#
# Options:
#   --skip-esm    Skip ESM model downloads (saves ~5GB, but ESM2 is required)
#   --with-esmfold  Also download ESMFold weights (~16GB extra)

set -e

# Parse arguments
MBER_DIR="${HOME}/.mber"
SKIP_ESM=false
WITH_ESMFOLD=false

for arg in "$@"; do
    case $arg in
        --skip-esm)
            SKIP_ESM=true
            shift
            ;;
        --with-esmfold)
            WITH_ESMFOLD=true
            shift
            ;;
        *)
            if [[ ! "$arg" == -* ]]; then
                MBER_DIR="$arg"
            fi
            ;;
    esac
done

echo "=========================================="
echo "mBER Model Weights Downloader"
echo "=========================================="
echo "Target directory: ${MBER_DIR}"
echo ""

# Create directory structure
mkdir -p "${MBER_DIR}/af_params"
mkdir -p "${MBER_DIR}/nbb2_weights"
mkdir -p "${MBER_DIR}/huggingface"

# ==========================================
# AlphaFold2 Weights
# ==========================================
AF_DIR="${MBER_DIR}/af_params"
AF_PARAMS_FILE="${AF_DIR}/alphafold_params_2022-12-06.tar"
AF_CHECK_FILE="${AF_DIR}/params_model_5_ptm.npz"

echo "[1/4] AlphaFold2 Weights"
echo "----------------------------------------"

if [ -f "${AF_CHECK_FILE}" ]; then
    echo "✓ AlphaFold2 weights already exist"
else
    echo "Downloading AlphaFold2 weights (~3.5GB)..."
    wget -q --show-progress -O "${AF_PARAMS_FILE}" \
        "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
    
    echo "Verifying archive integrity..."
    if ! tar tf "${AF_PARAMS_FILE}" >/dev/null 2>&1; then
        echo "ERROR: Downloaded archive is corrupt. Please try again."
        rm -f "${AF_PARAMS_FILE}"
        exit 1
    fi
    
    echo "Extracting weights..."
    tar -xf "${AF_PARAMS_FILE}" -C "${AF_DIR}"
    
    if [ ! -f "${AF_CHECK_FILE}" ]; then
        echo "ERROR: Extraction failed. Expected file not found."
        exit 1
    fi
    
    echo "Cleaning up archive..."
    rm -f "${AF_PARAMS_FILE}"
    
    echo "✓ AlphaFold2 weights installed"
fi
echo ""

# ==========================================
# NanoBodyBuilder2 Weights
# ==========================================
NBB2_DIR="${MBER_DIR}/nbb2_weights"

echo "[2/4] NanoBodyBuilder2 Weights"
echo "----------------------------------------"

NBB2_MODELS=(
    "nanobody_model_1:https://zenodo.org/record/7258553/files/nanobody_model_1?download=1"
    "nanobody_model_2:https://zenodo.org/record/7258553/files/nanobody_model_2?download=1"
    "nanobody_model_3:https://zenodo.org/record/7258553/files/nanobody_model_3?download=1"
    "nanobody_model_4:https://zenodo.org/record/7258553/files/nanobody_model_4?download=1"
)

all_exist=true
for model_entry in "${NBB2_MODELS[@]}"; do
    model_name="${model_entry%%:*}"
    model_path="${NBB2_DIR}/${model_name}"
    if [ ! -f "${model_path}" ] || [ ! -s "${model_path}" ]; then
        all_exist=false
        break
    fi
done

if [ "$all_exist" = true ]; then
    echo "✓ NanoBodyBuilder2 weights already exist"
else
    echo "Downloading NanoBodyBuilder2 weights (4 models)..."
    
    for model_entry in "${NBB2_MODELS[@]}"; do
        model_name="${model_entry%%:*}"
        model_url="${model_entry#*:}"
        model_path="${NBB2_DIR}/${model_name}"
        
        if [ -f "${model_path}" ] && [ -s "${model_path}" ]; then
            echo "  ✓ ${model_name} (exists)"
        else
            echo "  Downloading ${model_name}..."
            wget -q --show-progress -O "${model_path}" "${model_url}"
            
            if [ ! -s "${model_path}" ]; then
                echo "  ERROR: Failed to download ${model_name}"
                rm -f "${model_path}"
                exit 1
            fi
            echo "  ✓ ${model_name}"
        fi
    done
    
    echo "✓ NanoBodyBuilder2 weights installed"
fi
echo ""

# ==========================================
# ESM2 Model (HuggingFace)
# ==========================================
HF_DIR="${MBER_DIR}/huggingface"
export HF_HOME="${HF_DIR}"

echo "[3/4] ESM2 Model (HuggingFace)"
echo "----------------------------------------"

if [ "$SKIP_ESM" = true ]; then
    echo "⊘ Skipped (--skip-esm flag)"
else
    ESM2_CHECK="${HF_DIR}/hub/models--facebook--esm2_t33_650M_UR50D"
    
    if [ -d "${ESM2_CHECK}" ]; then
        echo "✓ ESM2 model already exists"
    else
        echo "Downloading ESM2 model (~5GB)..."
        echo "This may take several minutes..."
        
        # Use Python to download via HuggingFace
        python3 -c "
import os
os.environ['HF_HOME'] = '${HF_DIR}'
from transformers import AutoTokenizer, EsmForMaskedLM
print('  Downloading tokenizer...')
AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D', use_safetensors=False)
print('  Downloading model weights...')
EsmForMaskedLM.from_pretrained('facebook/esm2_t33_650M_UR50D', use_safetensors=False)
print('  ✓ ESM2 model installed')
" 2>&1 | grep -v "^$"
    fi
fi
echo ""

# ==========================================
# ESMFold Model (Optional, HuggingFace)
# ==========================================
echo "[4/4] ESMFold Model (Optional)"
echo "----------------------------------------"

if [ "$WITH_ESMFOLD" = true ]; then
    ESMFOLD_CHECK="${HF_DIR}/hub/models--facebook--esmfold_v1"
    
    if [ -d "${ESMFOLD_CHECK}" ]; then
        echo "✓ ESMFold model already exists"
    else
        echo "Downloading ESMFold model (~16GB)..."
        echo "This may take a long time..."
        
        python3 -c "
import os
os.environ['HF_HOME'] = '${HF_DIR}'
from transformers import EsmForProteinFolding
print('  Downloading ESMFold model...')
EsmForProteinFolding.from_pretrained('facebook/esmfold_v1', use_safetensors=False)
print('  ✓ ESMFold model installed')
" 2>&1 | grep -v "^$"
    fi
else
    echo "⊘ Skipped (use --with-esmfold to download)"
    echo "  Note: ESMFold is only needed if using esmfold for structure prediction."
    echo "  The default VHH protocol uses NanoBodyBuilder2 instead."
fi
echo ""

# ==========================================
# Stereochemical parameters (OpenStructure)
# ==========================================
# Bond length/angle table used by AMBER relaxation. It is part of OpenStructure
# (LGPL-3.0), so mBER fetches it instead of committing it, as AlphaFold's Dockerfile
# and OpenFold's install script do: from ColabFold's byte-identical copy (which ships
# with OpenStructure's LGPL text) at a pinned commit, falling back to OpenStructure's
# own GitLab. The checksum guarantees both give the same file.
SCP_FILE="${MBER_DIR}/stereo_chemical_props.txt"
SCP_URL="https://raw.githubusercontent.com/sokrypton/ColabFold/136bec6375b8f2e13ceaeec76d3320a1590cabc1/colabfold/openstructure/stereo_chemical_props.txt"
SCP_FALLBACK="https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt"
SCP_SHA256="24510899eeb49167cffedec8fa45363a4d08279c0c637a403b452f7d0ac09451"

echo "Stereochemical parameters (OpenStructure, LGPL-3.0)"
echo "----------------------------------------"
if [ -f "${SCP_FILE}" ] && echo "${SCP_SHA256}  ${SCP_FILE}" | sha256sum -c --quiet - 2>/dev/null; then
    echo "✓ stereo_chemical_props.txt already exists"
else
    if ! { wget -q -T 30 -t 2 -O "${SCP_FILE}" "${SCP_URL}" || wget -q -T 30 -t 2 -O "${SCP_FILE}" "${SCP_FALLBACK}"; }; then
        rm -f "${SCP_FILE}"; echo "✗ could not download stereo_chemical_props.txt"; exit 1
    fi
    echo "${SCP_SHA256}  ${SCP_FILE}" | sha256sum -c --quiet - \
        || { rm -f "${SCP_FILE}"; echo "✗ checksum mismatch for stereo_chemical_props.txt"; exit 1; }
    echo "✓ stereo_chemical_props.txt installed"
fi
echo ""

# ==========================================
# Summary
# ==========================================
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo ""
echo "Weights location: ${MBER_DIR}"
echo ""
echo "Directory sizes:"
du -sh "${MBER_DIR}"/* 2>/dev/null | sed 's/^/  /'
echo ""
echo "To use with Docker, mount this directory:"
echo "  docker run --gpus all \\"
echo "    -v ${MBER_DIR}:/mber_weights:ro \\"
echo "    ..."
