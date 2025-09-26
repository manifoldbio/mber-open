#!/bin/bash
# Adapted from BindCraft (https://github.com/martinpacesa/BindCraft/blob/main/install_bindcraft.sh)

# Define parameters
af_dir="${HOME}/.mber/af_params"
params_file="${af_dir}/alphafold_params_2022-12-06.tar"

# Create directory structure
echo -e "Creating AlphaFold2 parameters directory at ${af_dir}"
mkdir -p "${af_dir}" || { echo -e "Error: Failed to create directory ${af_dir}. Please check permissions for your home directory."; exit 1; }

# Download AlphaFold2 weights
echo -e "Downloading AlphaFold2 model weights to ${af_dir}\n"
wget -O "${params_file}" "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar" || { 
    echo -e "Error: Failed to download AlphaFold2 weights. Please check your internet connection."; 
    echo -e "If you're behind a proxy, you may need to configure wget to use it.";
    exit 1; 
}

# Verify download completed successfully
[ -s "${params_file}" ] || { 
    echo -e "Error: Downloaded AlphaFold2 weights file is empty or missing."; 
    echo -e "Please check available disk space in your home directory.";
    exit 1; 
}

# Verify archive integrity
echo -e "Verifying archive integrity..."
tar tf "${params_file}" >/dev/null 2>&1 || { 
    echo -e "Error: The AlphaFold2 weights archive appears to be corrupt."; 
    echo -e "Please try downloading again or check if the source file has changed.";
    exit 1; 
}

# Extract weights
echo -e "Extracting AlphaFold2 weights to ${af_dir}"
tar -xf "${params_file}" -C "${af_dir}" || { 
    echo -e "Error: Failed to extract AlphaFold2 weights."; 
    echo -e "Please check available disk space and permissions for ${af_dir}.";
    exit 1; 
}

# Verify extraction completed successfully
[ -f "${af_dir}/params_model_5_ptm.npz" ] || { 
    echo -e "Error: Could not locate extracted AlphaFold2 weights files."; 
    echo -e "The extraction might have partially completed or the archive structure has changed.";
    exit 1; 
}

# Cleanup tar file to save space
echo -e "Cleaning up downloaded archive..."
rm "${params_file}" || { 
    echo -e "Warning: Failed to remove AlphaFold2 weights archive at ${params_file}."; 
    echo -e "You may want to manually remove it to save disk space.";
}

echo -e "\nAlphaFold2 weights successfully installed at: ${af_dir}"
echo -e "Total disk usage: $(du -sh ${af_dir} | cut -f1)"