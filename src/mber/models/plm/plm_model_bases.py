from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Optional, List, Tuple, Union
from pathlib import Path
import os
from urllib.parse import urlparse
import boto3

# Standard amino acids in alphabetical order (by three-letter code)
STANDARD_AAS = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]


def download_from_s3(s3_path: str) -> str:
    """Download file from S3 to local NVMe storage."""
    parsed_url = urlparse(s3_path)
    bucket = parsed_url.netloc
    key = parsed_url.path.lstrip("/")

    local_path = f"/tmp/s3/{bucket}/{key}"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if not os.path.exists(local_path):
        print(f"Downloading {s3_path} to {local_path}")
        s3 = boto3.client("s3")
        s3.download_file(bucket, key, local_path)
    else:
        print(f"Using cached file at {local_path}")

    return local_path


class ProteinLanguageModel(ABC):
    """Abstract base class for protein language models that can process masked sequences."""

    @property
    @abstractmethod
    def id_to_tok(self) -> dict:
        """Mapping from token IDs to amino acid characters."""
        pass

    @property
    @abstractmethod
    def tok_to_id(self) -> dict:
        """Mapping from amino acid characters to token IDs."""
        pass

    @abstractmethod
    def get_logits(self, masked_sequence: str) -> Union[np.ndarray, torch.Tensor]:
        """
        Get logits for each position in the sequence, including masked positions.

        Args:
            masked_sequence: String containing amino acids and mask tokens (*)

        Returns:
            numpy array of shape (sequence_length, num_amino_acids) containing logits
            for each position and possible amino acid
        """
        pass

    def get_probabilities(
        self,
        masked_sequence: str,
        alphabetic: bool = True,
        temperature: float = 1.0,
        enforce_framework: bool = True,
        omit_AAs: str = "",
    ) -> np.ndarray:
        """
        Get probability distribution over amino acids for each position.

        Args:
            masked_sequence: String containing amino acids and mask tokens (*)
            alphabetic: If True, reorganize output probabilities to be in alphabetical order (A-Y)

        Returns:
            numpy array of shape (sequence_length, 20) containing probabilities
            for each position and the 20 standard amino acids
        """
        import numpy as np

        aa_to_pos = {aa: idx for idx, aa in enumerate(STANDARD_AAS)}

        # Find non-masked (framework) positions
        framework_positions = [i for i, c in enumerate(masked_sequence) if c != "*"]

        # Get list of omitted AAs
        if omit_AAs != "":
            print(f"Omitting amino acids: {omit_AAs} from bias pwm.")
            omitted_aas = list(omit_AAs)
        else:
            omitted_aas = []

        # Get logits and ensure it's a numpy array
        logits = np.asarray(self.get_logits(masked_sequence))

        # Get indices for the 20 standard amino acids in the model's vocabulary
        aa_indices = np.array([self.tok_to_id[aa] for aa in STANDARD_AAS])

        # Filter logits to only include standard amino acids
        filtered_logits = logits[:, aa_indices]

        # subtract a large number from the logits of the omitted AAs
        for aa in omitted_aas:
            aa_idx = self.tok_to_id[aa]
            filtered_logits[:, aa_indices == aa_idx] -= 1e6

        # Apply temperature scaling
        filtered_logits /= temperature

        # Apply softmax over the amino acid dimension
        exp_logits = np.exp(
            filtered_logits - np.max(filtered_logits, axis=-1, keepdims=True)
        )
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

        if not alphabetic:
            # If not alphabetic, rearrange to match the model's internal order
            model_order = [self.id_to_tok[idx] for idx in aa_indices]
            new_order = [aa_to_pos[aa] for aa in model_order]
            probs = probs[:, new_order]

        # Forcibly set the probabilities at framework positions to one-hot
        if enforce_framework:
            for pos in framework_positions:
                probs[pos] = 0.0 # USE ALL ZEROS TO INDICATE DIFFERENT TOKEN
                if masked_sequence[pos] not in STANDARD_AAS:
                    continue # Skip non-standard amino acids, mostly for dealing with '|' character for now
                probs[pos, aa_to_pos[masked_sequence[pos]]] = 1.0
                
        return probs
    
    def sample_sequences_from_pwm(
            self, probabilities: np.ndarray, num_samples: int = 1
    ) -> List[str]:
        """
        Sample complete sequences from a position weight matrix.

        Args:
            probabilities: numpy array of shape (sequence_length, num_amino_acids)
            num_samples: Number of sequences to sample

        Returns:
            List of complete sequences with masks filled in
        """
        # Get the number of amino acids
        num_amino_acids = probabilities.shape[1]

        # Sample sequences
        sampled_sequences = []
        for _ in range(num_samples):
            sampled_sequence = ""
            for pos in range(probabilities.shape[0]):
                # if all probabilities are zero, use a '|' character
                if np.all(probabilities[pos] == 0):
                    sampled_sequence += "|"
                    continue
                # Sample an amino acid from the distribution
                aa = np.random.choice(num_amino_acids, p=probabilities[pos])
                sampled_sequence += STANDARD_AAS[aa]
            sampled_sequences.append(sampled_sequence)

        return sampled_sequences

    @abstractmethod
    def sample_sequences(
        self, masked_sequence: str, num_samples: int = 1, temperature: float = 1.0
    ) -> List[str]:
        """
        Sample complete sequences by filling in masked positions.

        Args:
            masked_sequence: String containing amino acids and mask tokens (*)
            num_samples: Number of sequences to sample
            temperature: Temperature parameter for sampling (higher = more diverse)

        Returns:
            List of complete sequences with masks filled in
        """
        pass

    def save_position_weight_matrix(
        self, probabilities: np.ndarray, output_path: Path
    ) -> None:
        """
        Save position weight matrix to a file.

        Args:
            probabilities: numpy array of shape (sequence_length, num_amino_acids)
            output_path: Path to save the matrix
        """
        np.save(output_path, probabilities)

    def generate_seqlogo(self, probabilities: np.ndarray, output_path: Path) -> None:
        """
        Generate and save a sequence logo visualization of the position weight matrix.

        Args:
            probabilities: numpy array of shape (sequence_length, num_amino_acids)
                        with probabilities in alphabetical order (A-Y)
            output_path: Path to save the visualization
        """
        import matplotlib.pyplot as plt
        import pandas as pd
        import logomaker as lm

        # Convert the probability matrix to a DataFrame that logomaker expects
        # Each row should be a position, each column should be an amino acid
        df = pd.DataFrame(
            probabilities,  # No transpose - each row is a position
            columns=STANDARD_AAS,  # Amino acids as columns
        )

        # Create figure with appropriate size
        seq_length = probabilities.shape[0]
        width = max(12, seq_length / 8)
        plt.figure(figsize=(width, 4))

        # Create Logo object
        logo = lm.Logo(
            df,
            shade_below=0.2,
            fade_below=0.01,
            font_name="DejaVu Sans",
            # color_scheme='skylign_protein'
        )

        # Customize the logo appearance
        logo.style_spines(visible=False)
        logo.style_spines(spines=["left", "bottom"], visible=True)

        # Set up the axis labels and ticks
        ax = logo.ax

        # For long sequences, show fewer tick labels to avoid overcrowding
        tick_spacing = max(1, seq_length // 20)  # Show ~20 ticks maximum
        tick_positions = range(0, seq_length, tick_spacing)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(i + 1) for i in tick_positions])

        # Add labels
        ax.set_xlabel("Position", fontsize=12, labelpad=10)
        ax.set_ylabel("Probability", fontsize=12, labelpad=10)

        # Set y-axis limits
        ax.set_ylim(0, 1.0)

        # Rotate x-axis labels for better readability in long sequences
        plt.xticks(rotation=45, ha="right")

        # Adjust layout
        plt.tight_layout()

        # Save the figure
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def get_unmask_cce_score(
        self,
        sequence: str,
    ):
        """
        Get the unmasked cross-entropy score summed across a given sequence.
        Taking Sergey's name for this score from this thread (https://github.com/ntranoslab/esm-variants/issues/12).
        """

        logits = self.get_logits(masked_sequence=sequence)
        logits = torch.tensor(logits)
        
        log_likelihood = torch.log_softmax(logits, dim=-1)

        tokenized_input = [self.tok_to_id[aa] for aa in sequence]

        unmask_cce_score = 0
        for j, token in enumerate(tokenized_input):
            unmask_cce_score += log_likelihood[j, token].item()
            
        return unmask_cce_score

    def reorder_seq_array(self, seq_array: np.ndarray, aa_order: dict) -> np.ndarray:
        """
        Reorder a sequence array to match a provided order of amino acids.

        Args:
            seq_array: numpy array of shape (sequence_length, num_amino_acids)
            aa_order: dictionary mapping amino acid characters to their indices (for the output array)

        Returns:
            numpy array reordered to match provided amino acid order
        """
        
        # Create an empty array to hold the reordered sequence
        reordered_array = np.zeros((seq_array.shape[0], len(aa_order)))
        for aa, idx in aa_order.items():
            reordered_array[:, idx] = seq_array[:, self.tok_to_id[aa]]
        return reordered_array