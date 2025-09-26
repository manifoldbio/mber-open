from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logomaker as lm


# Standard amino acids in alphabetical order (by three-letter code)
STANDARD_AAS = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]


def generate_seqlogo_from_logits(logits: np.ndarray, output_path: Path) -> None:
    """
    Generate and save a sequence logo visualization from the logits of a position weight matrix.

    Args:
        logits: numpy array of shape (sequence_length, num_amino_acids)
                with logits in alphabetical order (A-Y)
        output_path: Path to save the visualization
    """
    # Convert logits to probabilities using softmax
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

    # Generate the sequence logo
    generate_seqlogo(probabilities, output_path)


def generate_seqlogo(probabilities: np.ndarray, output_path: Path) -> None:
    """
    Generate and save a sequence logo visualization of the position weight matrix.

    Args:
        probabilities: numpy array of shape (sequence_length, num_amino_acids)
                    with probabilities in alphabetical order (A-Y)
        output_path: Path to save the visualization
    """

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
