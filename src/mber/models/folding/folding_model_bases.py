from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, List, Tuple, Union
from pathlib import Path


class ProteinFoldingModel(ABC):
    """Abstract base class for protein structure prediction models."""

    @abstractmethod
    def predict_structure(
        self,
        sequence: str,
        output_pdb_path: Path,
        **kwargs
    ) -> Path:
        """
        Predict the 3D structure of a protein sequence and save it as a PDB file.

        Args:
            sequence: Amino acid sequence
            output_pdb_path: Path to save the PDB file
            confidence_threshold: Optional threshold for model confidence

        Returns:
            Tuple of (path to saved PDB file, predicted confidence score)
        """
        pass