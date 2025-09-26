from .folding_model_bases import ProteinFoldingModel
from ImmuneBuilder import NanoBodyBuilder2
from ImmuneBuilder.NanoBodyBuilder2 import StructureModule, are_weights_ready, download_file, embed_dim, model_urls

import os
from typing import Optional, List, Tuple, Union
from pathlib import Path
import torch


class NBB2Model(ProteinFoldingModel):
    """NanoBodyBuilder2 implementation of the ProteinFoldingModel interface."""

    def __init__(
        self,
        model: NanoBodyBuilder2 = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        weights_dir: Optional[str] = "~/.mber/nbb2_weights",
    ):
        """Initialize NanoBodyBuilder2 model."""
        # if weights dir does not exist, create it
        weights_dir = os.path.expanduser(weights_dir)
        os.makedirs(weights_dir, exist_ok=True)

        if model is None:
            model = NanoBodyBuilder2(numbering_scheme='raw', weights_dir=weights_dir)
        self.model = model
        self.device = device

    def predict_structure(
        self,
        sequence: Union[str, dict],
        output_pdb_path: Union[str, Path],
    ) -> str:
        """
        Predict the 3D structure of a protein sequence and save it as a PDB file.
        
        Args:
            sequence: Amino acid sequence as string or dict ({"H": sequence})
            output_pdb_path: Path to save the PDB file
        
        Returns:
            Path to the saved PDB file
        """
        # Ensure output path is a string
        output_pdb_path = str(output_pdb_path)
        
        # Convert string sequence to dictionary format if needed
        if isinstance(sequence, str):
            sequence = {"H": sequence}
            
        # NBB2 expects dict format
        if not isinstance(sequence, dict):
            raise ValueError("Sequence must be a string or dictionary with chain as key")
            
        with torch.no_grad():
            nanobody = self.model.predict(sequence)
            
        nanobody.save(output_pdb_path)

        return output_pdb_path