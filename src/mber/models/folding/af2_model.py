from .folding_model_bases import ProteinFoldingModel
from typing import Optional, List, Tuple, Union
from pathlib import Path
from colabdesign import mk_afdesign_model, clear_mem
import torch
import re


class AF2Model(ProteinFoldingModel):
    """ColabDesign implementation of AlphaFold2 in the ProteinFoldingModel interface."""

    def __init__(
        self,
        model = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize AF2 model (ColabDesign)."""
        self.device = device
        self.model = mk_afdesign_model(
                            protocol="hallucination",
                            use_templates=False,
                            initial_guess=False,
                            use_initial_atom_pos=False,
                            num_recycles=3,
                            data_dir='~/.mber/af_params',
                            use_multimer=False,
                        )

    def predict_structure(
        self,
        sequence: str,
        output_pdb_path: Path,
    ) -> Path:
        """Predict the 3D structure of a protein sequence and save it as a PDB file."""
        if '|' in sequence:
            raise ValueError("AF2Model does not currently support multiple chains.")

        sequence = re.sub("[^A-Z]", "", sequence.upper())
        self.model.set_seq(sequence)
        print("Folding sequence with AF2 model (ColabDesign)...")
        self.model.predict(models=[0], num_recycles=3, verbose=False)
        self.model.save_pdb(output_pdb_path)

        return output_pdb_path