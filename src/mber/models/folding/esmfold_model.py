from .folding_model_bases import ProteinFoldingModel
from pathlib import Path
from typing import Optional, Union
import torch
import numpy as np

class ESMFoldModel(ProteinFoldingModel):
    """ESMFold implementation of the ProteinFoldingModel interface."""

    def __init__(
        self,
        model_name: str = "facebook/esmfold_v1",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        chunk_size: int = 128,
        num_recycles: int = 4,
        model = None,
    ):
        """
        Initialize ESMFold model.
        
        Args:
            model_name: Name of the ESMFold model to use from Hugging Face
            device: Device to run the model on ('cuda' or 'cpu')
            chunk_size: Maximum chunk size for processing sequences
            num_recycles: Number of recycles to use for structure prediction
            model: Optional pre-loaded model instance
        """
        self.model_name = model_name
        self.device = device
        self.chunk_size = chunk_size
        self.num_recycles = num_recycles
        
        # Load model if not provided
        if model is None:
            try:
                from transformers import EsmForProteinFolding
                print(f"Loading ESMFold model {model_name} on {device}...")
                self.model = EsmForProteinFolding.from_pretrained(model_name, use_safetensors=False)
                self.model.to(device)
            except ImportError:
                raise ImportError("Failed to import EsmForProteinFolding from transformers. "
                                 "Please install transformers: pip install transformers")
        else:
            self.model = model
            
        # Set to evaluation mode
        self.model.eval()
        
    def predict_structure(
        self,
        sequence: str,
        output_pdb_path: Union[str, Path],
        confidence_threshold: Optional[float] = None,
    ) -> Path:
        """
        Predict the 3D structure of a protein sequence and save it as a PDB file.
        
        Args:
            sequence: Amino acid sequence
            output_pdb_path: Path to save the PDB file
            confidence_threshold: Optional threshold for model confidence (unused in this implementation)
            
        Returns:
            Path to the saved PDB file
        """
        # Ensure output_pdb_path is a Path object
        if isinstance(output_pdb_path, str):
            output_pdb_path = Path(output_pdb_path)
            
        # Make parent directory if it doesn't exist
        output_pdb_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Clean sequence (remove whitespace and non-standard characters)
        clean_sequence = ''.join(c for c in sequence if c.isalpha())
        
        # Check if we need to chunk the sequence due to length
        seq_len = len(clean_sequence)
        
        # Process the sequence with the model
        with torch.no_grad():
            # Predict structure
            print(f"Predicting structure with ESMFold for sequence of length {seq_len}...")
            # Note: The transformers implementation automatically takes care of batching
            # and other details, but we could add custom chunking for very long sequences if needed
            
            outputs = self.model.infer_pdb(
                seqs=clean_sequence,
            )
            
            # Get predicted PDB as string (outputs is already a PDB string)
            pdb_str = outputs
            
        # Write the PDB file
        with open(output_pdb_path, 'w') as f:
            f.write(pdb_str)
            
        print(f"Structure prediction complete. Saved PDB to {output_pdb_path}")
        return output_pdb_path
        
    def get_confidence_scores(self, sequence: str) -> dict:
        """
        Get confidence scores for a predicted structure.
        
        Args:
            sequence: Amino acid sequence
            
        Returns:
            Dictionary with confidence metrics
        """
        # Clean sequence
        clean_sequence = ''.join(c for c in sequence if c.isalpha())
        
        with torch.no_grad():
            # Process with model
            outputs = self.model(
                sequences=[clean_sequence],
                num_recycles=self.num_recycles,
                return_confidence=True,
            )
            
            # Extract confidence metrics
            confidence_metrics = {}
            
            # pLDDT scores (per-residue confidence)
            if hasattr(outputs, "plddt"):
                confidence_metrics["plddt"] = outputs.plddt.cpu().numpy()[0]
                confidence_metrics["avg_plddt"] = float(np.mean(confidence_metrics["plddt"]))
                
            # pTM score if available
            if hasattr(outputs, "ptm"):
                confidence_metrics["ptm"] = float(outputs.ptm.cpu().numpy()[0])
                
            return confidence_metrics