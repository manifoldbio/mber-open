import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict, Union
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers import EsmForMaskedLM
from transformers import BatchEncoding
from tqdm import tqdm
from pathlib import Path
import logomaker
import matplotlib.pyplot as plt
from .plm_model_bases import ProteinLanguageModel, download_from_s3

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", use_safetensors=False)

class ESM2Model(ProteinLanguageModel):
    """ESM2 implementation of the ProteinLanguageModel interface."""

    id_to_tok = {v: k for k, v in tokenizer.get_vocab().items()}
    tok_to_id = tokenizer.get_vocab()
    
    def __init__(
        self,
        pretrained_model_name_or_path: str = "facebook/esm2_t33_650M_UR50D",
        model: Optional[nn.Module] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_path = None,
    ):
        """Initialize ESM2 model and tokenizer."""
        if model is None:
            model = EsmForMaskedLM.from_pretrained(pretrained_model_name_or_path, use_safetensors=False)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_safetensors=False)

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        
        # Cache amino acid tokens for sampling
        self.aa_tokens = [self.tokenizer.convert_tokens_to_ids(aa) for aa in 
                         ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", 
                          "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]]
        
        if checkpoint_path:
            local_checkpoint_path = download_from_s3(checkpoint_path)
            checkpoint = torch.load(local_checkpoint_path)
            # checkpoint state dict contains the spurious key "esm.embeddings.position_ids". remove it.
            # checkpoint = {k: v for k, v in checkpoint.items() if not k.startswith('esm.embeddings.position_ids')}
            self.model.load_state_dict(checkpoint)
        self.model.eval()

    def get_logits(self, masked_sequence: str, as_numpy=True) -> Union[np.ndarray, torch.Tensor]:
        """
        Get logits for each position in the sequence, including masked positions.

        Args:
            masked_sequence: String containing amino acids and mask tokens (*)

        Returns:
            numpy array of shape (sequence_length, num_amino_acids) containing logits
            for each position and possible amino acid
        """
        self.model.eval()
        
        # Replace * with mask token
        sequence = masked_sequence.replace('*', self.tokenizer.mask_token)
        
        # Tokenize the sequence
        tokenized = self.tokenizer(
            sequence,
            add_special_tokens=True,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                output_hidden_states=False,
                return_dict=True
            )
        
        # Remove CLS and EOS tokens from logits and convert to numpy
        if as_numpy:
            logits = outputs.logits[0, 1:-1, :].cpu().numpy()
        else:
            logits = outputs.logits[0, 1:-1, :].cpu()
        
        return logits

    def sample_sequences(self, 
                        masked_sequence: str,
                        num_samples: int = 1,
                        temperature: float = 1.0,
                        batch_size: int = 8) -> List[str]:
        """Sample complete sequences by filling in masked positions."""
        # Find masked positions
        mask_positions = [i for i, c in enumerate(masked_sequence) if c =='*']

        # replace all '*' with mask tokens
        masked_sequence_input = masked_sequence.replace('*', self.tokenizer.mask_token)

        logits = self.get_logits(masked_sequence_input, as_numpy=False)
        
        # Get logits only for amino acid tokens at masked positions
        mask_logits = logits[mask_positions][:, self.aa_tokens]
        
        # Apply temperature
        mask_logits = mask_logits / temperature
        
        # Sample from logits
        probs = torch.softmax(mask_logits, dim=-1)
        samples = torch.multinomial(probs, num_samples=num_samples, replacement=True)
        
        # Convert samples to amino acids
        aa_list = [self.tokenizer.convert_ids_to_tokens(self.aa_tokens)[i] for i in range(20)]
        
        sequences = []
        for i in range(num_samples):
            seq_list = list(masked_sequence)
            for pos_idx, mask_pos in enumerate(mask_positions):
                seq_list[mask_pos] = aa_list[samples[pos_idx, i]]
            sequences.append(''.join(seq_list))
            
        return sequences

    # def generate_seqlogo(self,
    #                     probabilities: np.ndarray,
    #                     output_path: Path) -> None:
    #     """Generate and save a sequence logo visualization using logomaker."""
    #     # Convert probabilities to pandas DataFrame for logomaker
    #     aa_list = [self.tokenizer.convert_ids_to_tokens(self.aa_tokens)[i] for i in range(20)]
    #     df = pd.DataFrame(probabilities, columns=aa_list)
        
    #     # Create logo
    #     fig, ax = plt.subplots(figsize=(10, 3))
    #     logo = logomaker.Logo(df, ax=ax)
        
    #     # Customize logo appearance
    #     logo.style_spines(visible=False)
    #     plt.tight_layout()
        
    #     # Save figure
    #     plt.savefig(output_path, dpi=300, bbox_inches='tight')
    #     plt.close()