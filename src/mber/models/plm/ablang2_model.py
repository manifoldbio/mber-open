import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Union
import ablang2
from ablang2.models.ablang2.vocab import ablang_vocab as TOK_TO_ID
from tqdm import tqdm
import numpy as np
from pathlib import Path
from .plm_model_bases import ProteinLanguageModel, download_from_s3
import os
from urllib.parse import urlparse


ID_TO_TOK = {v: k for k, v in TOK_TO_ID.items()}

class AbLang2Model(ProteinLanguageModel):
    """AbLang2 implementation of the ProteinLanguageModel interface."""

    tok_to_id = TOK_TO_ID
    id_to_tok = ID_TO_TOK
    
    def __init__(
        self,
        model_to_use: str = "ablang2-paired",
        model: Optional[nn.Module] = None,
        tokenizer: Optional[nn.Module] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        ncpu: int = 1,
        random_init: bool = False,
        checkpoint_path: Optional[str] = None,
    ):
        """Initialize AbLang2 model and tokenizer."""
        pretrained = ablang2.pretrained(
            model_to_use=model_to_use,
            random_init=random_init,
            ncpu=ncpu,
            device=device,
        )
        
        self.model = model if model is not None else pretrained.AbLang
        self.tokenizer = tokenizer if tokenizer is not None else pretrained.tokenizer
        self.device = device
        self.pretrained = pretrained

        if checkpoint_path:
            print("Loading model checkpoint...")
            local_checkpoint_path = download_from_s3(checkpoint_path)
            checkpoint = torch.load(local_checkpoint_path)
            checkpoint = {k: v for k, v in checkpoint.items() if k.startswith('ablang.')}
            checkpoint = {k.replace('ablang.', ''): v for k, v in checkpoint.items()}
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        
    def _ablang_forward(
        self,
        tokens: torch.Tensor,
        return_attn_weights: bool = False,
        return_rep_layers: List[int] = [],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Internal method for model forward pass."""
        # if tokens is 1 dimensional, add batch dimension
        if len(tokens.shape) == 1:
            tokens = tokens.unsqueeze(0)
        representations = self.model.AbRep(tokens, return_attn_weights, return_rep_layers)
        logits = self.model.AbHead(representations.last_hidden_states)
        return representations.last_hidden_states, logits

    def _ensure_pipe(self, sequence: str) -> str:
        """Ensure sequence ends with a pipe if it doesn't contain one."""
        # if sequence contains (G4S)3, assume it is scFv and split it into H and L chains
        if "GGGGSGGGGSGGGGS" in sequence:
            sequence = sequence.split("GGGGSGGGGSGGGGS")
            sequence = {"L": sequence[0], "H": sequence[1]}
            sequence = f"{sequence['H']}|{sequence['L']}"
        return sequence if '|' in sequence else f"{sequence}|"

    def get_logits(
        self,
        masked_sequence: str,
        as_numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Get logits for each position in the sequence.

        Args:
            sequence: String containing the sequence to process
            as_numpy: If True, return numpy array; if False, return torch tensor

        Returns:
            Logits for each position as either numpy array or torch tensor
            of shape (sequence_length, vocab_size)
        """
        # Add pipes if needed
        processed_seq = self._ensure_pipe(masked_sequence)
        
        # Tokenize the sequence
        tokens = self.tokenizer(
            processed_seq, 
            w_extra_tkns=False, 
            device=self.device, 
            pad=True
        ).to(self.device)
        
        # Get logits
        with torch.no_grad():
            _, logits = self._ablang_forward(tokens=tokens.squeeze(0).squeeze(0))
            # Remove the pipe token from logits if it is the last character
            if processed_seq[-1] == '|':
                logits = logits[0][:-1]  # Take first sequence and remove last position
            else:
                logits = logits[0]
            
        if as_numpy:
            return logits.cpu().numpy()
        return logits

    def sample_sequences(
        self,
        masked_sequence: str,
        num_samples: int = 1,
        temperature: float = 1.0,
        **kwargs
    ) -> List[str]:
        """Sample complete sequences by filling in masked positions."""
        # Get logits for the masked sequence
        logits = self.get_logits(
            masked_sequence,
            as_numpy=False
        )
        
        # Find masked positions
        mask_positions = [i for i, c in enumerate(masked_sequence) if c == '*']
        
        # Get relevant logits and apply temperature
        mask_logits = logits[mask_positions] / temperature
        
        # Sample from logits
        probs = torch.softmax(mask_logits, dim=-1)
        samples = torch.multinomial(probs, num_samples=num_samples, replacement=True)
        
        # Convert to sequences
        sequences = []
        for i in range(num_samples):
            seq_list = list(masked_sequence)
            for pos_idx, mask_pos in enumerate(mask_positions):
                # Convert token ID to amino acid using model's vocabulary
                token = samples[pos_idx, i].item()
                aa = ID_TO_TOK[token]
                seq_list[mask_pos] = aa
            sequences.append(''.join(seq_list))
            
        return sequences

    def get_sequence_embeddings(
        self,
        sequences: Union[str, List[str]],
        reduction: str = "mean",
        batch_size: int = 8,
        as_numpy: bool = True,
        **kwargs
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """Get embeddings for sequences."""
        if isinstance(sequences, str):
            sequences = [sequences]
            
        # Ensure all sequences have pipes
        processed_seqs = [self._ensure_pipe(seq) for seq in sequences]
            
        # Group by length and tokenize
        embeddings_dict = {}
        with torch.no_grad():
            for sequence, orig_seq in zip(processed_seqs, sequences):
                tokenized = self.tokenizer(
                    sequence, w_extra_tkns=False, device=self.device, pad=True
                ).to(self.device)
                
                # Get embeddings
                embeddings, _ = self._ablang_forward(tokenized)
                
                # Apply reduction
                if reduction == "mean":
                    embeddings = embeddings.mean(dim=1)
                elif reduction == "sum":
                    embeddings = embeddings.sum(dim=1)
                
                if as_numpy:
                    embeddings_dict[orig_seq] = embeddings.cpu().numpy()
                else:
                    embeddings_dict[orig_seq] = embeddings
                    
        return embeddings_dict