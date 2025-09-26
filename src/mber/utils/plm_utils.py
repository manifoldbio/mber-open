from typing import Optional, List, Tuple, Dict, Union
import numpy as np

from colabdesign.shared.model import aa_order

from mber.models.plm import ProteinLanguageModel
from mber.core.data.state import TemplateData


def generate_sequence_from_mask(
    model: ProteinLanguageModel,
    masked_sequence: str,
    temperature: float = 0.1,
    num_samples: int = 1,
) -> str:
    """Generate full sequence from masked sequence using a protein language model."""
    # Sample from model to fill masked positions
    sequences = model.sample_sequences(
        masked_sequence, num_samples=num_samples, temperature=temperature
    )

    # Return first sequence
    return sequences[0]


def generate_bias_from_mask(
    model: ProteinLanguageModel,
    template_data: TemplateData,
    omit_aas: str = None,
    temperature=1.0,
) -> np.ndarray:
    """Generate position-specific bias from PLM for masked sequence."""
    # Get logits for the masked sequence
    bias = model.get_logits(masked_sequence=template_data.masked_binder_seq)
    bias = model.reorder_seq_array(bias, aa_order)

    # scale logits by temperature
    bias = bias / temperature

    if omit_aas:
        # if comma-separated, make list
        if "," in omit_aas:
            omit_aas = omit_aas.split(",")
        elif len(omit_aas) > 1:
            omit_aas = list(omit_aas)
        else:
            omit_aas = [omit_aas]
        # Set logits for omitted amino acids in flexible positions to a very negative value
        flex_pos = template_data.get_flex_pos(as_array=True) - 1
        for aa in omit_aas:
            aa_index = aa_order[aa]
            bias[flex_pos, aa_index] = -1e6

    # Add fixation bias
    try:
        bias = bias + template_data.get_fix_bias()
    except:
        pass

    return bias


def generate_bias_unmasked(
    unmasked_seq: str,
    model: ProteinLanguageModel,
    template_data: TemplateData,
    omit_aas: str = None,
) -> np.ndarray:
    """Generate position-specific bias from PLM for unmasked sequence. Use masked sequence for bias fixing"""
    # Get logits for the unmasked sequence
    bias = model.get_logits(masked_sequence=unmasked_seq)
    bias = model.reorder_seq_array(bias, aa_order)

    if omit_aas:
        # if comma-separated, make list
        if "," in omit_aas:
            omit_aas = omit_aas.split(",")
        elif len(omit_aas) > 1:
            omit_aas = list(omit_aas)
        else:
            omit_aas = [omit_aas]
        # Set logits for omitted amino acids in flexible positions to a very negative value
        flex_pos = template_data.get_flex_pos(as_array=True) - 1
        for aa in omit_aas:
            aa_index = aa_order[aa]
            bias[flex_pos, aa_index] = -1e6

    # Add fixation bias from masked sequence
    bias = bias + template_data.get_fix_bias()

    return bias
