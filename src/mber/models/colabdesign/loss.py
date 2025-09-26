import numpy as np
import jax
import jax.numpy as jnp
from colabdesign.af.alphafold.common import residue_constants
from colabdesign.af.loss import _af_loss, get_ptm, mask_loss, get_dgram_bins, _get_con_loss


class _mber_af_loss(_af_loss):
    """Custom loss class for colabdesign, inheriting from _af_loss."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


###
# `add_rg_loss`, `add_i_ptm_loss`, `add_helix_loss`, `add_termini_distance_loss` are adapted from BindCraft: (https://github.com/martinpacesa/BindCraft/blob/main/functions/colabdesign_utils.py)
###

# Define radius of gyration loss for colabdesign
def add_rg_loss(self, weight=0.1):
    """add radius of gyration loss"""

    def loss_fn(inputs, outputs):
        xyz = outputs["structure_module"]
        ca = xyz["final_atom_positions"][:, residue_constants.atom_order["CA"]]
        ca = ca[-self._binder_len :]
        rg = jnp.sqrt(jnp.square(ca - ca.mean(0)).sum(-1).mean() + 1e-8)
        rg_th = 2.38 * ca.shape[0] ** 0.365

        rg = jax.nn.elu(rg - rg_th)
        return {"rg": rg}

    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["rg"] = weight


# Define interface pTM loss for colabdesign
def add_i_ptm_loss(self, weight=0.1):
    def loss_iptm(inputs, outputs):
        p = 1 - get_ptm(inputs, outputs, interface=True)
        i_ptm = mask_loss(p)
        return {"i_ptm": i_ptm}

    self._callbacks["model"]["loss"].append(loss_iptm)
    self.opt["weights"]["i_ptm"] = weight


# add helicity loss
def add_helix_loss(self, weight=0):
    def binder_helicity(inputs, outputs):
        if "offset" in inputs:
            offset = inputs["offset"]
        else:
            idx = inputs["residue_index"].flatten()
            offset = idx[:, None] - idx[None, :]

        # define distogram
        dgram = outputs["distogram"]["logits"]
        dgram_bins = get_dgram_bins(outputs)
        mask_2d = np.outer(
            np.append(np.zeros(self._target_len), np.ones(self._binder_len)),
            np.append(np.zeros(self._target_len), np.ones(self._binder_len)),
        )

        x = _get_con_loss(dgram, dgram_bins, cutoff=6.0, binary=True)
        if offset is None:
            if mask_2d is None:
                helix_loss = jnp.diagonal(x, 3).mean()
            else:
                helix_loss = jnp.diagonal(x * mask_2d, 3).sum() + (
                    jnp.diagonal(mask_2d, 3).sum() + 1e-8
                )
        else:
            mask = offset == 3
            if mask_2d is not None:
                mask = jnp.where(mask_2d, mask, 0)
            helix_loss = jnp.where(mask, x, 0.0).sum() / (mask.sum() + 1e-8)

        return {"helix": helix_loss}

    self._callbacks["model"]["loss"].append(binder_helicity)
    self.opt["weights"]["helix"] = weight


# add N- and C-terminus distance loss
def add_termini_distance_loss(self, weight=0.1, threshold_distance=7.0):
    """Add loss penalizing the distance between N and C termini"""

    def loss_fn(inputs, outputs):
        xyz = outputs["structure_module"]
        ca = xyz["final_atom_positions"][:, residue_constants.atom_order["CA"]]
        ca = ca[-self._binder_len :]  # Considering only the last _binder_len residues

        # Extract N-terminus (first CA atom) and C-terminus (last CA atom)
        n_terminus = ca[0]
        c_terminus = ca[-1]

        # Compute the distance between N and C termini
        termini_distance = jnp.linalg.norm(n_terminus - c_terminus)

        # Compute the deviation from the threshold distance using ELU activation
        deviation = jax.nn.elu(termini_distance - threshold_distance)

        # Ensure the loss is never lower than 0
        termini_distance_loss = jax.nn.relu(deviation)
        return {"NC": termini_distance_loss}

    # Append the loss function to the model callbacks
    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["NC"] = weight


import jax
import jax.numpy as jnp
from colabdesign.af.alphafold.common import residue_constants
from colabdesign.af.loss import get_dgram_bins

def add_hbond_loss(self, weight=0.5):
    """
    Add hydrogen bonding loss to encourage H-bonds at the interface.
    
    This loss function rewards close contacts between polar residues
    at distances appropriate for hydrogen bonding.
    """
    
    def loss_fn(inputs, outputs, aux):
        # Get residue indices for target and binder
        target_len = self._target_len
        binder_len = self._binder_len
        
        # Get residue types
        target_aatype = jnp.clip(inputs["aatype"][:target_len], 0, 19)
        binder_aatype = jnp.clip(inputs["aatype"][target_len:], 0, 19)
        
        # Define residue polarity (probability of forming H-bonds)
        # Based on sidechain properties and involvement in H-bonding
        aa_polarity = jnp.array([
            0.2,  # A - Alanine (low)
            1.0,  # R - Arginine (high)
            0.9,  # N - Asparagine (high)
            0.9,  # D - Aspartic acid (high)
            0.2,  # C - Cysteine (low)
            0.9,  # Q - Glutamine (high)
            0.9,  # E - Glutamic acid (high)
            0.2,  # G - Glycine (low)
            0.8,  # H - Histidine (medium-high)
            0.1,  # I - Isoleucine (very low)
            0.1,  # L - Leucine (very low)
            1.0,  # K - Lysine (high)
            0.2,  # M - Methionine (low)
            0.2,  # F - Phenylalanine (low)
            0.3,  # P - Proline (low-medium)
            0.8,  # S - Serine (medium-high)
            0.8,  # T - Threonine (medium-high)
            0.5,  # W - Tryptophan (medium)
            0.7,  # Y - Tyrosine (medium-high)
            0.1,  # V - Valine (very low)
        ])
        
        # Get polarity scores for each residue
        target_polarity = aa_polarity[target_aatype]  # [target_len]
        binder_polarity = aa_polarity[binder_aatype]  # [binder_len]
        
        # Get residue pair distances from distogram
        dist_logits = outputs["distogram"]["logits"]  # [L, L, num_bins]
        dist_bins = get_dgram_bins(outputs)  # Array of distance bins
        
        # Convert distogram logits to expected distances
        dist_probs = jax.nn.softmax(dist_logits, axis=-1)  # [L, L, num_bins]
        expected_dist = jnp.sum(dist_probs * dist_bins[None, None, :], axis=-1)  # [L, L]
        
        # Extract interface distances
        interface_dist = expected_dist[:target_len, target_len:]  # [target_len, binder_len]
        
        # H-bond distance scoring function (Gaussian centered at optimal H-bond distance)
        # For C-alpha distances, H-bonds typically occur around 5-8Å
        optimal_dist = 6.5  # Å (for C-alpha distances)
        dist_std = 1.5      # Å (wider to account for C-alpha approximation)
        dist_score = jnp.exp(-((interface_dist - optimal_dist) ** 2) / (2 * dist_std ** 2))
        
        # Weight by residue polarity (higher score for polar-polar interactions)
        polarity_weight = target_polarity[:, None] * binder_polarity[None, :]  # [target_len, binder_len]
        weighted_score = dist_score * polarity_weight
        
        # Use mean of top 10% of scores (avoiding dynamic slicing)
        # First compute a threshold value using percentile
        flat_scores = weighted_score.flatten()
        num_elements = flat_scores.shape[0]
        k = num_elements // 10  # Use top 10% (static computation)
        k = jnp.maximum(k, 1)   # Ensure at least one element
        
        # Use topk instead of sort+slice (JAX-friendly)
        top_k_values, _ = jax.lax.top_k(flat_scores, k=10)  # Use fixed k=10
        
        # Return negative mean of top k scores (negative because we're minimizing)
        hbond_loss = -jnp.mean(top_k_values)
        
        aux.update({"hbond": hbond_loss})

        return {"hbond": hbond_loss}
    
    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["hbond"] = weight


def add_salt_bridge_loss(self, weight=0.3):
    """
    Add salt bridge loss to encourage complementary charged residues at the interface.
    Salt bridges are electrostatic interactions between oppositely charged residues.
    """
    
    def loss_fn(inputs, outputs, aux):
        # Get residue indices for target and binder
        target_len = self._target_len
        binder_len = self._binder_len
        
        # Get residue types
        target_aatype = jnp.clip(inputs["aatype"][:target_len], 0, 19)
        binder_aatype = jnp.clip(inputs["aatype"][target_len:], 0, 19)
        
        # Define residue charge properties (+1 positive, -1 negative, 0 neutral)
        aa_charge = jnp.array([
            0,   # A - Alanine (neutral)
            1,   # R - Arginine (positive)
            0,   # N - Asparagine (neutral)
            -1,  # D - Aspartic acid (negative)
            0,   # C - Cysteine (neutral)
            0,   # Q - Glutamine (neutral)
            -1,  # E - Glutamic acid (negative)
            0,   # G - Glycine (neutral)
            0.5, # H - Histidine (sometimes positive)
            0,   # I - Isoleucine (neutral)
            0,   # L - Leucine (neutral)
            1,   # K - Lysine (positive)
            0,   # M - Methionine (neutral)
            0,   # F - Phenylalanine (neutral) 
            0,   # P - Proline (neutral)
            0,   # S - Serine (neutral)
            0,   # T - Threonine (neutral)
            0,   # W - Tryptophan (neutral)
            0,   # Y - Tyrosine (neutral)
            0,   # V - Valine (neutral)
        ])
        
        # Get charge scores for each residue
        target_charge = aa_charge[target_aatype]
        binder_charge = aa_charge[binder_aatype]
        
        # Get residue pair distances from distogram
        dist_logits = outputs["distogram"]["logits"]
        dist_bins = get_dgram_bins(outputs)
        
        # Convert distogram logits to expected distances
        dist_probs = jax.nn.softmax(dist_logits, axis=-1)
        expected_dist = jnp.sum(dist_probs * dist_bins[None, None, :], axis=-1)
        
        # Extract interface distances
        interface_dist = expected_dist[:target_len, target_len:]
        
        # Salt bridge distance scoring (Gaussian centered at optimal salt bridge distance)
        optimal_dist = 8.0  # Å (for C-alpha distances of charged residues)
        dist_std = 2.0      # Å
        dist_score = jnp.exp(-((interface_dist - optimal_dist) ** 2) / (2 * dist_std ** 2))
        
        # Favor opposite charges (product of charges should be negative)
        charge_product = target_charge[:, None] * binder_charge[None, :]
        charge_score = -jnp.minimum(charge_product, 0)  # Only keep negative products (opposite charges)
        
        # Weight by charge complementarity and distance
        weighted_score = dist_score * charge_score
        
        # Use mean of top values with fixed k
        flat_scores = weighted_score.flatten()
        top_values, _ = jax.lax.top_k(flat_scores, k=10)
        
        # Return negative mean (negative because we're minimizing)
        salt_bridge_loss = -jnp.mean(top_values)
        
        aux.update({"salt_bridge": salt_bridge_loss})

        return {"salt_bridge": salt_bridge_loss}
    
    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["salt_bridge"] = weight
