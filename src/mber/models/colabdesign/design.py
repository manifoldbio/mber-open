from colabdesign.af.design import _af_design
from colabdesign.shared.utils import softmax, categorical
import jax
import jax.numpy as jnp
import numpy as np

class _mber_af_design(_af_design):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _mutate(self, seq, plddt=None, logits=None, mutation_rate=1):
        '''mutate random position'''
        seq = np.array(seq)
        N,L = seq.shape

        # fix some positions
        i_prob = np.ones(L) if plddt is None else np.maximum(1-plddt,0)
        i_prob[np.isnan(i_prob)] = 0
        if "fix_pos" in self.opt:
            if "pos" in self.opt:
                p = self.opt["pos"][self.opt["fix_pos"]]
                seq[...,p] = self._wt_aatype_sub
            else:
                p = self.opt["fix_pos"]
                seq[...,p] = self._wt_aatype[...,p]
            i_prob[p] = 0

        # Use different key to avoid conflict with other ColabDesign methods
        if "fix_binder_pos" in self.opt:
            if "pos" in self.opt:
                p = self.opt["pos"][self.opt["fix_binder_pos"]]
                seq[...,p] = self._wt_aatype_sub
            else:
                p = self.opt["fix_binder_pos"]
                seq[...,p] = self._wt_aatype[...,p]
            i_prob[p] = 0
        
        for m in range(mutation_rate):
            # sample position
            # https://www.biorxiv.org/content/10.1101/2021.08.24.457549v1
            i = np.random.choice(np.arange(L),p=i_prob/i_prob.sum())

            # sample amino acid
            logits = np.array(0 if logits is None else logits)
            if logits.ndim == 3: logits = logits[:,i]
            elif logits.ndim == 2: logits = logits[i]
            a_logits = logits - np.eye(self._args["alphabet_size"])[seq[:,i]] * 1e8
            a = categorical(softmax(a_logits))

            # return mutant
            seq[:,i] = a
        
        return seq