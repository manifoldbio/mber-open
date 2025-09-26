from colabdesign.af.prep import _af_prep, prep_pdb, make_fixed_size, prep_pos, get_multi_id

import numpy as np

class _mber_af_prep(_af_prep):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _prep_binder(self, pdb_filename,
                   target_chain="A", binder_len=50,                                         
                   rm_target = False,
                   rm_target_seq = False,
                   rm_target_sc = False,
                   
                   # if binder_chain is defined
                   binder_chain=None,
                   rm_binder=True,
                   rm_binder_seq=True,
                   rm_binder_sc=True,
                   rm_template_ic=False,
                                      
                   hotspot=None, ignore_missing=True, **kwargs):
        '''
        MAINTAIN FUNCTIONALITY FROM COLABDESIGN, BUT ADD ABILITY TO MASK TEMPLATE IN DEFINED REGIONS

        prep inputs for binder design
        ---------------------------------------------------
        -binder_len = length of binder to hallucinate (option ignored if binder_chain is defined)
        -binder_chain = chain of binder to redesign
        -use_binder_template = use binder coordinates as template input
        -rm_template_ic = use target and binder coordinates as seperate template inputs
        -hotspot = define position/hotspots on target
        -rm_[binder/target]_seq = remove sequence info from template
        -rm_[binder/target]_sc  = remove sidechain info from template
        -ignore_missing=True - skip positions that have missing density (no CA coordinate)
        ---------------------------------------------------
        '''
        redesign = binder_chain is not None
        # rm_binder = not kwargs.pop("use_binder_template", not rm_binder) COMMENTING THIS OUT TO ALLOW FOR SELECTIVE BINDER MASKING
        
        self._args.update({"redesign":redesign})

        # get pdb info
        target_chain = kwargs.pop("chain",target_chain) # backward comp
        chains = f"{target_chain},{binder_chain}" if redesign else target_chain
        im = [True] * len(target_chain.split(",")) 
        if redesign: im += [ignore_missing] * len(binder_chain.split(","))

        self._pdb = prep_pdb(pdb_filename, chain=chains, ignore_missing=im)
        res_idx = self._pdb["residue_index"]

        if redesign:
            self._target_len = sum([(self._pdb["idx"]["chain"] == c).sum() for c in target_chain.split(",")])
            self._binder_len = sum([(self._pdb["idx"]["chain"] == c).sum() for c in binder_chain.split(",")])
        else:
            self._target_len = self._pdb["residue_index"].shape[0]
            self._binder_len = binder_len
            res_idx = np.append(res_idx, res_idx[-1] + np.arange(binder_len) + 50)
        
        self._len = self._binder_len
        self._lengths = [self._target_len, self._binder_len]

        # gather hotspot info
        if hotspot is not None:
            self.opt["hotspot"] = prep_pos(hotspot, **self._pdb["idx"])["pos"]

        if redesign:
            # binder redesign
            self._wt_aatype = self._pdb["batch"]["aatype"][self._target_len:]
            self.opt["weights"].update({"dgram_cce":1.0, "rmsd":0.0, "fape":0.0,
                                    "con":0.0, "i_con":0.0, "i_pae":0.0})
        else:
            # binder hallucination
            self._pdb["batch"] = make_fixed_size(self._pdb["batch"], num_res=sum(self._lengths))
            self.opt["weights"].update({"plddt":0.1, "con":0.0, "i_con":1.0, "i_pae":0.0})

        # configure input features
        self._inputs = self._prep_features(num_res=sum(self._lengths), num_seq=1)
        self._inputs["residue_index"] = res_idx
        self._inputs["batch"] = self._pdb["batch"]
        self._inputs.update(get_multi_id(self._lengths))

        # configure template rm masks
        (T,L,rm) = (self._lengths[0],sum(self._lengths),{})
        rm_opt = {
                "rm_template":    {"target":rm_target,    "binder":rm_binder},
                "rm_template_seq":{"target":rm_target_seq,"binder":rm_binder_seq},
                "rm_template_sc": {"target":rm_target_sc, "binder":rm_binder_sc}
                }

        for n,x in rm_opt.items():
            rm[n] = np.full(L,False)
            for m,y in x.items():
                if isinstance(y,str):
                    rm[n][prep_pos(y,**self._pdb["idx"])["pos"]] = True
                else:
                    if m == "target": rm[n][:T] = y
                    if m == "binder": rm[n][T:] = y
            
        # set template [opt]ions
        self.opt["template"]["rm_ic"] = rm_template_ic
        self._inputs.update(rm)

        self._prep_model(**kwargs)