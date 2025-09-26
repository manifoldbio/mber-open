from typing import List, Dict, Optional, Union, Literal
from dataclasses import dataclass, field
from copy import deepcopy as copy_dict
from colabdesign.shared.model import order_aa
from colabdesign import mk_af_model

from mber.models.colabdesign.design import _mber_af_design
from mber.models.colabdesign.prep import _mber_af_prep
from mber.models.colabdesign.loss import _mber_af_loss
from mber.models.colabdesign.inputs import _mber_af_inputs
from mber.models.colabdesign.utils import _mber_af_utils

import os
import jax
import optax
import numpy as np
from numpy.typing import NDArray

class AFModel(mk_af_model, _mber_af_design, _mber_af_prep, _mber_af_loss, _mber_af_inputs, _mber_af_utils):
    """Custom model class for colabdesign, inheriting from mk_af_model and other components."""
    def __init__(self, *args, **kwargs):
        # expand user path for data_dir
        if 'data_dir' in kwargs:
            kwargs['data_dir'] = os.path.expanduser(kwargs['data_dir'])
        super().__init__(*args, **kwargs)

    def get_trajectory_metrics(self) -> List[dict]:
        return self._tmp["log"]
    
    def get_trajectory_seqs(self, as_str: bool=True) -> Union[List[str], NDArray]:
        seqs_order = np.argmax(np.array(self._tmp['traj']['seq']), axis=-1).squeeze(axis=1)

        if as_str:
            vectorized_order_aa = np.vectorize(lambda x: order_aa[int(x)])
            seqs_aa = vectorized_order_aa(seqs_order)
            return [''.join(seq) for seq in seqs_aa]
        else:
            return seqs_order
            
    def set_optimizer(
        self, 
        optimizer: Optional[str] = None, 
        learning_rate: Optional[float] = None, 
        norm_seq_grad: Optional[bool] = None, 
        **kwargs
    ):
        '''
        Set optimizer with support for both standard and schedule-free optimizers.
        Treats all optimizer types uniformly without special evaluation handling.
        
        Args:
            optimizer: Optimizer type ('adam', 'sgd', 'adamw', 'schedule_free_adam', 'schedule_free_sgd')
            learning_rate: Learning rate
            norm_seq_grad: Whether to normalize sequence gradients
            **kwargs: Optimizer-specific parameters
        '''
        from mber.models.colabdesign.optimizer import get_optimizer
        
        # Set defaults
        if optimizer is None: 
            optimizer = self._args["optimizer"]
        if learning_rate is not None: 
            self.opt["learning_rate"] = learning_rate
        if norm_seq_grad is not None: 
            self.opt["norm_seq_grad"] = norm_seq_grad
        
        # Map common parameter names to optax names
        param_mapping = {
            'beta1': 'b1',
            'beta2': 'b2',
            'epsilon': 'eps'
        }
        
        # Convert parameter names if needed
        for common_name, optax_name in param_mapping.items():
            if common_name in kwargs and optax_name not in kwargs:
                kwargs[optax_name] = kwargs.pop(common_name)
        
        # Get optimizer
        optimizer_obj = get_optimizer(
            optimizer_type=optimizer,
            learning_rate=self.opt["learning_rate"],
            **kwargs
        )
        
        # Initialize optimizer state
        self._state = optimizer_obj.init(self._params)
        
        # Define optimizer update function
        def update_grad(state, grad, params):
            updates, state = optimizer_obj.update(grad, state, params)
            grad = jax.tree_map(lambda x: -x, updates)
            return state, grad
        
        self._optimizer = jax.jit(update_grad)