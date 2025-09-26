from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union
from enum import Enum


@dataclass
class BaseTemplateConfig:
    """Base configuration for template preparation."""
    # Hotspot selection parameters
    sasa_threshold: float = 50.0
    hotspot_strategy: Literal['top_k', 'random', 'none'] = 'random'
    
    # Truncation parameters
    pae_threshold: float = 30.0
    distance_threshold: float = 25.0
    gap_penalty: float = 10.0
    include_surrounding_context: bool = False
    
    # Sequence generation parameters
    plm_model: str = "esm2-650M"
    sampling_temperature: float = 0.1
    bias_temperature: float = 1.0
    omit_amino_acids: str = "C"
    
    # Structure parameters
    folding_model: str = "esmfold"
    target_chain: str = "A"
    binder_chain: str = "H"


@dataclass
class BaseModelConfig:
    """Base configuration for models."""
    design_models: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    use_multimer_design: bool = True
    num_recycles_design: int = 3
    eval_models: List[int] = field(default_factory=lambda: [0])
    use_multimer_eval: bool = True
    num_recycles_eval: int = 3


@dataclass
class BaseLossConfig:
    """Base configuration for loss functions."""
    weights_con_intra: float = 0.4
    weights_con_inter: float = 0.5
    weights_pae_intra: float = 0.8
    weights_pae_inter: float = 1.0
    intra_contact_number: int = 2
    intra_contact_distance: float = 14.0
    inter_contact_number: int = 2
    inter_contact_distance: float = 20.0
    weights_rg: float = 0.3
    weights_iptm: float = 0.1
    weights_plddt: float = 0.1
    weights_seq_ent: float = 0.1
    weights_hbond: float = 2.5
    weights_salt_bridge: float = 2.0


@dataclass
class BaseTrajectoryConfig:
    """Base configuration for trajectory optimization."""
    mask_binder: bool = False
    mask_binder_variable_template: bool = True
    mask_binder_seq: bool = False
    mask_binder_sidechains: bool = False
    mask_target: bool = False
    mask_target_seq: bool = False
    mask_target_sidechains: bool = False
    soft_iters: int = 65
    temp_iters: int = 25
    hard_iters: int = 0
    pssm_iters: int = 10
    greedy_tries: int = 10
    rm_aa: str = "C"
    update_esm_bias: bool = False
    early_stopping: bool = True
    early_stop_iptm: float = 0.8
    early_stop_fraction: float = 0.6

    plm_model: str = "esm2-650M"
    
    # Optimizer configuration
    optimizer_type: Literal["adam", "sgd", "schedule_free_adam", "schedule_free_sgd"] = "sgd"
    optimizer_learning_rate: float = 0.1
    optimizer_b1: float = 0.9  # Beta1 parameter (used by Adam and schedule_free)
    optimizer_b2: float = 0.999  # Beta2 parameter (used by Adam)
    optimizer_eps: float = 1e-8  # Epsilon parameter for numerical stability
    optimizer_weight_decay: Optional[float] = None  # Weight decay parameter
    optimizer_weight_lr_power: float = 2.0  # Weight learning rate power for schedule_free
    optimizer_warmup_steps: Optional[int] = None  # Warmup steps for schedule_free


@dataclass
class BaseEvaluationConfig:
    """Base configuration for evaluation."""
    # Add monomer folding model parameter
    monomer_folding_model: str = "esmfold"  # Model to use for monomer folding

    plm_model: str = "esm2-650M"

    # AMBER relax_me parameters
    use_gpu: bool = False
    max_iterations: int = 0
    tolerance: float = 2.39
    stiffness: float = 10.0
    max_outer_iterations: int = 3


@dataclass
class BaseEnvironmentConfig:
    """Base configuration for environment."""
    af_params_dir: str = "~/.mber/af_params"
    device: str = "cuda:0"