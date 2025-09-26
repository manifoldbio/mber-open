from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

from mber.core.modules.config import (
    BaseTemplateConfig,
    BaseModelConfig,
    BaseLossConfig,
    BaseTrajectoryConfig,
    BaseEvaluationConfig,
    BaseEnvironmentConfig,
)


@dataclass
class TemplateConfig(BaseTemplateConfig):
    """VHH-specific template configuration."""

    # Override defaults specific to VHH binder design
    folding_model: Literal["nbb2", "esmfold"] = "nbb2"


@dataclass
class ModelConfig(BaseModelConfig):
    """VHH-specific model configuration."""

    # Inherit base class with default values


@dataclass
class LossConfig(BaseLossConfig):
    """VHH-specific loss configuration."""

    # VHH-specific weights
    weights_hbond: float = 2.5
    weights_salt_bridge: float = 2.0


@dataclass
class TrajectoryConfig(BaseTrajectoryConfig):
    """VHH-specific trajectory configuration."""

    # VHH-specific defaults
    soft_iters: int = 65
    temp_iters: int = 25
    hard_iters: int = 0
    pssm_iters: int = 10
    greedy_tries: int = 10
    early_stop_iptm: float = 0.7

    # Optimizer configuration with VHH-specific defaults
    optimizer_type: Literal[
        "adam", "sgd", "schedule_free_adam", "schedule_free_sgd"
    ] = "schedule_free_sgd"
    optimizer_learning_rate: float = 4e-1
    optimizer_b1: float = 0.9
    optimizer_b2: float = 0.999
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: Optional[float] = None
    optimizer_weight_lr_power: float = 2.0
    optimizer_warmup_steps: Optional[int] = None


@dataclass
class EvaluationConfig(BaseEvaluationConfig):
    """VHH-specific evaluation configuration."""

    # Override default to use nbb2 for VHH binders
    monomer_folding_model: str = "nbb2"


@dataclass
class EnvironmentConfig(BaseEnvironmentConfig):
    """VHH-specific environment configuration."""

    # Inherit base class with default values
