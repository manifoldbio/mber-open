from mber.core.modules.evaluation import BaseEvaluationModule
from mber_protocols.stable.VHH_binder_design.config import (
    ModelConfig,
    LossConfig,
    EvaluationConfig,
    EnvironmentConfig
)


class EvaluationModule(BaseEvaluationModule):
    """
    VHH-specific evaluation module implementation.
    Inherits all functionality from BaseEvaluationModule.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        loss_config: LossConfig,
        evaluation_config: EvaluationConfig,
        environment_config: EnvironmentConfig,
    ) -> None:
        super().__init__(model_config, loss_config, evaluation_config, environment_config)
        
        # VHH-specific initializations can be added here
        
    # Override methods as needed for VHH-specific functionality
    # For now, we're inheriting everything from the base class