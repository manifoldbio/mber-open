from mber.core.modules.trajectory import BaseTrajectoryModule
from mber_protocols.stable.VHH_binder_design.config import (
    ModelConfig,
    LossConfig,
    TrajectoryConfig,
    EnvironmentConfig
)


class TrajectoryModule(BaseTrajectoryModule):
    """
    VHH-specific trajectory module implementation.
    Inherits all functionality from BaseTrajectoryModule.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        loss_config: LossConfig,
        trajectory_config: TrajectoryConfig,
        environment_config: EnvironmentConfig,
    ) -> None:
        super().__init__(model_config, loss_config, trajectory_config, environment_config)
        
        # VHH-specific initializations can be added here
        
    # Override methods as needed for VHH-specific functionality
    # For now, we're inheriting everything from the base class