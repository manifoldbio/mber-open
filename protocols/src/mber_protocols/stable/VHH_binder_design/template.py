from mber.core.modules.template import BaseTemplateModule
from mber_protocols.stable.VHH_binder_design.config import TemplateConfig, EnvironmentConfig


class TemplateModule(BaseTemplateModule):
    """
    VHH-specific template module implementation.
    Inherits all functionality from BaseTemplateModule.
    """
    
    def __init__(
        self,
        template_config: TemplateConfig,
        environment_config: EnvironmentConfig,
        verbose: bool = True,
    ) -> None:
        super().__init__(template_config, environment_config, verbose)
        
        # VHH-specific initializations can be added here
        
    # Override methods as needed for VHH-specific functionality
    # For now, we're inheriting everything from the base class