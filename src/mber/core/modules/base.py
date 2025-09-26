"""
Base module implementation for all mBER modules.
This provides common functionality to reduce code duplication.
"""

from typing import Optional, Callable, Dict, Any, List
import gc
import torch
from dataclasses import asdict

from mber.core.data.state import DesignState


class BaseModule:
    """
    Base class for all mBER modules.
    Provides common functionality like logging setup, teardown, and config management.
    """
    
    def __init__(
        self,
        config: Any,
        environment_config: Any,
        verbose: bool = True,
        name: Optional[str] = None,
    ) -> None:
        """
        Initialize base module with common functionality.
        
        Args:
            config: Module-specific configuration
            environment_config: Environment configuration
            verbose: Whether to print log messages to console
            name: Module name for logging (defaults to class name)
        """
        self.config = config
        self.environment_config = environment_config
        self.verbose = verbose
        self.module_name = name or self.__class__.__name__
        
        # Initialize member variables to be set up in _setup_logging
        self.logger = None
        self.log_store = None
        self.original_stdout = None
        self._log = None
    
    def _setup_logging(self) -> None:
        """Set up logging using the centralized MberLogger."""
        from mber.core.logging import MberLogger
        
        self.logger, self.log_store = MberLogger.setup_logger(self.module_name)
        self._log = lambda message, level="info": MberLogger.log(
            self.logger, self.log_store, message, level, verbose=self.verbose
        )
        
        # Start capturing stdout
        self.original_stdout = MberLogger.start_stdout_capture(self.logger, self.log_store)
        
        # Log a test message
        self._log(f"{self.module_name} initialized with logging")
    
    def _stop_logging(self, design_state: DesignState) -> None:
        """
        Stop logging and save logs to the appropriate data attribute in design_state.
        
        Args:
            design_state: The design state to update with logs
        """
        from mber.core.logging import MberLogger
        
        # Determine the appropriate data attribute based on module name
        data_attr = None
        module_type = self.module_name.lower().replace('module', '')
        
        if hasattr(design_state, f"{module_type}_data"):
            data_attr = getattr(design_state, f"{module_type}_data")
        
        # Stop capturing stdout
        if self.original_stdout:
            MberLogger.stop_stdout_capture(self.original_stdout)
        
        # Store logs in the appropriate attribute if available
        if data_attr is not None and hasattr(data_attr, "logs"):
            data_attr.logs = self.log_store.get_logs()
    
    def _save_configuration_data(self, design_state: DesignState) -> None:
        """
        Save configuration data to design state.
        
        Args:
            design_state: The design state to update with configuration data
        """
        import dataclasses
        
        # Determine configuration attribute name based on module
        config_attr = None
        module_type = self.module_name.lower().replace('module', '')
        
        if module_type in ["template", "model", "loss", "trajectory", "evaluation", "environment"]:
            config_attr = f"{module_type}_config"
        
        # Save configuration if attribute exists
        if config_attr and hasattr(design_state.config_data, config_attr):
            setattr(design_state.config_data, config_attr, asdict(self.config))
        
        # Always save environment config if not already present
        if not design_state.config_data.environment_config and hasattr(self, 'environment_config'):
            design_state.config_data.environment_config = asdict(self.environment_config)
    
    def _cleanup_resources(self) -> None:
        """Clean up resources to free memory."""
        # Clean up torch models
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.nn.Module):
                delattr(self, attr_name)
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()