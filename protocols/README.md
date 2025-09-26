# mber Protocol Development Guide

This document outlines how to develop new protocols for mber. A protocol is a specialized implementation of the mber workflow for a specific type of protein design problem.

## Protocol Architecture

Protocols in mber have a standard architecture:

```
mber_protocols/[status]/[protocol_name]/
├── __init__.py         # Exports key module classes
├── config.py           # Protocol-specific configuration
├── template.py         # Template module implementation
├── trajectory.py       # Trajectory module implementation
├── evaluation.py       # Evaluation module implementation
├── state.py            # Custom state definitions (if needed)
├── pipeline.py         # End-to-end pipeline functionality
└── [protocol_specific] # Additional protocol-specific modules
```

## Creating a New Protocol

### 1. Define Protocol Configuration

Start by defining protocol-specific configurations by subclassing the core config classes:

```python
# mber_protocols/experimental/my_protocol/config.py
from dataclasses import dataclass
from mber.core.modules.config import (
    BaseTemplateConfig, BaseModelConfig, BaseLossConfig,
    BaseTrajectoryConfig, BaseEvaluationConfig, BaseEnvironmentConfig
)

@dataclass
class TemplateConfig(BaseTemplateConfig):
    """MyProtocol-specific template configuration."""
    # Override defaults or add new parameters
    sasa_threshold: float = 60.0  # Custom threshold for this protocol
    folding_model: str = 'custom_model'  # Custom default model

@dataclass
class ModelConfig(BaseModelConfig):
    """MyProtocol-specific model configuration."""
    # Protocol-specific parameters
    use_custom_feature: bool = True
    
# Similarly for other configs...
```

### 2. Implement Protocol Modules

Create protocol-specific modules by subclassing base modules and implementing specialized functionality:

```python
# mber_protocols/experimental/my_protocol/template.py
from mber.core.modules.template import BaseTemplateModule
from mber_protocols.experimental.my_protocol.config import TemplateConfig, EnvironmentConfig

class TemplateModule(BaseTemplateModule):
    """
    MyProtocol-specific template module implementation.
    """
    
    def __init__(
        self,
        template_config: TemplateConfig,
        environment_config: EnvironmentConfig,
        verbose: bool = True,
    ) -> None:
        super().__init__(template_config, environment_config, verbose)
        # Protocol-specific initialization
        
    def _process_hotspots(self, design_state):
        """Override with protocol-specific hotspot processing."""
        # Call parent method first if needed
        design_state = super()._process_hotspots(design_state)
        
        # Add protocol-specific processing
        self._log("Adding MyProtocol-specific hotspot processing")
        # [Custom implementation...]
        
        return design_state
```

### 3. Define Protocol State (Optional)

If your protocol requires custom state information, extend the core state classes:

```python
# mber_protocols/experimental/my_protocol/state.py
from dataclasses import dataclass, field
from typing import List, Optional
from mber.core.data.state import (
    DesignState as CoreDesignState,
    TemplateData as CoreTemplateData,
)

@dataclass
class TemplateData(CoreTemplateData):
    """MyProtocol-specific template data with additional fields."""
    custom_field: str = None
    special_analysis_results: List[float] = field(default_factory=list)

@dataclass
class DesignState(CoreDesignState):
    """MyProtocol-specific design state."""
    # Use the custom template data class
    template_data: TemplateData = field(default_factory=lambda: TemplateData(target_id="", target_name=""))
```

### 4. Create Protocol Pipeline

Create a pipeline module to orchestrate the full protocol workflow:

```python
# mber_protocols/experimental/my_protocol/pipeline.py
from typing import Optional
from mber_protocols.experimental.my_protocol.config import (
    TemplateConfig, ModelConfig, LossConfig, 
    TrajectoryConfig, EvaluationConfig, EnvironmentConfig
)
from mber_protocols.experimental.my_protocol.template import TemplateModule
from mber_protocols.experimental.my_protocol.trajectory import TrajectoryModule
from mber_protocols.experimental.my_protocol.evaluation import EvaluationModule
from mber_protocols.experimental.my_protocol.state import DesignState, TemplateData

def create_default_configs():
    """Create default configurations for this protocol."""
    return {
        "template_config": TemplateConfig(),
        "model_config": ModelConfig(),
        "loss_config": LossConfig(),
        "trajectory_config": TrajectoryConfig(),
        "evaluation_config": EvaluationConfig(),
        "environment_config": EnvironmentConfig(),
    }

def run_pipeline(
    target_id: str,
    target_name: str,
    masked_binder_seq: str,
    region: Optional[str] = None,
    hotspots: Optional[str] = None,
    configs: Optional[dict] = None,
    output_dir: Optional[str] = None,
    verbose: bool = True,
):
    """Run the complete pipeline for MyProtocol."""
    # Use provided configs or defaults
    if configs is None:
        configs = create_default_configs()
    
    # Initialize design state
    design_state = DesignState(
        template_data=TemplateData(
            target_id=target_id,
            target_name=target_name,
            region=region,
            target_hotspot_residues=hotspots,
            masked_binder_seq=masked_binder_seq,
        )
    )
    
    # Template module
    template_module = TemplateModule(
        configs["template_config"], 
        configs["environment_config"],
        verbose=verbose
    )
    template_module.setup(design_state)
    design_state = template_module.run(design_state)
    template_module.teardown(design_state)
    
    # Trajectory module
    trajectory_module = TrajectoryModule(
        configs["model_config"],
        configs["loss_config"],
        configs["trajectory_config"],
        configs["environment_config"],
        verbose=verbose
    )
    trajectory_module.setup(design_state)
    design_state = trajectory_module.run(design_state)
    trajectory_module.teardown(design_state)
    
    # Evaluation module
    evaluation_module = EvaluationModule(
        configs["model_config"],
        configs["loss_config"],
        configs["evaluation_config"],
        configs["environment_config"],
        verbose=verbose
    )
    evaluation_module.setup(design_state)
    design_state = evaluation_module.run(design_state)
    evaluation_module.teardown(design_state)
    
    # Save results if output directory is provided
    if output_dir:
        design_state.to_dir(output_dir)
    
    return design_state
```

### 5. Export Protocol Components

Make protocol components easily accessible by exporting them in `__init__.py`:

```python
# mber_protocols/experimental/my_protocol/__init__.py
from mber_protocols.experimental.my_protocol.template import TemplateModule
from mber_protocols.experimental.my_protocol.trajectory import TrajectoryModule
from mber_protocols.experimental.my_protocol.evaluation import EvaluationModule
from mber_protocols.experimental.my_protocol.pipeline import run_pipeline, create_default_configs
```

## Best Practices

1. **Inheritance vs. Composition**: Use inheritance for small modifications to base functionality. For completely different implementations, consider using composition instead.

2. **Logging**: Use the built-in logging utilities (`self._log()` in modules) for consistent logging.

3. **Timing**: Use the `@time_method()` decorator and `with timer()` context manager for performance monitoring.

4. **Configuration**: Make configuration parameters explicit in dataclasses with type annotations and default values.

5. **Error Handling**: Implement proper error handling with informative error messages. Use the logger for reporting errors.

6. **Documentation**: Document protocol-specific functionality with docstrings and comments.

7. **Testing**: Create unit tests for protocol-specific implementations.

## Example: VHH Binder Design Protocol

The `VHH_binder_design` protocol in `mber_protocols.stable.VHH_binder_design` provides a comprehensive example of a well-structured protocol.

See `notebooks/template_trajectory_evaluation_example_PDL1.ipynb` for a working example of this protocol.