# mber Core Components

This document describes the core components of the mber package, which provide the foundation for all protocols and workflows.

## Overview

The core components of mber are organized into several key areas:

1. **Data Structures** - Classes for representing design states and related data
2. **Modules** - Base implementations of the design pipeline modules
3. **Configuration** - Configuration classes for various components
4. **Utilities** - Core utility functions for common tasks

## Data Structures

### `mber.core.data.serializable.SerializableDataclass`

Base class providing serialization capabilities for dataclasses. All state-related classes inherit from this.

Key features:
- Pickle serialization/deserialization
- Formatted string representation
- Value formatting based on data types

### `mber.core.data.state`

Contains the core data structures for representing design states:

- `ProtocolInfo` - Protocol version and metadata
- `ConfigData` - Container for all configuration objects
- `TemplateData` - Template information and preparation results
- `TrajectoryData` - Trajectory optimization results
- `BinderData` - Individual binder data and metrics
- `EvaluationData` - Collection of evaluated binders
- `DesignState` - Main design state container combining all data

Example of working with design state:

```python
from mber.core.data.state import DesignState, TemplateData

# Create new design state
state = DesignState(
    template_data=TemplateData(
        target_id="P01234",
        target_name="Example Protein"
    )
)

# Add data to state
state.template_data.target_hotspot_residues = "A10,A25"

# Save state to directory
state.to_dir("./output/my_design")

# Load state from directory
loaded_state = DesignState.from_dir("./output/my_design")
```

## Modules

### Base Module Classes

The core modules define the base implementations of the main pipeline components:

- `BaseTemplateModule` - Template preparation
- `BaseTrajectoryModule` - Trajectory optimization  
- `BaseEvaluationModule` - Binder evaluation

All modules inherit from `BaseModule` which provides common functionality like logging, configuration management, and resource cleanup.

Each module follows a standard interface:

```python
# Standard module interface
def setup(self, design_state): ...    # Set up resources
def run(self, design_state): ...      # Run the main logic
def teardown(self, design_state): ... # Clean up
```

### `mber.core.modules.template.BaseTemplateModule`

Handles template preparation, including:
- Target structure processing
- Hotspot detection
- Structure truncation
- Binder sequence initialization
- Position-specific bias generation

### `mber.core.modules.trajectory.BaseTrajectoryModule`

Handles trajectory optimization, including:
- AlphaFold model setup
- Loss function configuration
- Sequence optimization
- Position-specific scoring matrix (PSSM) generation

### `mber.core.modules.evaluation.BaseEvaluationModule`

Handles binder evaluation, including:
- Complex structure prediction
- Structure relaxation
- Scoring and metrics calculation
- Sequence-based evaluation

## Configuration

### `mber.core.modules.config`

Contains base configuration classes for all modules:

- `BaseTemplateConfig` - Template module configuration
- `BaseModelConfig` - Model configuration 
- `BaseLossConfig` - Loss function configuration
- `BaseTrajectoryConfig` - Trajectory optimization configuration
- `BaseEvaluationConfig` - Evaluation configuration
- `BaseEnvironmentConfig` - Computational environment configuration

Example:

```python
from mber.core.modules.config import BaseTemplateConfig

config = BaseTemplateConfig(
    sasa_threshold=50.0,
    pae_threshold=30.0,
    distance_threshold=25.0,
    folding_model="esmfold"
)
```

## Core Functions

### `mber.core.sasa`

Solvent accessible surface area (SASA) calculation and hotspot detection:

- `SASAHotspotFinder` - Find surface-exposed residues
- `HotspotSelectionStrategy` - Select hotspots using different strategies
- `find_hotspots` - Main entry point for hotspot detection

Example:

```python
from mber.core.sasa import find_hotspots, HotspotSelectionStrategy

# Find hotspots with 'top_k' strategy
hotspots = find_hotspots(
    pdb_content="...",  # PDB file content as string
    region_str="A:10-100",  # Region of interest
    sasa_threshold=50.0,  # Threshold for surface exposure
    hotspot_strategy=HotspotSelectionStrategy.top_k  # Selection strategy
)
```

### `mber.core.truncation`

Protein structure truncation functionality:

- `ProteinTruncator` - Truncate structures based on distances and PAE
- `parse_region_str` - Parse region specification strings

Example:

```python
from mber.core.truncation import ProteinTruncator

truncator = ProteinTruncator(
    pdb_content="...",  # PDB file content
    region_str="A:10-100",  # Region of interest
    pae_matrix=None  # Optional PAE matrix
)

# Create truncation
truncated_pdb, full_pdb, target_chain = truncator.create_truncation(
    hotspots_str="A10,A25",  # Hotspot residues
    pae_threshold=25.0,  # PAE threshold
    distance_threshold=25.0,  # Distance threshold
    gap_penalty=10.0  # Gap penalty for optimization
)
```

### `mber.core.logging`

Centralized logging utilities:

- `MberLogger` - Set up and manage logging
- `LogStore` - Store and retrieve logs
- `StdoutCapturer` - Capture stdout for logging

Example:

```python
from mber.core.logging import MberLogger

# Set up logger
logger, log_store = MberLogger.setup_logger("MyModule")

# Log messages
MberLogger.log(logger, log_store, "This is an info message", level="info")
MberLogger.log(logger, log_store, "This is a warning", level="warning")

# Get logs
logs = log_store.get_logs()
```

## Design State Lifecycle

The typical lifecycle of a design state through the mber pipeline:

1. **Initialization**: Create a new `DesignState` with basic target information
2. **Template Preparation**: 
   - Process target structure
   - Identify hotspots
   - Create structure truncation
   - Initialize binder sequence

3. **Trajectory Optimization**:
   - Set up AlphaFold model
   - Configure loss functions
   - Run sequence optimization
   - Create position-specific scoring matrix

4. **Evaluation**:
   - Predict complex structures
   - Calculate metrics (iPTM, pLDDT, etc.)
   - Perform structure relaxation
   - Score sequences

5. **Serialization**: Save the complete design state to disk

## Implementation Guide

When extending the core functionality, follow these guidelines:

1. **Subclassing**: Extend the base classes rather than modifying them directly
2. **Configuration**: Define configuration parameters in dataclasses
3. **Timing**: Use the `@time_method()` decorator for performance tracking
4. **Logging**: Use the standard logging utilities
5. **Serialization**: Override serialization methods if needed
6. **Documentation**: Document extensions thoroughly
7. **Type Hints**: Use proper type annotations

## Key Extension Points

When developing new protocols or features, these are key points to extend:

1. **Custom Modules**: Extend base modules with protocol-specific logic
2. **Custom Configs**: Add protocol-specific configuration parameters
3. **Custom State**: Add protocol-specific data fields
4. **Custom Losses**: Implement specialized loss functions
5. **Custom Models**: Add new folding or language models