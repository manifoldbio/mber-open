from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np

from mber.core.data.state import (
    DesignState as CoreDesignState,
    TemplateData as CoreTemplateData,
    TrajectoryData as CoreTrajectoryData,
    EvaluationData as CoreEvaluationData,
    BinderData as CoreBinderData,
    ProtocolInfo,
    ConfigData
)

from mber_protocols.version import get_protocol_version

# Use core implementations as our defaults
TemplateData = CoreTemplateData
TrajectoryData = CoreTrajectoryData
BinderData = CoreBinderData
EvaluationData = CoreEvaluationData

@dataclass(repr=False)
class DesignState(CoreDesignState):
    """VHH-specific design state with appropriate protocol info."""
    # Override the protocol_info field with VHH-specific defaults
    protocol_info: ProtocolInfo = field(default_factory=lambda: ProtocolInfo(
        name="VHH_binder_design",
        version=str(get_protocol_version("VHH_binder_design")),
        description="Protocol for designing VHH binders"
    ))