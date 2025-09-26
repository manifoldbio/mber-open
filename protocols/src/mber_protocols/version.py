from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class ProtocolVersion:
    """Version information for a protocol."""
    major: int
    minor: int
    patch: int
    
    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"
    
    @classmethod
    def from_string(cls, version_str: str) -> 'ProtocolVersion':
        """Parse a version string."""
        parts = version_str.split('.')
        if len(parts) != 3:
            raise ValueError(f"Invalid version string: {version_str}")
        return cls(int(parts[0]), int(parts[1]), int(parts[2]))

# Define current versions for all protocols
PROTOCOL_VERSIONS = {
    "VHH_binder_design": ProtocolVersion(1, 0, 0),
    # Add other protocols here as they are developed
}

def get_protocol_version(protocol_name: str) -> Optional[ProtocolVersion]:
    """Get the current version for a protocol."""
    return PROTOCOL_VERSIONS.get(protocol_name)