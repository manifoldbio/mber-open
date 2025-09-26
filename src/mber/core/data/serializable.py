# src/mber/core/data.py

from dataclasses import asdict, fields, is_dataclass
from typing import Any, Dict, List, Union
import pickle
import os
import json
from pathlib import Path
import numpy as np

class SerializableDataclass:
    """Base class providing serialization methods for dataclasses."""

    def to_pickle(self, path: str) -> None:
        """Serialize this object to a pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, path: str) -> Any:
        """Load a dataclass from a pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def _format_value(self, key: str, value: Any) -> str:
        """Format a value for display based on its type and key name."""
        if value is None:
            return "None"

        # PDB data
        if key.endswith("_pdb") and isinstance(value, str) and len(value) > 100:
            return f"[PDB data: {len(value.splitlines())} lines]"

        # Sequence data
        elif key.endswith("_seq") and isinstance(value, str) and len(value) > 30:
            return f"{value[:15]}...{value[-15:]} ({len(value)} chars)"

        elif "animat" in key and isinstance(value, str):
            return f"[Animation data: {len(value)} chars]"

        # Lists (shorten if long)
        elif isinstance(value, list):
            if len(value) > 5:
                return f"[{len(value)} items]"
            return str(value)

        # Numpy arrays
        elif isinstance(value, np.ndarray):
            return f"array(shape={value.shape}, dtype={value.dtype})"

        # Floats (round for display)
        elif isinstance(value, float):
            return f"{value:.4f}"

        # Default
        return str(value)

    def __repr__(self) -> str:
        """Default string representation with nice formatting."""
        class_name = self.__class__.__name__
        lines = [f"{class_name}:"]

        # Get all instance attributes
        for k, v in vars(self).items():
            if k.startswith("_"):
                continue

            formatted_value = self._format_value(k, v)
            lines.append(f"  {k}: {formatted_value}")

        return "\n".join(lines)