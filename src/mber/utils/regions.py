# src/mber/utils/regions.py

from dataclasses import dataclass
from typing import Optional
from Bio.PDB.Residue import Residue
import argparse


@dataclass
class RegionSpec:
    """Specification for a protein region to analyze."""

    chain: str
    start: Optional[int] = None
    end: Optional[int] = None

    def contains_residue(self, residue: Residue) -> bool:
        """Check if a residue falls within this region."""
        if residue.parent.id != self.chain:
            return False

        res_num = residue.id[1]
        if self.start is not None and res_num < self.start:
            return False
        if self.end is not None and res_num > self.end:
            return False
        return True


def parse_region(region_str: str) -> RegionSpec:
    """Parse a region string in the format 'chain[:start-end]'."""
    parts = region_str.split(":")
    chain = parts[0]

    if len(parts) == 1:
        return RegionSpec(chain)

    try:
        start, end = map(int, parts[1].split("-"))
        return RegionSpec(chain, start, end)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid region format: {region_str}. Use 'chain[:start-end]'"
        )


@dataclass
class HotspotSpec:
    """Specification for a protein hotspot."""

    chain: str
    residue: int

    @classmethod
    def parse_hotspot(cls, hotspot_str: str) -> "HotspotSpec":
        """Parse a hotspot string in the format 'chain:residue'."""
        try:
            chain, residue = hotspot_str.split(":")
            return cls(chain=chain, residue=int(residue))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid hotspot format: {hotspot_str}. Use 'chain:residue'"
            )
