import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
from tempfile import NamedTemporaryFile
import os
import random
import re

import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
from biotite.structure import sasa

from mber.utils.regions import RegionSpec

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SASAHotspotFinder:
    def __init__(self, 
                 pdb_file: str,
                 sasa_threshold: float = 50.0,
                 min_distance: int = 0,
                 regions: Optional[List[RegionSpec]] = None):
        """
        Initialize the SASA hotspot finder.
        """
        self.pdb_file = Path(pdb_file)
        self.sasa_threshold = sasa_threshold
        self.min_distance = min_distance
        self.regions = regions if regions else []
        
        # Load structure using Biotite
        pdb_file_obj = pdb.PDBFile.read(pdb_file)
        self.atom_array = pdb.get_structure(pdb_file_obj, model=1)
        
        # Calculate SASA
        try:
            self.atom_sasa = sasa(
                self.atom_array,
                probe_radius=1.4,
                point_number=100
            )
        except Exception as e:
            logging.error(f"Error calculating SASA: {e}")
            # Create a fallback array of zeros
            self.atom_sasa = np.zeros(len(self.atom_array))
        
        # Process residues
        self.residues = self._process_residues()
    
    def _process_residues(self):
        """Process atoms and calculate residue SASA values using vectorized operations."""
        # Create a dictionary to store residue data
        residue_dict = {}
        
        # Get unique chain IDs
        chain_ids = np.unique(self.atom_array.chain_id)
        
        # Process each chain
        for chain_id in chain_ids:
            # Skip chain if not in regions (when regions are specified)
            if self.regions and not any(region.chain == chain_id for region in self.regions):
                continue
            
            # Get atoms in this chain
            chain_mask = self.atom_array.chain_id == chain_id
            chain_atoms = self.atom_array[chain_mask]
            chain_sasa = self.atom_sasa[chain_mask]
            
            # Get unique residue IDs and names
            _, res_indices = np.unique(chain_atoms.res_id, return_index=True)
            unique_res_ids = chain_atoms.res_id[res_indices]
            unique_res_names = chain_atoms.res_name[res_indices]
            
            # Process each residue
            for i, (res_id, res_name) in enumerate(zip(unique_res_ids, unique_res_names)):
                # Skip non-amino acid residues
                if not self._is_amino_acid(res_name):
                    continue
                
                # Check if residue is in any specified region
                if self.regions and not self._is_in_region(chain_id, res_id):
                    continue
                
                # Find all atoms for this residue
                res_mask = chain_atoms.res_id == res_id
                res_sasa_values = chain_sasa[res_mask]
                
                # Filter out NaN values
                res_sasa_values = res_sasa_values[~np.isnan(res_sasa_values)]
                
                if len(res_sasa_values) > 0:
                    # Calculate total SASA for this residue
                    total_sasa = np.sum(res_sasa_values)
                    
                    residue_info = {
                        'chain': chain_id,
                        'residue': res_id,
                        'res_name': res_name
                    }
                    
                    residue_dict[(chain_id, res_id)] = (residue_info, total_sasa)
        
        return list(residue_dict.values())
    
    def _is_amino_acid(self, res_name: str) -> bool:
        """Check if a residue name is a standard amino acid."""
        # Common amino acid 3-letter codes
        aa_codes = {
            'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
            'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
            # Modified amino acids common in PDB files
            'MSE', 'CSO', 'PTR', 'TPO', 'SEP', 'HYP', 'KCX'
        }
        return res_name in aa_codes
    
    def _is_in_region(self, chain_id: str, res_id: int) -> bool:
        """Check if a residue is in any of the specified regions."""
        return any(
            (region.chain == chain_id) and
            (region.start is None or res_id >= region.start) and
            (region.end is None or res_id <= region.end)
            for region in self.regions
        )
    
    def find_hotspots(self) -> List[Dict]:
        """
        Find surface-exposed residues that can serve as hotspots.
        Optimized implementation with vectorized operations where possible.
        """
        # Sort residues by SASA value (descending)
        sorted_residues = sorted(
            self.residues,
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select hotspots
        hotspots = []
        for residue_info, sasa_value in sorted_residues:
            if sasa_value < self.sasa_threshold:
                continue
            
            # Check if candidate is valid (far enough from other hotspots)
            if hotspots:
                is_valid = True
                candidate_chain = residue_info['chain']
                candidate_resnum = residue_info['residue']
                
                # For each chain, check distance to existing hotspots
                for chain_id in set(h['chain'] for h in hotspots):
                    if chain_id == candidate_chain:
                        # Get all residue numbers for this chain
                        chain_resnums = np.array([h['residue'] for h in hotspots 
                                                if h['chain'] == chain_id])
                        
                        # Calculate distances to all residues in this chain at once
                        distances = np.abs(chain_resnums - candidate_resnum)
                        
                        # If any distance is less than min_distance, this hotspot is invalid
                        if np.any(distances < self.min_distance):
                            is_valid = False
                            break
                
                if not is_valid:
                    continue
            
            hotspot = {
                'residue': residue_info['residue'],
                'chain': residue_info['chain'],
                'sasa': sasa_value,
                'res_name': residue_info.get('res_name', '')
            }
            hotspots.append(hotspot)
        
        return hotspots
    
    def _is_valid_hotspot(self, 
                         candidate: Dict, 
                         selected: List[Dict]) -> bool:
        """Check if a candidate residue is far enough from existing hotspots."""
        candidate_chain = candidate['chain']
        candidate_resnum = candidate['residue']
        
        # Get all residues in the same chain
        same_chain_hotspots = [h['residue'] for h in selected if h['chain'] == candidate_chain]
        
        if same_chain_hotspots:
            # Convert to numpy array for vectorized operations
            chain_resnums = np.array(same_chain_hotspots)
            distances = np.abs(chain_resnums - candidate_resnum)
            
            # Check if any distance is less than min_distance
            if np.any(distances < self.min_distance):
                return False
                
        return True
    
    def write_hotspots(self, output_file: str):
        """Write hotspots to a CSV file."""
        hotspots = self.find_hotspots()
        
        with open(output_file, 'w') as f:
            f.write("chain,residue,sasa,res_name\n")
            for hotspot in hotspots:
                f.write(f"{hotspot['chain']},{hotspot['residue']},{hotspot['sasa']:.2f},{hotspot['res_name']}\n")
            
        logging.info(f"Found {len(hotspots)} hotspots")
        logging.info(f"Wrote hotspots to {output_file}")


class HotspotSelectionStrategy:
    """Strategies for selecting hotspots from candidates."""
    
    @staticmethod
    def top_k(hotspots: List[Dict], k: int = 3) -> List[Dict]:
        """Select top k hotspots by SASA value."""
        return hotspots[:k]
    
    @staticmethod
    def random(hotspots: List[Dict], k: int = 1) -> List[Dict]:
        """Select k random hotspots."""
        if not hotspots:
            return []
        return random.sample(hotspots, min(k, len(hotspots)))
    
    @staticmethod
    def none(hotspots: List[Dict], k: int = 0) -> List[Dict]:
        """Return an empty list (no hotspots)."""
        return []


def parse_region_str(region_str: Optional[str]) -> List[RegionSpec]:
    """
    Parse a region string into RegionSpec objects.
    Optimized implementation with regex for better performance.
    """
    if not region_str:
        return []
    
    # Compile regex patterns for faster matching
    chain_range_pattern = re.compile(r'([A-Za-z]+):(\d+)-(\d+)')
    chain_single_pattern = re.compile(r'([A-Za-z]+):(\d+)')
    chain_only_pattern = re.compile(r'([A-Za-z]+)')
    
    regions = []
    for r in region_str.split(","):
        r = r.strip()
        
        # Try to match chain:start-end pattern
        match = chain_range_pattern.match(r)
        if match:
            chain, start, end = match.groups()
            regions.append(RegionSpec(chain=chain, start=int(start), end=int(end)))
            continue
        
        # Try to match chain:residue pattern
        match = chain_single_pattern.match(r)
        if match:
            chain, res_num = match.groups()
            res_num = int(res_num)
            regions.append(RegionSpec(chain=chain, start=res_num, end=res_num))
            continue
        
        # Try to match just chain pattern
        match = chain_only_pattern.match(r)
        if match:
            chain = match.group(1)
            regions.append(RegionSpec(chain=chain))
            continue
        
        # If no pattern matches, raise error
        if r:  # Skip empty strings
            raise ValueError(f"Invalid region format: {r}")
    
    return regions


def _fallback_hotspot(pdb_path: str, regions: List[RegionSpec]) -> str:
    """
    Find a fallback hotspot when SASA analysis finds none.
    Vectorized implementation for better performance.
    """
    # Load structure using Biotite
    pdb_file_obj = pdb.PDBFile.read(pdb_path)
    atom_array = pdb.get_structure(pdb_file_obj)
    
    # Get unique chain IDs
    unique_chains = np.unique(atom_array.chain_id)
    
    # Create a set of chain IDs from regions for faster lookups
    region_chains = set()
    if regions:
        region_chains = {r.chain for r in regions}
    
    valid_residues = []
    
    # Process each chain
    for chain_id in unique_chains:
        # Skip chains not in regions if regions are specified
        if regions and chain_id not in region_chains:
            continue
            
        # Get atoms in this chain
        chain_mask = (atom_array.chain_id == chain_id)
        chain_atoms = atom_array[chain_mask]
        
        # Get unique residue IDs
        res_ids = np.unique(chain_atoms.res_id)
        
        # If no regions specified, add all residues
        if not regions:
            for res_id in res_ids:
                valid_residues.append((chain_id, res_id))
            continue
        
        # Filter residues based on region ranges
        for res_id in res_ids:
            for region in [r for r in regions if r.chain == chain_id]:
                start = region.start if region.start is not None else -np.inf
                end = region.end if region.end is not None else np.inf
                
                if start <= res_id <= end:
                    valid_residues.append((chain_id, res_id))
                    break  # No need to check other regions
    
    if valid_residues:
        # Randomly select one residue
        chain_id, res_id = random.choice(valid_residues)
        return f"{chain_id}{res_id}"
                        
    raise ValueError("Could not find any suitable hotspot residue")


def _last_resort_fallback(pdb_path: str) -> str:
    """Absolute last resort fallback to find any residue."""
    pdb_file_obj = pdb.PDBFile.read(pdb_path)
    atom_array = pdb.get_structure(pdb_file_obj)
    
    # Get a list of all chain_id, res_id combinations
    chains = np.unique(atom_array.chain_id)
    
    if len(chains) == 0:
        raise ValueError("No chains found in PDB file")
    
    # Pick the first chain
    chain_id = chains[0]
    chain_mask = (atom_array.chain_id == chain_id)
    chain_atoms = atom_array[chain_mask]
    
    # Get residue IDs
    res_ids = np.unique(chain_atoms.res_id)
    
    if len(res_ids) == 0:
        raise ValueError("No residues found in chain")
    
    # Pick the middle residue
    middle_idx = len(res_ids) // 2
    res_id = res_ids[middle_idx]
    
    return f"{chain_id}{res_id}"


def find_hotspots(pdb_content: str, region_str: Optional[str] = None, 
                sasa_threshold: float = 50.0, hotspot_strategy: Callable = None) -> str:
    """
    Find surface-exposed hotspots in the protein structure.
    Optimized implementation with better error handling.
    """
    # Parse region if provided
    regions = parse_region_str(region_str)
    
    # Write PDB to temporary file
    with NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as temp_file:
        temp_file.write(pdb_content)
        temp_file_path = temp_file.name
    
    try:
        # Find hotspots using SASA analysis
        finder = SASAHotspotFinder(
            pdb_file=temp_file_path,
            sasa_threshold=sasa_threshold,
            min_distance=10,  # Ensure hotspots are not too close
            regions=regions
        )
        
        hotspots = finder.find_hotspots()
        
        # Select hotspots using the specified strategy
        if hotspot_strategy is None:
            hotspot_strategy = HotspotSelectionStrategy.random
            
        if hotspots:
            # Use the strategy to select hotspots
            selected_hotspots = hotspot_strategy(hotspots)
            # Convert to string format
            hotspot_str = ",".join([f"{h['chain']}{h['residue']}" for h in selected_hotspots])
            return hotspot_str
        else:
            # Fallback approach if no hotspots found
            return _fallback_hotspot(temp_file_path, regions)
    except Exception as e:
        logging.error(f"Error finding hotspots: {e}")
        # Last resort fallback - return a random residue
        try:
            return _last_resort_fallback(temp_file_path)
        except:
            raise ValueError("Could not find any suitable hotspot residue")
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
        except:
            pass