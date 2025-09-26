import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from itertools import groupby
import tempfile
import re

import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
# can drop fastpdb in like this, but need to add Rust package manager and Rust to environment.
# import fastpdb as pdb

from mber.utils.regions import RegionSpec, HotspotSpec, parse_region

class ProteinTruncator:
    """
    A class for creating protein truncations based on multiple hotspots, distances, and PAE values.
    Using Biotite for structure handling for improved performance.
    """
    
    # Cache for structure loading
    _structure_cache = {}
    
    @classmethod
    def get_structure(cls, pdb_file: str) -> Tuple[struc.AtomArray, Dict[str, np.ndarray]]:
        """
        Get structure from cache or load it if not present.
        Optimized to avoid redundant computations.
        """
        if pdb_file not in cls._structure_cache:
            # Load PDB file using Biotite
            pdb_file_obj = pdb.PDBFile.read(pdb_file)
            atom_array = pdb.get_structure(pdb_file_obj, model=1)
            
            # Create a mapping of chain IDs to residue indices more efficiently
            chains = {}
            unique_chains = np.unique(atom_array.chain_id)
            
            for chain_id in unique_chains:
                # Create mask for this chain
                chain_mask = atom_array.chain_id == chain_id
                
                # Get unique residues more efficiently
                # Using numpy's unique function with return_index to preserve order
                _, res_indices = np.unique(atom_array.res_id[chain_mask], return_index=True)
                chain_indices = np.where(chain_mask)[0]
                chain_residues = atom_array.res_id[chain_indices[res_indices]]
                
                chains[chain_id] = chain_residues
            
            cls._structure_cache[pdb_file] = (atom_array, chains)
        
        return cls._structure_cache[pdb_file]

    def __init__(self, 
                 pdb_file: Optional[str] = None, 
                 pdb_content: Optional[str] = None,
                 region_str: Optional[str] = None,
                 regions: Optional[List[RegionSpec]] = None,
                 pae_file: Optional[str] = None,
                 pae_matrix: Optional[str] = None,
                 include_surrounding_context: Optional[bool] = False):
        """
        Initialize the truncator with support for multiple regions.
        
        Args:
            pdb_file: Path to PDB file
            pdb_content: PDB content as string (alternative to pdb_file)
            region_str: Region string in format 'chain:start-end' (alternative to regions)
            regions: List of regions to analyze
            pae_matrix: Optional PAE matrix from JSON file
            include_surrounding_context: If False (default), restrict truncation to the specified regions.
                                       If True, allow truncation to include residues outside the 
                                       specified regions while still restricting hotspot selection to
                                       the specified regions. Useful for multipass membrane proteins where
                                       you want more structural context.
        """
        # Handle either pdb_file or pdb_content
        if pdb_content:
            # Create a temporary file with the content
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
                f.write(pdb_content)
                temp_pdb_file = f.name
            self.pdb_file = Path(temp_pdb_file)
            self._temp_file = temp_pdb_file  # Store for cleanup
        elif pdb_file:
            self.pdb_file = Path(pdb_file)
            self._temp_file = None
        else:
            raise ValueError("Either pdb_file or pdb_content must be provided")
        
        # Handle either region_str or regions
        if region_str and not regions:
            self.regions = parse_region_str(region_str)
        else:
            self.regions = regions if regions else []
        
        # Get all chains in the specified regions
        self.region_chains = list({region.chain for region in self.regions})
        
        # Load structure using optimized method
        self.atom_array, chain_residues_map = self.get_structure(str(self.pdb_file))
        
        self.include_surrounding_context = include_surrounding_context
        
        # If no regions specified, use chain 'A'
        if not self.regions:
            self.regions = [RegionSpec('A')]
            self.region_chains = ['A']
        
        # Use vectorized operations to process residues - now with optional region restriction
        self.process_residues_vectorized(chain_residues_map)
        
        # Load PAE matrix if provided
        if pae_matrix is not None:
            self.pae_matrix = self.truncate_pae_matrix(pae_matrix)
        elif pae_file:
            self.pae_matrix = self.load_pae_matrix(pae_file)
        else:
            self.pae_matrix = None 

    def truncate_pae_matrix(self, pae_matrix):
        # Calculate total filtered residues
        total_filtered_residues = sum(self.n_residues.values())
        
        # Create the filtered PAE matrix
        filtered_pae = np.zeros((total_filtered_residues, total_filtered_residues))
        
        # Collect all indices in a single array
        all_indices = []
        for chain_id in self.region_chains:
            all_indices.extend(self.residue_indices[chain_id])
        
        if not all_indices:
            return None
            
        # Convert to numpy array for efficient indexing
        all_indices = np.array(all_indices)
        
        # Use numpy's advanced indexing to populate the filtered matrix
        # This is much faster than nested loops
        for i in range(len(all_indices)):
            filtered_pae[i] = pae_matrix[all_indices[i], all_indices]

        return filtered_pae
    
    def load_pae_matrix(self, pae_file: str):
        """
        Load and validate the PAE matrix from a JSON file.
        Optimized implementation with better error handling.
        """
        try:
            with open(pae_file, 'r') as f:
                pae_data = json.load(f)
                
            if not isinstance(pae_data, list) or not pae_data:
                raise ValueError("PAE file must contain a non-empty list")
                
            full_pae_matrix = np.array(pae_data[0]['predicted_aligned_error'])
            
            return self.truncate_pae_matrix(full_pae_matrix)

            
        except Exception as e:
            print(f"Error loading PAE matrix: {e}")
            self.pae_matrix = None
    
    def __del__(self):
        """Clean up temporary file if created."""
        if hasattr(self, '_temp_file') and self._temp_file and os.path.exists(self._temp_file):
            try:
                os.unlink(self._temp_file)
            except:
                pass

    def process_residues_vectorized(self, chain_residues_map):
        """
        Process residues using vectorized operations.
        """
        self.chain_residues = {}
        self.n_residues = {}
        self.residue_indices = {}
        self.in_region_mask = {}  # Track which residues are in regions (for hotspot selection)
        cumulative_index = 0
        
        # Process each chain that contains regions of interest
        for chain_id in self.region_chains:
            # Check if chain exists
            if chain_id not in chain_residues_map:
                raise ValueError(f"Chain {chain_id} not found in structure")
            
            # Get all atoms in the chain
            chain_mask = self.atom_array.chain_id == chain_id
            chain_atoms = self.atom_array[chain_mask]
            
            # Get unique residue IDs
            unique_res_ids = np.unique(chain_atoms.res_id)
            
            # Filter residues based on regions
            chain_regions = [r for r in self.regions if r.chain == chain_id]
            filtered_residues = []
            indices = []
            in_region = []  # Track which residues are in the specified regions
            
            # Pre-compute region bounds for faster filtering
            region_bounds = []
            for region in chain_regions:
                start = region.start if region.start is not None else -np.inf
                end = region.end if region.end is not None else np.inf
                region_bounds.append((start, end))
            
            # Create a mask for residues in regions
            in_region_mask = np.zeros(len(unique_res_ids), dtype=bool)
            for start, end in region_bounds:
                in_region_mask |= (unique_res_ids >= start) & (unique_res_ids <= end)
            
            # Process residues
            for i, res_id in enumerate(unique_res_ids):
                # Check if residue is in any of the regions
                residue_in_region = in_region_mask[i]
                
                # If restricting to regions and residue is not in region, skip it
                if not self.include_surrounding_context and not residue_in_region:
                    continue
                
                # Get atoms for this residue
                res_mask = (chain_atoms.res_id == res_id)
                res_atoms = chain_atoms[res_mask]
                
                # Find CA atom index if present
                ca_indices = np.where(res_atoms.atom_name == "CA")[0]
                ca_idx = ca_indices[0] if len(ca_indices) > 0 else None
                
                # Store residue info
                filtered_residues.append({
                    'res_id': res_id,
                    'atoms': res_atoms,
                    'ca_idx': ca_idx
                })
                indices.append(cumulative_index + i)
                in_region.append(residue_in_region)
            
            self.chain_residues[chain_id] = filtered_residues
            self.n_residues[chain_id] = len(filtered_residues)
            self.residue_indices[chain_id] = indices
            self.in_region_mask[chain_id] = np.array(in_region, dtype=bool)
            cumulative_index += len(unique_res_ids)

    def calculate_distances(self, hotspots: List[HotspotSpec]) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[int]]]:
        """
        Calculate distances and PAE values from hotspots to all residues in all chains.
        """
        # Validate hotspots
        for hotspot in hotspots:
            if hotspot.chain not in self.region_chains:
                raise ValueError(f"Hotspot chain {hotspot.chain} not specified in chains to process")
        
        # Find hotspots and store their indices
        hotspot_data = {}  # chain -> List[Tuple[residue, idx]]
        for hotspot in hotspots:
            if hotspot.chain not in hotspot_data:
                hotspot_data[hotspot.chain] = []
                
            found = False
            for i, residue in enumerate(self.chain_residues[hotspot.chain]):
                if residue['res_id'] == hotspot.residue:
                    # Ensure hotspot is within a region of interest
                    if not self.in_region_mask[hotspot.chain][i]:
                        raise ValueError(f"Hotspot residue {hotspot.residue} is not within the specified regions in chain {hotspot.chain}")
                    
                    hotspot_data[hotspot.chain].append((residue, i))
                    found = True
                    break
                    
            if not found:
                raise ValueError(f"Hotspot residue {hotspot.residue} not found in chain {hotspot.chain}")
        
        distances = {}
        for chain_id in self.region_chains:
            chain_residues = self.chain_residues[chain_id]
            n_residues = len(chain_residues)
            
            # Initialize distance arrays
            euclidean_distances = np.full(n_residues, np.inf)
            pae_distances = np.full(n_residues, np.inf)
            
            # Calculate distances using vectorized operations
            chain_hotspots = hotspot_data.get(chain_id, [])
            
            if chain_hotspots:
                # Within same chain - vectorized approach
                self._calculate_chain_distances_vectorized(chain_id, chain_residues, chain_hotspots, 
                                                        euclidean_distances, pae_distances)
            else:
                # Cross-chain distances - vectorized approach
                self._calculate_cross_chain_distances_vectorized(chain_id, chain_residues, hotspot_data,
                                                              euclidean_distances, pae_distances)
            
            if self.pae_matrix is None:
                pae_distances = np.zeros_like(euclidean_distances)
                
            distances[chain_id] = (
                euclidean_distances,
                pae_distances,
                [idx for _, idx in chain_hotspots]
            )
            
        return distances

    def _calculate_chain_distances_vectorized(self, chain_id, chain_residues, chain_hotspots, 
                                           euclidean_distances, pae_distances):
        """Calculate distances to hotspots within the same chain using vectorization."""
        # Extract CA coordinates for residues and hotspots
        residue_coords = []
        res_indices_with_ca = []
        
        for i, residue in enumerate(chain_residues):
            if residue['ca_idx'] is not None:
                residue_coords.append(residue['atoms'].coord[residue['ca_idx']])
                res_indices_with_ca.append(i)
        
        hotspot_coords = []
        hotspot_indices = []
        
        for hotspot_res, hotspot_idx in chain_hotspots:
            if hotspot_res['ca_idx'] is not None:
                hotspot_coords.append(hotspot_res['atoms'].coord[hotspot_res['ca_idx']])
                hotspot_indices.append(hotspot_idx)
        
        # Skip calculation if either list is empty
        if not residue_coords or not hotspot_coords:
            return
        
        # Convert to numpy arrays
        residue_coords = np.array(residue_coords)
        hotspot_coords = np.array(hotspot_coords)
        
        # Calculate all distances at once using broadcasting
        # Shape: (n_residues, n_hotspots, 3) - (n_residues, 1, 3) - (1, n_hotspots, 3)
        distances = np.sqrt(np.sum((residue_coords[:, np.newaxis, :] - hotspot_coords[np.newaxis, :, :]) ** 2, axis=2))
        
        # Find minimum distance for each residue
        min_distances = np.min(distances, axis=1)
        
        # Update the euclidean_distances array
        for res_idx, min_dist in zip(res_indices_with_ca, min_distances):
            euclidean_distances[res_idx] = min(euclidean_distances[res_idx], min_dist)
        
        # Update PAE distances if available
        if self.pae_matrix is not None:
            for i, res_idx in enumerate(res_indices_with_ca):
                for j, hot_idx in enumerate(hotspot_indices):
                    pae_distances[res_idx] = min(pae_distances[res_idx], self.pae_matrix[hot_idx][res_idx])

    def _calculate_cross_chain_distances_vectorized(self, chain_id, chain_residues, hotspot_data,
                                                 euclidean_distances, pae_distances):
        """Calculate distances to hotspots in other chains using vectorization."""
        # Extract CA coordinates for residues
        residue_coords = []
        res_indices_with_ca = []
        
        for i, residue in enumerate(chain_residues):
            if residue['ca_idx'] is not None:
                residue_coords.append(residue['atoms'].coord[residue['ca_idx']])
                res_indices_with_ca.append(i)
        
        # Skip calculation if no residues have CA atoms
        if not residue_coords:
            return
        
        residue_coords = np.array(residue_coords)
        
        # Process each chain's hotspots
        for other_chain, other_hotspots in hotspot_data.items():
            if other_chain == chain_id:
                continue  # Skip same chain
            
            hotspot_coords = []
            hotspot_indices = []
            
            for hotspot_res, hotspot_idx in other_hotspots:
                if hotspot_res['ca_idx'] is not None:
                    hotspot_coords.append(hotspot_res['atoms'].coord[hotspot_res['ca_idx']])
                    hotspot_indices.append(hotspot_idx)
            
            # Skip if no hotspots with CA atoms
            if not hotspot_coords:
                continue
            
            hotspot_coords = np.array(hotspot_coords)
            
            # Calculate all distances at once
            distances = np.sqrt(np.sum((residue_coords[:, np.newaxis, :] - hotspot_coords[np.newaxis, :, :]) ** 2, axis=2))
            
            # Find minimum distance for each residue
            min_distances = np.min(distances, axis=1)
            
            # Update the euclidean_distances array
            for res_idx, min_dist in zip(res_indices_with_ca, min_distances):
                euclidean_distances[res_idx] = min(euclidean_distances[res_idx], min_dist)
            
            # Update PAE distances if available
            if self.pae_matrix is not None:
                # Map indices for PAE matrix
                chain_start = sum(self.n_residues[c] for c in self.region_chains if c < chain_id)
                other_chain_start = sum(self.n_residues[c] for c in self.region_chains if c < other_chain)
                
                for i, res_idx in enumerate(res_indices_with_ca):
                    for j, hot_idx in enumerate(hotspot_indices):
                        pae = self.pae_matrix[other_chain_start + hot_idx][chain_start + res_idx]
                        pae_distances[res_idx] = min(pae_distances[res_idx], pae)

    def optimize_truncation(self, 
                          hotspots: List[HotspotSpec],
                          pae_threshold: float = 25.0,
                          distance_threshold: float = 25.0,
                          gap_penalty: float = 10.0) -> Dict[str, Tuple[List[bool], float]]:
        """
        Optimize truncation for all chains using dynamic programming.
        """
        chain_distances = self.calculate_distances(hotspots)
        results = {}
        
        for chain_id in self.region_chains:
            euclidean_distances, pae_distances, hotspot_indices = chain_distances[chain_id]
            n_res = len(euclidean_distances)
            
            if n_res == 0:
                results[chain_id] = ([], 0.0)
                continue
            
            # Create a combined mask for residues that meet both criteria
            # This is a vectorized operation instead of checking each residue in a loop
            is_close_mask = (euclidean_distances <= distance_threshold) & (pae_distances <= pae_threshold)
            
            # Initialize DP matrices
            F = np.zeros((n_res, 2))
            tb = np.zeros((n_res, 2), dtype=int)
            
            # Initialize first position
            F[0][0] = 0.0 if not is_close_mask[0] else -1.0
            F[0][1] = 2.0 if is_close_mask[0] else -2.0
            
            # Pre-compute base scores for inclusion/exclusion
            # This avoids redundant condition checking in the loop
            base_include_scores = np.where(is_close_mask, 1.0, -1.0)
            base_exclude_scores = np.where(is_close_mask, -1.0, 1.0)
            
            # Fill the DP matrix - we still need a loop here due to dependencies
            for i in range(1, n_res):
                # Calculate scores for exclusion
                score_ex_ex = F[i-1][0] + base_exclude_scores[i]
                score_in_ex = F[i-1][1] + base_exclude_scores[i] - gap_penalty
                
                F[i][0] = max(score_ex_ex, score_in_ex)
                tb[i][0] = 0 if score_ex_ex >= score_in_ex else 1
                
                # Calculate scores for inclusion
                score_in_in = F[i-1][1] + base_include_scores[i]
                score_ex_in = F[i-1][0] + base_include_scores[i] - gap_penalty
                
                F[i][1] = max(score_in_in, score_ex_in)
                tb[i][1] = 1 if score_in_in >= score_ex_in else 0
                
                # Force inclusion of hotspots in this chain
                if i in hotspot_indices:
                    F[i][0] = float('-inf')
            
            # Traceback
            inclusion_mask = np.zeros(n_res, dtype=bool)
            pos = n_res - 1
            current_state = 1 if F[pos][1] >= F[pos][0] else 0
            
            while pos >= 0:
                inclusion_mask[pos] = bool(current_state)
                if pos > 0:
                    current_state = tb[pos][current_state]
                pos -= 1
            
            results[chain_id] = (inclusion_mask.tolist(), max(F[-1]))
        
        return results

    @staticmethod
    def format_residue_ranges(residue_nums: List[int], chain_id: str) -> str:
        """Convert a list of residue numbers into a compact range representation."""
        if not residue_nums:
            return ""
        
        # Sort residue numbers
        residue_nums = sorted(residue_nums)
        
        # Use numpy to find discontinuities
        if len(residue_nums) > 1:
            residue_array = np.array(residue_nums)
            diffs = np.diff(residue_array)
            boundaries = np.where(diffs > 1)[0] + 1
            ranges = np.split(residue_array, boundaries)
        else:
            ranges = [np.array(residue_nums)]
        
        # Format the ranges
        formatted_ranges = []
        for r in ranges:
            if len(r) == 1:
                formatted_ranges.append(f"{chain_id}{r[0]}")
            else:
                formatted_ranges.append(f"{chain_id}{r[0]}-{r[-1]}")
        
        return ",".join(formatted_ranges)

    def write_truncated_pdb(self, output_path: str, inclusion_masks: Dict[str, List[bool]]):
        """Write a new PDB file containing only the truncated residues with chain A."""
        # Create a new atom array for truncated residues
        all_included_atoms = []
        
        for chain_id, mask in inclusion_masks.items():
            for include, residue in zip(mask, self.chain_residues[chain_id]):
                if include:
                    # Add all atoms for this residue, changing chain to 'A'
                    res_atoms = residue['atoms'].copy()
                    res_atoms.chain_id = np.full(len(res_atoms), 'A')
                    all_included_atoms.append(res_atoms)
        
        # Combine into a single array if there are any atoms
        if all_included_atoms:
            combined_array = struc.concatenate(all_included_atoms)
            
            # Write to PDB file
            pdb_file = pdb.PDBFile()
            pdb_file.set_structure(combined_array)
            pdb_file.write(output_path)
        else:
            # Create empty PDB if no residues selected
            with open(output_path, 'w') as f:
                f.write("HEADER    EMPTY TRUNCATION\nEND\n")

    def write_composite_pdb(self, output_path: str, inclusion_masks: Dict[str, List[bool]]):
        """Write a new PDB file with truncated (chain A) and untruncated (chain Z) residues."""
        # Create atom collections for each part
        included_atoms = []
        excluded_atoms = []
        
        for chain_id, mask in inclusion_masks.items():
            # Convert mask to numpy array for vectorized operations
            mask_array = np.array(mask, dtype=bool)
            
            # Get residues to include
            included_residues = [residue for include, residue in zip(mask_array, self.chain_residues[chain_id]) if include]
            
            # Get residues to exclude
            excluded_residues = [residue for include, residue in zip(mask_array, self.chain_residues[chain_id]) if not include]
            
            # Process included residues
            for residue in included_residues:
                res_atoms = residue['atoms'].copy()
                res_atoms.chain_id = np.full(len(res_atoms), 'A')
                included_atoms.append(res_atoms)
            
            # Process excluded residues
            for residue in excluded_residues:
                res_atoms = residue['atoms'].copy()
                res_atoms.chain_id = np.full(len(res_atoms), 'Z')
                excluded_atoms.append(res_atoms)
        
        # Combine all atoms
        all_atoms = []
        
        if included_atoms:
            included_array = struc.concatenate(included_atoms)
            all_atoms.append(included_array)
            
        if excluded_atoms:
            excluded_array = struc.concatenate(excluded_atoms)
            all_atoms.append(excluded_array)
        
        if all_atoms:
            combined_array = struc.concatenate(all_atoms)
            
            # Write to PDB file
            pdb_file = pdb.PDBFile()
            pdb_file.set_structure(combined_array)
            pdb_file.write(output_path)
        else:
            # Create empty PDB if no residues selected
            with open(output_path, 'w') as f:
                f.write("HEADER    EMPTY STRUCTURE\nEND\n")

    def create_truncation(self, 
                        hotspots_str: str,
                        output_dir: Optional[Path] = None,
                        pae_threshold: float = 25.0,
                        distance_threshold: float = 25.0,
                        gap_penalty: float = 10.0) -> Tuple[str, str, str]:
        """
        Create truncated structure based on hotspots string.
        """
        # Parse hotspots
        hotspots = []
        for hotspot in hotspots_str.split(","):
            # Extract chain and residue
            match = re.match(r'([A-Za-z]+)(\d+)', hotspot)
            if match:
                chain, residue = match.groups()
                hotspots.append(HotspotSpec(chain=chain, residue=int(residue)))
            else:
                raise ValueError(f"Invalid hotspot format: {hotspot}")
                
        # Run optimization
        chain_results = self.optimize_truncation(
            hotspots=hotspots,
            pae_threshold=pae_threshold,
            distance_threshold=distance_threshold,
            gap_penalty=gap_penalty
        )
        
        # Get masks for each chain
        chain_masks = {chain_id: mask for chain_id, (mask, _) in chain_results.items()}
        
        # Create temporary files for PDB output
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as temp_trunc:
            truncated_path = temp_trunc.name
            
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as temp_full:
            full_path = temp_full.name
        
        try:
            # Write PDB files
            self.write_truncated_pdb(truncated_path, chain_masks)
            self.write_composite_pdb(full_path, chain_masks)
            
            # Read PDB content
            with open(truncated_path, 'r') as f:
                truncated_pdb = f.read()
                
            with open(full_path, 'r') as f:
                full_target_pdb = f.read()
            
            # Write the PDB files to output_dir if provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
                out_truncated_path = output_dir / "truncated.pdb"
                with open(out_truncated_path, "w") as f:
                    f.write(truncated_pdb)
                    
                out_full_path = output_dir / "full.pdb"
                with open(out_full_path, "w") as f:
                    f.write(full_target_pdb)
        
        finally:
            # Clean up temporary files
            for temp_file in [truncated_path, full_path]:
                try:
                    os.unlink(temp_file)
                except:
                    pass
        
        # We'll use 'A' as the target chain in the combined structure
        target_chain = "A"
        
        return truncated_pdb, full_target_pdb, target_chain


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