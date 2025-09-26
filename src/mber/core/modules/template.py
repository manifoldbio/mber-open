"""
Base module for setting up template information needed for protein design.
This provides the core functionality that specific template modules can inherit.
"""

import tempfile
from pathlib import Path
import torch
from typing import Optional, List, Dict, Union, Callable

from mber.core.modules.base import BaseModule
from mber.core.modules.config import BaseTemplateConfig, BaseEnvironmentConfig
from mber.core.data.state import DesignState, TemplateData
from mber.core.sasa import find_hotspots, HotspotSelectionStrategy
from mber.core.truncation import ProteinTruncator
from mber.models.plm import PLM_MODELS
from mber.models.folding import FOLDING_MODELS
from mber.utils.pdb_utils import process_target, combine_structures, fold_binder
from mber.utils.plm_utils import generate_sequence_from_mask, generate_bias_from_mask
from mber.utils.timing_utils import timer, time_method


class BaseTemplateModule(BaseModule):
    """
    Base module for setting up template information needed for protein design.
    This provides the core functionality that specific template modules can inherit.
    """
    
    def __init__(
        self,
        template_config: BaseTemplateConfig,
        environment_config: BaseEnvironmentConfig,
        verbose: bool = True,
    ) -> None:
        # Initialize the base module first
        super().__init__(
            config=template_config, 
            environment_config=environment_config,
            verbose=verbose,
            name="TemplateModule"
        )
        
        self.template_config = template_config  # For specific access if needed
        
        # Create a temporary directory that will be automatically cleaned up
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Initialize member variables for models
        self.esm_model = None
        self.folding_model = None
    
    @time_method()
    def setup(self, design_state: DesignState) -> None:
        """Set up models and utilities needed for template preparation."""
        self._setup_logging()
        self._setup_models(design_state)
        self._ensure_template_data(design_state)
    
    @time_method()
    def run(self, design_state: DesignState) -> DesignState:
        """Run the template module to prepare all necessary template data."""
        self._log("Starting template preparation")
        design_state = self._process_target(design_state)
        design_state = self._process_hotspots(design_state)
        design_state = self._create_truncation(design_state)
        
        if design_state.template_data.masked_binder_seq:
            design_state = self._process_masked_sequence(design_state)
        
        self._log("Template preparation complete")
        return design_state
    
    @time_method()
    def teardown(self, design_state: DesignState) -> None:
        """Clean up resources after template preparation."""
        self._cleanup_temp_directory()
        self._save_configuration_data(design_state)
        self._stop_logging(design_state)
        self._cleanup_resources()
        
        self._log("Template teardown complete")
        design_state.template_data.template_preparation_complete = True
    
    @time_method()
    def _setup_models(self, design_state: DesignState) -> None:
        """Initialize models needed for template processing."""
        with timer("Initialize ESM2 model", self._log):
            self._log(f"Initializing ESM2 model on {self.environment_config.device}")
            self.esm_model = PLM_MODELS[self.template_config.plm_model](device=self.environment_config.device)
        
        with timer("Initialize folding model", self._log, design_state.template_data.timings):
            self._log(f"Initializing template folding model: {self.template_config.folding_model}")
            self.folding_model = FOLDING_MODELS[self.template_config.folding_model]()
    
    @time_method()
    def _ensure_template_data(self, design_state: DesignState) -> None:
        """Initialize template data if it doesn't exist."""
        if not design_state.template_data:
            self._log("Template data not found, initializing empty template data", level="warning")
            design_state.template_data = TemplateData(
                target_id="",
                target_name=""
            )
    
    @time_method()
    def _process_target(self, design_state: DesignState) -> DesignState:
        """Process target protein to get structure and PAE matrix."""
        target_pdb, pae_matrix = process_target(design_state.template_data.target_id)
        design_state.template_data.template_pdb = target_pdb
        return design_state
    
    @time_method()
    def _process_hotspots(self, design_state: DesignState) -> DesignState:
        """Analyze structure for hotspots if not provided."""
        if not design_state.template_data.target_hotspot_residues:
            hotspot_strategy = getattr(
                HotspotSelectionStrategy(),
                self.template_config.hotspot_strategy
            )
            
            hotspots = find_hotspots(
                design_state.template_data.template_pdb, 
                design_state.template_data.region,
                sasa_threshold=self.template_config.sasa_threshold,
                hotspot_strategy=hotspot_strategy
            )
            
            design_state.template_data.target_hotspot_residues = hotspots
            self._log(f"Using automatically detected hotspots: {hotspots}")
        else:
            self._log(f"Using provided hotspots: {design_state.template_data.target_hotspot_residues}")
        
        return design_state
    
    @time_method()
    def _create_truncation(self, design_state: DesignState) -> DesignState:
        """Create truncation of target structure."""
        self._log("Creating truncation of target structure")
        
        truncator = ProteinTruncator(
            pdb_content=design_state.template_data.template_pdb, 
            region_str=design_state.template_data.region,
            pae_matrix=None,  # Will be populated by process_target if available
            include_surrounding_context=design_state.template_data.include_surrounding_context
        )
        
        # Get path to temporary directory
        temp_dir_path = Path(self.temp_dir.name)
        
        # Create truncation
        truncation_result = truncator.create_truncation(
            hotspots_str=design_state.template_data.target_hotspot_residues,
            output_dir=temp_dir_path,
            pae_threshold=self.template_config.pae_threshold,
            distance_threshold=self.template_config.distance_threshold,
            gap_penalty=self.template_config.gap_penalty
        )
        
        # Unpack the truncation results
        truncated_pdb, full_target_pdb, target_chain = truncation_result
        design_state.template_data.target_chain = target_chain
        design_state.template_data.full_target_pdb = full_target_pdb
        
        return design_state
    
    @time_method()
    def _process_masked_sequence(self, design_state: DesignState) -> DesignState:
        """Process masked sequence to generate binder sequence and structure."""
        self._log("Processing masked binder sequence")
        
        # Get path to temporary directory
        temp_dir_path = Path(self.temp_dir.name)
        
        # Generate complete sequence from masked sequence
        binder_seq = generate_sequence_from_mask(
            self.esm_model,
            design_state.template_data.masked_binder_seq,
            temperature=self.template_config.sampling_temperature,
            num_samples=1
        )
        design_state.template_data.binder_seq = binder_seq
        design_state.template_data.binder_len = len(binder_seq)
        
        # Generate bias from masked sequence
        self._log("Generating position-specific bias")
        binder_bias = generate_bias_from_mask(
            self.esm_model,
            design_state.template_data,
            self.template_config.omit_amino_acids,
            temperature=self.template_config.bias_temperature
        )
        design_state.template_data.binder_bias = binder_bias
        
        # Fold binder structure
        self._log("Folding binder structure")
        binder_pdb = fold_binder(
            self.folding_model,
            binder_seq, 
            output_dir=temp_dir_path
        )
        
        # Combine structures
        self._log("Combining target and binder structures")
        combined_pdb = combine_structures(
            design_state.template_data.full_target_pdb, 
            binder_pdb, 
            design_state.template_data.target_chain,
            output_dir=temp_dir_path
        )
        
        # Update template PDB with combined structure
        design_state.template_data.template_pdb = combined_pdb
        design_state.template_data.binder_chain = self.template_config.binder_chain
        
        return design_state
    
    def _cleanup_temp_directory(self) -> None:
        """Clean up temporary directory."""
        self._log("Cleaning up temporary directory")
        try:
            self.temp_dir.cleanup()
        except Exception as e:
            self._log(f"Error cleaning up temporary directory: {e}", level="warning")
    
    def __del__(self):
        """Destructor to ensure temporary directory is cleaned up."""
        try:
            if hasattr(self, 'temp_dir'):
                self.temp_dir.cleanup()
        except Exception:
            pass