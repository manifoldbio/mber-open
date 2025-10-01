from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
import numpy as np
import os
import json
from pathlib import Path
from colabdesign.shared.model import aa_order
from dataclasses import asdict, is_dataclass

from mber.core.data.serializable import SerializableDataclass
from mber.utils.yaml_summary_utils import write_metrics_summary

@dataclass(repr=False)
class ProtocolInfo(SerializableDataclass):
    """Information about the protocol version and metadata."""
    name: str
    version: str
    description: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass(repr=False)
class ConfigData(SerializableDataclass):
    """Store all configuration parameters from different modules."""
    template_config: Optional[dict] = None
    model_config: Optional[dict] = None
    loss_config: Optional[dict] = None
    trajectory_config: Optional[dict] = None
    evaluation_config: Optional[dict] = None
    environment_config: Optional[dict] = None


@dataclass(repr=False)
class TemplateData(SerializableDataclass):
    """Data class for template information."""
    # Required inputs
    target_id: str  # UniProt, PDB, input pdb filename, or other unique ID for target protein
    target_name: str  # Colloquial name for target protein

    # Optional inputs
    region: Optional[str] = None  # Region of interest (e.g. "A:101-200")
    target_hotspot_residues: Optional[str] = None  # Hotspot residues (e.g. "A10,A25")
    masked_binder_seq: Optional[str] = None  # Sequence with '*' marking positions to fill
    include_surrounding_context: Optional[bool]= False
    
    # Outputs filled by TemplateModule
    target_source: Optional[str] = None  # Source of target (e.g. "uniprot", "pdb", "input")
    template_pdb: Optional[str] = None  # Combined PDB file content as string
    full_target_pdb: Optional[str] = None  # PDB with both truncated (chain A) and untruncated (chain Z) residues
    target_chain: Optional[str] = None  # Chain ID for target protein
    binder_chain: Optional[str] = None  # Chain ID for binder protein
    binder_len: Optional[int] = None  # Length of binder sequence
    binder_seq: Optional[str] = None  # Full binder sequence
    binder_bias: Optional[np.ndarray] = None  # Bias matrix for design
    template_preparation_complete: bool = False  # Flag to indicate preparation is complete
    logs: List[Dict] = field(default_factory=list)  # Module logs
    timings: Dict[str, float] = field(default_factory=dict)  # Timing information

    def __repr__(self) -> str:
        """Custom representation for template data."""
        class_name = self.__class__.__name__
        lines = [f"{class_name}:"]

        for k, v in vars(self).items():
            if k.startswith("_"):
                continue

            # Custom formatting for template-specific fields
            if k == "target_hotspot_residues" and isinstance(v, list):
                if v and len(v) > 8:
                    v_str = f"{v[:4]}...{v[-4:]} ({len(v)} residues)"
                else:
                    v_str = str(v)
            else:
                v_str = self._format_value(k, v)

            lines.append(f"  {k}: {v_str}")

        return "\n".join(lines)
    
    def get_fix_pos(self, as_array: bool = True):
        """
        Define positions in the sequence that are fixed by the mask.
        
        Args:
            as_array: If True, return positions as numpy array; otherwise as comma-separated string
            
        Returns:
            Fixed positions (1-indexed)
        """
        if self.masked_binder_seq is None:
            return None
        else:
            if as_array:
                # Find positions in masked_binder_seq that are not '*' (1-indexed)
                return np.array([i + 1 for i, aa in enumerate(self.masked_binder_seq) if aa != '*'])
            else:
                # Find positions in masked_binder_seq that are not '*' (1-indexed)
                return ','.join([self.binder_chain + str(i + 1) for i, aa in enumerate(self.masked_binder_seq) if aa != '*'])
            
    def get_flex_pos(self, as_array: bool = True):
        """
        Define positions in the sequence that are flexible (i.e., not fixed by the mask).
        
        Args:
            as_array: If True, return positions as numpy array; otherwise as comma-separated string
            
        Returns:
            Flexible positions (1-indexed)
        """
        if self.masked_binder_seq is None:
            return None
        else:
            if as_array:
                # Find positions in masked_binder_seq that are '*' (1-indexed)
                return np.array([i + 1 for i, aa in enumerate(self.masked_binder_seq) if aa == '*'])
            else:
                # Find positions in masked_binder_seq that are '*' (1-indexed)
                return ','.join([self.binder_chain + str(i + 1) for i, aa in enumerate(self.masked_binder_seq) if aa == '*'])
            
    def get_fix_bias(self):
        """
        Define a bias to fix certain positions in the sequence.
        
        Returns:
            Bias matrix to apply to fixed positions
        """
        if self.masked_binder_seq is None or self.binder_seq is None:
            return None
        else:
            fix_pos = self.get_fix_pos(as_array=True) - 1  # Convert to 0-indexed
            fix_bias = np.zeros((len(self.binder_seq), len(aa_order)))
            fix_bias[fix_pos] = -1e6  # Set very negative values for all AAs at fixed positions
            
            # Set the correct AA at each fixed position to have zero bias
            for pos in fix_pos:
                res_aa = self.binder_seq[pos]
                fix_bias[pos, aa_order[res_aa]] = 0
                
            return fix_bias


@dataclass(repr=False)
class TrajectoryData(SerializableDataclass):
    # inputs
    seed: int = None
    trajectory_name: str = None

    # outputs
    metrics: List[dict] = None
    best_pdb: str = None
    final_seqs: List[str] = None
    updated_bias: np.ndarray = None
    pssm_logits: np.ndarray = None
    animated_trajectory: str = None
    early_stop: bool = False  # Flag to indicate if early stopping was triggered
    trajectory_complete: bool = False
    logs: List[Dict] = field(default_factory=list)  # Module logs
    timings: Dict[str, float] = field(default_factory=dict)  # Timing information
    # Detailed step timings
    step_timings: Dict[str, float] = field(default_factory=dict)  # Detailed step timings

    def __repr__(self) -> str:
        """Custom representation for trajectory data."""
        class_name = self.__class__.__name__
        lines = [f"{class_name}:"]

        for k, v in vars(self).items():
            if k.startswith("_"):
                continue

            # Custom formatting for trajectory-specific fields
            if k == "metrics" and isinstance(v, list) and v:
                # Show summary of last metrics
                last_metrics = v[-1]
                if isinstance(last_metrics, dict):
                    v_str = f"[{len(v)} steps, last: "
                    # Include a sample of metrics from the last step
                    metrics_sample = list(last_metrics.items())[:3]
                    metrics_str = ", ".join(
                        f"{k}={val:.3f}" if isinstance(val, float) else f"{k}={val}"
                        for k, val in metrics_sample
                    )
                    v_str += f"{metrics_str}...]"
                else:
                    v_str = f"[{len(v)} steps]"
            elif k == "final_seqs" and isinstance(v, list) and v:
                if len(v) == 1:
                    v_str = f"[1 sequence: {v[0][:10]}...]"
                else:
                    v_str = f"[{len(v)} sequences]"
            else:
                v_str = self._format_value(k, v)

            lines.append(f"  {k}: {v_str}")

        return "\n".join(lines)


@dataclass(repr=False)
class BinderData(SerializableDataclass):
    binder_seq: str = None
    complex_pdb: str = None
    monomer_pdb: str = None  # Add field for monomer PDB
    relaxed_pdb: str = None
    plddt: float = None
    ptm: float = None
    i_ptm: float = None
    pae: float = None
    i_pae: float = None
    seq_ent: float = None
    unrelaxed_energy: float = None
    relaxed_energy: float = None
    relax_rmsd: float = None
    esm_score: float = None
    timings: Dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Custom representation for binder data."""
        class_name = self.__class__.__name__
        lines = [f"{class_name}:"]

        # First show sequence and key metrics
        important_keys = ["binder_seq", "plddt", "ptm", "i_ptm", "esm_score"]
        for k in important_keys:
            if k in vars(self) and vars(self)[k] is not None:
                v = vars(self)[k]
                v_str = self._format_value(k, v)
                lines.append(f"  {k}: {v_str}")

        # Then show remaining fields
        for k, v in vars(self).items():
            if k.startswith("_") or k in important_keys:
                continue

            v_str = self._format_value(k, v)
            lines.append(f"  {k}: {v_str}")

        return "\n".join(lines)


@dataclass(repr=False)
class EvaluationData(SerializableDataclass):
    # outputs
    binders: List[BinderData] = field(default_factory=lambda: [])
    evaluation_complete: bool = False
    logs: List[Dict] = field(default_factory=list)  # Module logs
    timings: Dict[str, float] = field(default_factory=dict)  # Timing information

    def __repr__(self) -> str:
        """Custom representation for evaluation data."""
        class_name = self.__class__.__name__
        lines = [f"{class_name}:"]

        for k, v in vars(self).items():
            if k.startswith("_"):
                continue

            # Custom formatting for binders list
            if k == "binders" and isinstance(v, list):
                if not v:
                    v_str = "[]"
                else:
                    # Calculate summary statistics
                    plddt_values = [
                        b.plddt
                        for b in v
                        if hasattr(b, "plddt") and b.plddt is not None
                    ]
                    ptm_values = [
                        b.ptm for b in v if hasattr(b, "ptm") and b.ptm is not None
                    ]
                    i_ptm_values = [
                        b.i_ptm
                        for b in v
                        if hasattr(b, "i_ptm") and b.i_ptm is not None
                    ]
                    pae_values = [
                        b.pae for b in v if hasattr(b, "pae") and b.pae is not None
                    ]
                    i_pae_values = [
                        b.i_pae
                        for b in v
                        if hasattr(b, "i_pae") and b.i_pae is not None
                    ]

                    stats = []
                    if plddt_values:
                        avg_plddt = sum(plddt_values) / len(plddt_values)
                        stats.append(f"avg pLDDT: {avg_plddt:.2f}")
                    if ptm_values:
                        avg_ptm = sum(ptm_values) / len(ptm_values)
                        stats.append(f"avg pTM: {avg_ptm:.2f}")
                    if i_ptm_values:
                        avg_i_ptm = sum(i_ptm_values) / len(i_ptm_values)
                        stats.append(f"avg iPTM: {avg_i_ptm:.2f}")
                    if pae_values:
                        avg_pae = sum(pae_values) / len(pae_values)
                        stats.append(f"avg pAE: {avg_pae:.2f}")
                    if i_pae_values:
                        avg_i_pae = sum(i_pae_values) / len(i_pae_values)
                        stats.append(f"avg iPAE: {avg_i_pae:.2f}")

                    v_str = f"[{len(v)} binders, {', '.join(stats)}]"
            else:
                v_str = self._format_value(k, v)

            lines.append(f"  {k}: {v_str}")

        return "\n".join(lines)


@dataclass(repr=False)
class DesignState(SerializableDataclass):
    """Base class for the overall design state."""
    template_data: TemplateData = field(default_factory=lambda: TemplateData(target_id="", target_name=""))
    trajectory_data: TrajectoryData = field(default_factory=TrajectoryData)
    evaluation_data: EvaluationData = field(default_factory=EvaluationData)
    protocol_info: ProtocolInfo = field(default_factory=lambda: ProtocolInfo(
        name="",
        version="",
        description=""
    ))
    config_data: ConfigData = field(default_factory=ConfigData)

    def __repr__(self) -> str:
        """Display a summary of the design state."""
        class_name = self.__class__.__name__
        lines = [f"{class_name}:"]

        for k, v in vars(self).items():
            if k.startswith("_"):
                continue

            if v is None:
                v_str = "None"
            elif is_dataclass(v):
                # For dataclasses, display their repr tabbed out another level
                component_repr = repr(v)
                component_repr = component_repr.replace("\n", "\n    ")
                v_str = component_repr
            else:
                v_str = self._format_value(k, v)

            lines.append(f"  {k}: {v_str}")

        return "\n".join(lines)

    def to_dir(
        self,
        dir_path: str,
        save_pickle: bool = True,
        save_png: bool = True,
        save_animations: bool = True,
    ) -> None:
        """
        Export the design state to a directory structure.

        Args:
            dir_path: Path to the directory where data will be stored
            save_pickle: If False, do not write design_state.pickle
            save_png: If False, do not write PNG images (e.g., seqlogos)
            save_animations: If False, do not write animation HTML files
        """
        os.makedirs(dir_path, exist_ok=True)

        # Save a pickle of the entire state for easy loading
        if save_pickle:
            pickle_path = os.path.join(dir_path, "design_state.pickle")
            self.to_pickle(pickle_path)

        # Generate YAML summary with key metrics
        yaml_path = os.path.join(dir_path, "design_summary.yaml")
        write_metrics_summary(self, yaml_path)

        # Process each component
        for component_name, component in vars(self).items():
            if component_name.startswith("_") or component is None:
                continue

            # Create component subdirectory
            component_dir = os.path.join(dir_path, component_name)
            os.makedirs(component_dir, exist_ok=True)

            # Handle each component type appropriately
            self._export_component(
                component,
                component_name,
                component_dir,
                save_png=save_png,
                save_animations=save_animations,
            )

    def _export_component(
        self,
        component: Any,
        component_name: str,
        component_dir: str,
        *,
        save_png: bool = True,
        save_animations: bool = True,
    ) -> None:
        """Export a component to its directory."""
        if not is_dataclass(component):
            return

        # Convert to dict for processing
        component_dict = asdict(component)
        json_data = {}

        # Process each field
        for field_name, field_value in component_dict.items():
            # Skip private fields
            if field_name.startswith("_"):
                continue

            # Handle based on field type
            if field_value is None:
                json_data[field_name] = None
            elif field_name.endswith("_pdb") and isinstance(field_value, str):
                # PDB files get their own files
                pdb_path = os.path.join(component_dir, f"{field_name}.pdb")
                with open(pdb_path, "w") as f:
                    f.write(field_value)
                json_data[field_name] = f"{field_name}.pdb"  # Store file reference
            elif isinstance(field_value, np.ndarray):
                # If logits, optionally save seqlogo
                if save_png and field_name.endswith("logits") and field_value.ndim == 2:
                    from mber.utils.plotting import generate_seqlogo_from_logits
                    logo_path = os.path.join(component_dir, f"{field_name}.png")
                    generate_seqlogo_from_logits(field_value, logo_path)
                # Convert numpy arrays to lists for JSON
                json_data[field_name] = field_value.tolist()
            elif (
                field_name == "binders"
                and isinstance(field_value, list)
                and field_value
            ):
                # Special handling for binders list in EvaluationData
                binders_dir = os.path.join(component_dir, "binders")
                os.makedirs(binders_dir, exist_ok=True)

                binders_json = []
                for i, binder in enumerate(field_value):
                    if is_dataclass(binder):
                        binder_dict = asdict(binder)
                    else:
                        binder_dict = binder  # Fallback if not a dataclass

                    binder_json = {}

                    # Process binder fields
                    for b_name, b_value in binder_dict.items():
                        if b_name.startswith("_"):
                            continue

                        if b_value is None:
                            binder_json[b_name] = None
                        elif b_name.endswith("_pdb") and isinstance(b_value, str):
                            # Get i_ptm value for filename formatting
                            iptm_value = binder_dict.get("i_ptm", 0.0)

                            # Determine pdb type
                            if b_name == "relaxed_pdb":
                                pdb_type = "relaxed"
                            elif b_name == "monomer_pdb":
                                pdb_type = "monomer"
                            else:
                                pdb_type = "complex"  # Changed from "unrelaxed" to "complex" for clarity

                            # Create formatted filename
                            formatted_filename = f"{self.trajectory_data.trajectory_name}_binder-{i}_iptm-{iptm_value:.4f}_{pdb_type}.pdb"

                            # Save PDB file with new naming format
                            pdb_path = os.path.join(binders_dir, formatted_filename)
                            with open(pdb_path, "w") as f:
                                f.write(b_value)
                            binder_json[b_name] = f"binders/{formatted_filename}"
                        elif isinstance(b_value, np.ndarray):
                            binder_json[b_name] = b_value.tolist()
                        else:
                            binder_json[b_name] = b_value

                    binders_json.append(binder_json)

                json_data[field_name] = binders_json
            elif "animat" in field_name and isinstance(field_value, str):
                # Optionally save animation as html file
                if save_animations:
                    animat_path = os.path.join(component_dir, f"{field_name}.html")
                    with open(animat_path, "w") as f:
                        f.write(field_value)
                    json_data[field_name] = f"{field_name}.html"
                else:
                    json_data[field_name] = None
            else:
                json_data[field_name] = field_value

        # Save component data as JSON
        json_path = os.path.join(component_dir, f"{component_name}.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

    @classmethod
    def from_dir(cls, dir_path: str, pickle_file: str = None) -> "DesignState":
        """
        Load a design state from a directory.

        Args:
            dir_path: Path to the directory containing the saved state

        Returns:
            Reconstructed DesignState object
        """
        # For simplicity and reliability, use the pickle file
        if pickle_file is None:
            pickle_path = "design_state.pickle"
        pickle_path = os.path.join(dir_path, pickle_path)
        if os.path.exists(pickle_path):
            return cls.from_pickle(pickle_path)

        # If pickle doesn't exist, could implement reconstruction from dir structure
        # (not implemented here since it's complex and pickle is more reliable)
        raise FileNotFoundError(
            f"Could not find {pickle_path} in {dir_path}. "
            "Reconstruction from directory structure without pickle is not implemented."
        )