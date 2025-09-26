import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import time
from datetime import datetime
import io


def create_metrics_summary(design_state) -> Dict[str, Any]:
    """
    Create a summary of the most important metrics from a design state.

    Args:
        design_state: The completed design state

    Returns:
        Dictionary with key metrics
    """
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Initialize summary dictionary with metadata
    summary = {
        "metadata": {
            "timestamp": timestamp,
            "protocol": design_state.protocol_info.name,
            "version": design_state.protocol_info.version,
        },
    }

    # Extract configuration information (particularly loss config)
    if hasattr(design_state, "config_data") and design_state.config_data:
        config_data = design_state.config_data
        summary["config"] = {}
        
        # Add loss configuration weights
        if config_data.loss_config:
            summary["config"]["loss_weights"] = {}
            # Extract all weight parameters
            for key, value in config_data.loss_config.items():
                if key.startswith("weights_"):
                    summary["config"]["loss_weights"][key] = value
        
        # Add trajectory configuration
        if config_data.trajectory_config:
            summary["config"]["trajectory"] = {
                "optimizer_type": config_data.trajectory_config.get("optimizer_type"),
                "optimizer_learning_rate": config_data.trajectory_config.get("optimizer_learning_rate"),
                "optimizer_b1": config_data.trajectory_config.get("optimizer_b1"),
                "optimizer_b2": config_data.trajectory_config.get("optimizer_b2"),
                "soft_iters": config_data.trajectory_config.get("soft_iters"),
                "temp_iters": config_data.trajectory_config.get("temp_iters"),
                "hard_iters": config_data.trajectory_config.get("hard_iters"),
                "pssm_iters": config_data.trajectory_config.get("pssm_iters"),
            }
            # Remove None values
            summary["config"]["trajectory"] = {k: v for k, v in summary["config"]["trajectory"].items() if v is not None}
        
        # Add model configuration
        if config_data.model_config:
            summary["config"]["model"] = {
                "num_recycles_design": config_data.model_config.get("num_recycles_design"),
                "num_recycles_eval": config_data.model_config.get("num_recycles_eval"),
                "use_multimer_design": config_data.model_config.get("use_multimer_design"),
            }
            # Remove None values
            summary["config"]["model"] = {k: v for k, v in summary["config"]["model"].items() if v is not None}

    # Extract template information
    if hasattr(design_state, "template_data") and design_state.template_data:
        template_data = design_state.template_data
        summary["template"] = {
            "target_id": template_data.target_id,
            "target_name": template_data.target_name,
        }

        # Add optional template fields if available
        if template_data.region:
            summary["template"]["region"] = template_data.region
        if template_data.target_hotspot_residues:
            summary["template"]["hotspots"] = template_data.target_hotspot_residues
        if template_data.masked_binder_seq:
            summary["template"]["masked_sequence"] = template_data.masked_binder_seq
        if template_data.binder_len:
            summary["template"]["binder_length"] = template_data.binder_len

        # Add timing information
        if hasattr(template_data, "timings"):
            summary["template"]["timings"] = template_data.timings

    # Extract trajectory information
    if hasattr(design_state, "trajectory_data") and design_state.trajectory_data:
        trajectory_data = design_state.trajectory_data
        summary["trajectory"] = {
            "seed": trajectory_data.seed,
            "completed": trajectory_data.trajectory_complete,
        }

        # Add metrics information if available
        if trajectory_data.metrics:
            # Find the best metrics
            best_loss = float("inf")
            best_iptm = 0
            best_ptm_energy = float("inf")  # Initialize to positive infinity since we want the minimum (most negative)

            for step in trajectory_data.metrics:
                if "loss" in step and step["loss"] < best_loss:
                    best_loss = step["loss"]
                if "i_ptm" in step and step["i_ptm"] > best_iptm:
                    best_iptm = step["i_ptm"]
                if "ptm_energy" in step and step["ptm_energy"] < best_ptm_energy:  # Changed to < for minimum
                    best_ptm_energy = step["ptm_energy"]

            summary["trajectory"]["metrics"] = {
                "best_loss": best_loss,
                "best_iptm": best_iptm,
                "best_ptm_energy": best_ptm_energy,
            }

        # Add timing information
        if hasattr(trajectory_data, "timings"):
            summary["trajectory"]["timings"] = trajectory_data.timings

        # Add step timings if available
        if hasattr(trajectory_data, "step_timings"):
            summary["trajectory"]["step_timings"] = trajectory_data.step_timings

        # Add sequence information
        if trajectory_data.final_seqs:
            # Limit to at most 10 sequences to keep summary concise
            summary["trajectory"]["sequences"] = trajectory_data.final_seqs[:10]
            if len(trajectory_data.final_seqs) > 10:
                summary["trajectory"]["total_sequences"] = len(
                    trajectory_data.final_seqs
                )

    # Extract evaluation information
    if hasattr(design_state, "evaluation_data") and design_state.evaluation_data:
        evaluation_data = design_state.evaluation_data
        summary["evaluation"] = {
            "completed": evaluation_data.evaluation_complete,
        }

        # Add timing information
        if hasattr(evaluation_data, "timings"):
            summary["evaluation"]["timings"] = evaluation_data.timings

        # Add binder metrics if available
        if evaluation_data.binders:
            best_iptm = 0
            best_plddt = 0
            best_loss = float("inf")
            best_binder = None
            best_ptm_energy = float("inf")  # Initialize to positive infinity since we want the minimum (most negative)

            # Find the best binder
            for binder in evaluation_data.binders:
                # Calculate a composite score giving more weight to interface metrics
                quality_score = 0
                if hasattr(binder, "i_ptm") and binder.i_ptm is not None:
                    quality_score += binder.i_ptm * 2
                if hasattr(binder, "plddt") and binder.plddt is not None:
                    quality_score += binder.plddt

                # Track the best metrics
                if (
                    hasattr(binder, "i_ptm")
                    and binder.i_ptm is not None
                    and binder.i_ptm > best_iptm
                ):
                    best_iptm = binder.i_ptm
                if (
                    hasattr(binder, "plddt")
                    and binder.plddt is not None
                    and binder.plddt > best_plddt
                ):
                    best_plddt = binder.plddt
                if (
                    hasattr(binder, "ptm_energy")
                    and binder.ptm_energy is not None
                    and binder.ptm_energy < best_ptm_energy  # Changed to < for minimum
                ):
                    best_ptm_energy = binder.ptm_energy

                # Keep track of the best binder
                if quality_score > 0 and (
                    best_binder is None or quality_score > best_binder["quality_score"]
                ):
                    best_binder = {
                        "sequence": binder.binder_seq,
                        "i_ptm": getattr(binder, "i_ptm", None),
                        "plddt": getattr(binder, "plddt", None),
                        "i_pae": getattr(binder, "i_pae", None),
                        "esm_score": getattr(binder, "esm_score", None),
                        "quality_score": quality_score,
                        "ptm_energy": getattr(binder, "ptm_energy", None),
                        "hbond": getattr(binder, "hbond", None),
                        "salt_bridge": getattr(binder, "salt_bridge", None),
                    }

            summary["evaluation"]["metrics"] = {
                "best_iptm": best_iptm,
                "best_plddt": best_plddt,
                "best_ptm_energy": best_ptm_energy,
            }

            if best_binder:
                summary["evaluation"]["best_binder"] = best_binder

            # Add metrics for all binders
            summary["evaluation"]["binders"] = []
            for i, binder in enumerate(
                evaluation_data.binders[:5]
            ):  # Limit to 5 binders for conciseness
                binder_metrics = {
                    "sequence": binder.binder_seq,
                }

                # Add available metrics
                for metric in ["i_ptm", "plddt", "i_pae", "esm_score", "ptm_energy"]:
                    if hasattr(binder, metric) and getattr(binder, metric) is not None:
                        binder_metrics[metric] = getattr(binder, metric)

                summary["evaluation"]["binders"].append(binder_metrics)

            if len(evaluation_data.binders) > 5:
                summary["evaluation"]["total_binders"] = len(evaluation_data.binders)

    # Calculate total runtime
    total_runtime = 0
    for module in ["template", "trajectory", "evaluation"]:
        if module in summary and "timings" in summary[module]:
            for operation, runtime in summary[module]["timings"].items():
                if operation in ["setup", "run", "teardown"]:
                    total_runtime += runtime

    summary["total_runtime"] = total_runtime

    return summary


def generate_yaml_with_comments() -> str:
    """Generate a template string for the YAML file with comments."""
    return """# MBER Design Summary
# This file contains key metrics and information about the design run.
# Generated automatically by mber.utils.yaml_summary_utils

# Metadata about this design run
metadata:
  timestamp: {timestamp}
  protocol: {protocol}
  version: {version}

# Configuration parameters used for this run
config:
{config}

# Information about the template used for design
template:
{template}

# Trajectory optimization results
trajectory:
{trajectory}

# Evaluation results
evaluation:
{evaluation}

# Total design runtime in seconds
total_runtime: {runtime}
"""


def write_metrics_summary(design_state, output_path: Union[str, Path]) -> None:
    """
    Create and write a YAML summary of the most important metrics.

    Args:
        design_state: The completed design state
        output_path: Path to save the YAML summary
    """
    # Convert path to Path object if it's a string
    if isinstance(output_path, str):
        output_path = Path(output_path)

    # Create the metrics summary
    summary = create_metrics_summary(design_state)

    # Output to string
    yaml_str = yaml.dump(summary, default_flow_style=False, sort_keys=False)

    # Add comments by creating a template and formatting it manually
    template = generate_yaml_with_comments()

    # Extract sections
    metadata = summary.get("metadata", {})
    config_data = yaml.dump(
        summary.get("config", {}), default_flow_style=False, sort_keys=False
    )
    template_data = yaml.dump(
        summary.get("template", {}), default_flow_style=False, sort_keys=False
    )
    trajectory_data = yaml.dump(
        summary.get("trajectory", {}), default_flow_style=False, sort_keys=False
    )
    evaluation_data = yaml.dump(
        summary.get("evaluation", {}), default_flow_style=False, sort_keys=False
    )

    # Apply indentation to sections
    config_data = "\n".join(["  " + line for line in config_data.splitlines()])
    template_data = "\n".join(["  " + line for line in template_data.splitlines()])
    trajectory_data = "\n".join(["  " + line for line in trajectory_data.splitlines()])
    evaluation_data = "\n".join(["  " + line for line in evaluation_data.splitlines()])

    # Format the template
    result = template.format(
        timestamp=metadata.get("timestamp", ""),
        protocol=metadata.get("protocol", ""),
        version=metadata.get("version", ""),
        config=config_data,
        template=template_data,
        trajectory=trajectory_data,
        evaluation=evaluation_data,
        runtime=summary.get("total_runtime", 0),
    )

    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the YAML file
    with open(output_path, "w") as f:
        f.write(result)


def add_section_comments(yaml_str: str) -> str:
    """
    Add section comments to a YAML string.

    Args:
        yaml_str: YAML string to add comments to

    Returns:
        YAML string with comments
    """
    lines = yaml_str.splitlines()
    result = []

    # Add header
    result.append("# MBER Design Summary")
    result.append(
        "# This file contains key metrics and information about the design run."
    )
    result.append("# Generated automatically by mber.utils.yaml_summary_utils")
    result.append("")

    section_comments = {
        "metadata:": "\n# Metadata about this design run",
        "config:": "\n# Configuration parameters used for this run",
        "template:": "\n# Information about the template used for design",
        "trajectory:": "\n# Trajectory optimization results",
        "evaluation:": "\n# Evaluation results",
        "total_runtime:": "\n# Total design runtime in seconds",
    }

    for line in lines:
        # Check if this line is a section header
        for header, comment in section_comments.items():
            if line.strip() == header:
                result.append(comment)
                break

        result.append(line)

    return "\n".join(result)
