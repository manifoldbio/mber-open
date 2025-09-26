import argparse
import time

from mber_protocols.stable.VHH_binder_design.config import ModelConfig, LossConfig, TrajectoryConfig, EnvironmentConfig, TemplateConfig, EvaluationConfig
from mber_protocols.stable.VHH_binder_design.template import TemplateModule
from mber_protocols.stable.VHH_binder_design.trajectory import TrajectoryModule
from mber_protocols.stable.VHH_binder_design.evaluation import EvaluationModule
from mber_protocols.stable.VHH_binder_design.state import DesignState, TemplateData

def parse_args():
    """
    Parse command line arguments for the VHH binder design protocol.

    Returns:
        Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="VHH Binder Design Protocol")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--target_id", type=str, required=True, help="Target protein ID (e.g., PDB ID)")
    parser.add_argument("--target_name", type=str, required=True, help="Name of the target protein")
    parser.add_argument("--masked_binder_seq", type=str, required=True, help="Masked sequence of the binder")
    parser.add_argument("--region", type=str, default=None, help="Region of the target protein to focus on")
    parser.add_argument("--target_hotspot_residues", type=str, default=None, help="Specific hotspot residues to use")

    return parser.parse_args()


def main(
    output_dir: str,
    target_id: str,
    target_name: str,
    masked_binder_seq: str,
    region: str = None,
    target_hotspot_residues: str = None
):
    """
    Run the VHH binder design protocol. This implementation follows the same process as the example in `template_trajectory_evaluation_example_PDL1.ipynb`.
    """
    start_time = time.time()
    
    template_config = TemplateConfig()
    model_config = ModelConfig()
    loss_config = LossConfig()
    trajectory_config = TrajectoryConfig()
    environment_config = EnvironmentConfig()
    evaluation_config = EvaluationConfig()
    
    
    design_state = DesignState(
        template_data=TemplateData(
            target_id=target_id,
            target_name=target_name,
            masked_binder_seq=masked_binder_seq,
            region=region,
            target_hotspot_residues=target_hotspot_residues
        )
    )

    template_module = TemplateModule(
        template_config=template_config,
        environment_config=environment_config,
    )

    template_module.setup(design_state)
    design_state = template_module.run(design_state)
    template_module.teardown(design_state)

    trajectory_module = TrajectoryModule(
        model_config=model_config,
        loss_config=loss_config,
        trajectory_config=trajectory_config,
        environment_config=environment_config,
    )

    trajectory_module.setup(design_state)
    design_state = trajectory_module.run(design_state)
    trajectory_module.teardown(design_state)

    evaluation_module = EvaluationModule(
        model_config=model_config,
        evaluation_config=evaluation_config,
        loss_config=loss_config,
        environment_config=environment_config,
    )

    evaluation_module.setup(design_state)
    design_state = evaluation_module.run(design_state)
    evaluation_module.teardown(design_state)

    design_state.to_dir(output_dir)

    print(f"Design completed for {target_name} and saved to {output_dir}. Design time: {time.time() - start_time} seconds")

    return