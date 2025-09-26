"""
Base module for evaluating designed protein binders.
This provides the core functionality that specific evaluation modules can inherit.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Union, Callable, Tuple, Any
import torch

from mber.core.modules.base import BaseModule
from mber.core.modules.config import (
    BaseModelConfig,
    BaseLossConfig,
    BaseEvaluationConfig,
    BaseEnvironmentConfig,
)
from mber.core.data.state import DesignState, BinderData, EvaluationData
from mber.models.colabdesign.model import AFModel
from mber.models.colabdesign.loss import (
    add_rg_loss,
    add_i_ptm_loss,
    add_hbond_loss,
    add_salt_bridge_loss,
)
from mber.models.plm import PLM_MODELS
from mber.models.folding import FOLDING_MODELS
from mber.models.colabfold.relax import relax_me
from mber.utils.timing_utils import timer, time_method
from colabdesign.shared.utils import copy_dict

class BaseEvaluationModule(BaseModule):
    """
    Base module for evaluating designed protein binders.
    This provides the core functionality that specific evaluation modules can inherit.
    """

    def __init__(
        self,
        model_config: BaseModelConfig,
        loss_config: BaseLossConfig,
        evaluation_config: BaseEvaluationConfig,
        environment_config: BaseEnvironmentConfig,
    ) -> None:
        # Initialize the base module
        super().__init__(
            config=evaluation_config,
            environment_config=environment_config,
            verbose=True,
            name="EvaluationModule",
        )

        # Store configs for specific access
        self.model_config = model_config
        self.loss_config = loss_config
        self.evaluation_config = evaluation_config

        # Initialize member variables
        self.af_complex_model = None
        self.esm_model = None
        self.pssm_logits = None

    @time_method()
    def setup(self, design_state: DesignState) -> None:
        """Set up models and evaluation parameters."""
        self._setup_logging()

        with timer("Setup models", self._log, design_state.evaluation_data.timings):
            self._setup_models(design_state)

        with timer(
            "Setup evaluation parameters",
            self._log,
            design_state.evaluation_data.timings,
        ):
            self._setup_evaluation(design_state)

        with timer(
            "Setup loss functions", self._log, design_state.evaluation_data.timings
        ):
            self._setup_loss()

    @time_method()
    def run(self, design_state: DesignState) -> DesignState:
        """Run the evaluation on each binder sequence."""
        self._log(
            f"Starting evaluation of {len(design_state.trajectory_data.final_seqs)} binder sequences"
        )

        for i, binder_seq in enumerate(design_state.trajectory_data.final_seqs):
            self._evaluate_binder_sequence(design_state, binder_seq, i)

        return design_state

    @time_method()
    def teardown(self, design_state: DesignState) -> None:
        """Clean up resources and update design state."""
        with timer(
            "Update design state", self._log, design_state.evaluation_data.timings
        ):
            self._update_design_state(design_state)

        self._stop_logging(design_state)
        self._save_configuration_data(design_state)

        self._log("Evaluation module teardown complete")
        design_state.evaluation_data.evaluation_complete = True

    @time_method()
    def _setup_models(self, design_state: DesignState) -> None:
        """Initialize models needed for evaluation."""
        with timer(
            "Initialize AF complex model",
            self._log,
            design_state.evaluation_data.timings,
        ):
            self._log("Initializing AlphaFold model for complex evaluation")
            self.af_complex_model = AFModel(
                protocol="binder",
                debug=False,
                data_dir=self.environment_config.af_params_dir,
                use_multimer=self.model_config.use_multimer_eval,
                num_recycles=self.model_config.num_recycles_eval,
            )

        with timer(
            "Initialize ESM model", self._log, design_state.evaluation_data.timings
        ):
            self._log("Initializing ESM model for sequence evaluation")
            self.esm_model = PLM_MODELS[self.evaluation_config.plm_model](device="cpu")

        # Add monomer folding model initialization
        with timer(
            "Initialize monomer folding model",
            self._log,
            design_state.evaluation_data.timings,
        ):
            self._log(
                f"Initializing monomer folding model: {self.evaluation_config.monomer_folding_model}"
            )
            self.monomer_folding_model = FOLDING_MODELS[
                self.evaluation_config.monomer_folding_model
            ]()

    @time_method()
    def _fold_monomer(self, design_state: DesignState, binder_seq: str) -> str:
        """Fold binder sequence as a monomer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir_path = Path(tmpdir)
            # Fold the binder structure
            from mber.utils.pdb_utils import fold_binder

            monomer_pdb = fold_binder(
                self.monomer_folding_model, binder_seq, output_dir=temp_dir_path
            )
        return monomer_pdb

    def _evaluate_binder_sequence(
        self, design_state: DesignState, binder_seq: str, index: int
    ) -> None:
        """Evaluate a single binder sequence."""
        self._log(
            f"Evaluating binder sequence {index+1}/{len(design_state.trajectory_data.final_seqs)}"
        )

        # Create timing dictionary for this binder
        binder_timings = {}

        with timer(f"Predict binder {index+1} complex", self._log, binder_timings):
            binder_data = self._predict_complex(
                design_state=design_state, binder_seq=binder_seq
            )

        # Add monomer folding
        with timer(f"Fold binder {index+1} monomer", self._log, binder_timings):
            binder_data.monomer_pdb = self._fold_monomer(
                design_state=design_state, binder_seq=binder_seq
            )

        with timer(
            f"Calculate ESM score for binder {index+1}", self._log, binder_timings
        ):
            binder_data.esm_score = self._get_esm_score(
                design_state=design_state, binder_seq=binder_seq
            )

        # Store the timing information in the binder data
        binder_data.timings = binder_timings

        # Add to evaluation data
        design_state.evaluation_data.binders.append(binder_data)

    def _setup_loss(self) -> None:
        """Setup and configure loss functions for evaluation."""
        # Configure main weight parameters
        
        self.af_complex_model.opt["weights"].update(    {
                "dgram_cce": 0.0,
                "rmsd": 0.0,
                "fape": 0.0,
                "con": self.loss_config.weights_con_intra,
                "i_con": self.loss_config.weights_con_inter,
                "pae": self.loss_config.weights_pae_intra,
                "i_pae": self.loss_config.weights_pae_inter,
                "rg": self.loss_config.weights_rg,
                "iptm": self.loss_config.weights_iptm,
                "plddt": self.loss_config.weights_plddt,
                "seq_ent": self.loss_config.weights_seq_ent,
                "hbond": self.loss_config.weights_hbond,
                "salt_bridge": self.loss_config.weights_salt_bridge,
            }
        )

        # Configure contact parameters
        self.af_complex_model.opt["con"].update(
            {
                "num": self.loss_config.intra_contact_number,
                "cutoff": self.loss_config.intra_contact_distance,
                "binary": False,
                "seqsep": 9,
            }
        )

        self.af_complex_model.opt["i_con"].update(
            {
                "num": self.loss_config.inter_contact_number,
                "cutoff": self.loss_config.inter_contact_distance,
                "binary": False,
            }
        )

        # Add additional loss functions
        add_rg_loss(self.af_complex_model, self.loss_config.weights_rg)
        add_i_ptm_loss(self.af_complex_model, self.loss_config.weights_iptm)
        add_hbond_loss(self.af_complex_model, self.loss_config.weights_hbond)
        add_salt_bridge_loss(
            self.af_complex_model, self.loss_config.weights_salt_bridge
        )

    @time_method()
    def _setup_evaluation(self, design_state: DesignState) -> None:
        """Set up the evaluation parameters and prepare models."""
        # Write template PDB to temporary file
        with tempfile.NamedTemporaryFile(suffix=".pdb") as template_pdb:
            # Write template_pdb to tmpfile as raw text
            template_pdb.write(design_state.template_data.template_pdb.encode())
            template_pdb.flush()
            pdb_filename = template_pdb.name

            # Determine which positions should be flexible or fixed
            rm_binder_template = design_state.template_data.get_flex_pos(as_array=False)
            rm_binder_seq = design_state.template_data.get_flex_pos(as_array=False)
            rm_binder_sc = design_state.template_data.get_flex_pos(as_array=False)

            # Initialize AF model with the template
            self.af_complex_model._prep_binder(
                pdb_filename=pdb_filename,
                chain=design_state.template_data.target_chain,
                binder_chain=design_state.template_data.binder_chain,
                binder_len=design_state.template_data.binder_len,
                hotspot=design_state.template_data.target_hotspot_residues,
                seed=design_state.trajectory_data.seed,
                rm_target=False,
                rm_target_seq=False,
                rm_target_sc=False,
                rm_template_ic=True,
                rm_binder=rm_binder_template,
                rm_binder_seq=rm_binder_seq,
                rm_binder_sc=rm_binder_sc,
            )

    @time_method()
    def _predict_complex(
        self, design_state: DesignState, binder_seq: str
    ) -> BinderData:
        """Predict and evaluate the protein complex."""
        # Run AlphaFold prediction
        self.af_complex_model.predict(
            seq=binder_seq,
            models=self.model_config.eval_models,
        )

        # Get PDB output
        complex_pdb = self.af_complex_model.save_pdb()

        # Relax structure
        relaxed_pdb, relax_data, relax_violations = self._relax_pdb(
            design_state=design_state, pdb_string=complex_pdb
        )

        # Extract evaluation metrics
        evaluation_metrics = copy_dict(self.af_complex_model.aux["log"])

        # Create and return binder data
        return BinderData(
            binder_seq=binder_seq,
            complex_pdb=complex_pdb,
            relaxed_pdb=relaxed_pdb,
            plddt=evaluation_metrics["plddt"],
            ptm=evaluation_metrics["ptm"],
            i_ptm=evaluation_metrics["i_ptm"],
            pae=evaluation_metrics["pae"],
            i_pae=evaluation_metrics["i_pae"],
            seq_ent=evaluation_metrics["seq_ent"],
            unrelaxed_energy=relax_data["initial_energy"],
            relaxed_energy=relax_data["final_energy"],
            relax_rmsd=relax_data["rmsd"],
        )

    @time_method()
    def _relax_pdb(self, design_state: DesignState, pdb_string: str) -> tuple:
        """Relax PDB structure using the AMBER force field."""
        with tempfile.NamedTemporaryFile(suffix=".pdb") as input_pdb:
            input_pdb.write(pdb_string.encode())
            input_pdb.flush()

            # Run relaxation with configurable parameters
            relaxed_pdb, relax_data, relax_violations = relax_me(
                pdb_filename=input_pdb.name,
                use_gpu=self.evaluation_config.use_gpu,
                max_iterations=self.evaluation_config.max_iterations,
                tolerance=self.evaluation_config.tolerance,
                stiffness=self.evaluation_config.stiffness,
                max_outer_iterations=self.evaluation_config.max_outer_iterations,
            )

        return relaxed_pdb, relax_data, relax_violations

    @time_method()
    def _get_esm_score(self, design_state: DesignState, binder_seq: str) -> float:
        """Calculate ESM model score for a binder sequence."""
        return self.esm_model.get_unmask_cce_score(binder_seq)

    @time_method()
    def _update_design_state(self, design_state: DesignState) -> None:
        """Update design state with evaluation results."""
        design_state.evaluation_data.evaluation_complete = True
