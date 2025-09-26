"""
Base module for trajectory optimization in protein design.
This provides the core functionality that specific trajectory modules can inherit.
"""

import copy
import os
import numpy as np
import jax
from typing import Optional, List, Dict, Union, Callable, Tuple, Any
import torch

from mber.core.modules.base import BaseModule
from mber.core.modules.config import (
    BaseModelConfig,
    BaseLossConfig,
    BaseTrajectoryConfig,
    BaseEnvironmentConfig,
)
from mber.core.data.state import DesignState, TrajectoryData
from mber.models.colabdesign.model import AFModel
from mber.models.colabdesign.loss import (
    add_rg_loss,
    add_i_ptm_loss,
    add_hbond_loss,
    add_salt_bridge_loss,
)
from mber.models.plm import PLM_MODELS
from mber.utils.plm_utils import generate_bias_unmasked
from mber.utils.timing_utils import timer, time_method


class BaseTrajectoryModule(BaseModule):
    """
    Base module for trajectory optimization in protein design.
    This provides the core functionality that specific trajectory modules can inherit.
    """

    def __init__(
        self,
        model_config: BaseModelConfig,
        loss_config: BaseLossConfig,
        trajectory_config: BaseTrajectoryConfig,
        environment_config: BaseEnvironmentConfig,
    ) -> None:
        # Initialize the base module
        super().__init__(
            config=trajectory_config,
            environment_config=environment_config,
            verbose=True,
            name="TrajectoryModule"
        )
        
        # Store configs for specific access
        self.model_config = model_config
        self.loss_config = loss_config
        self.trajectory_config = trajectory_config

        # Initialize member variables
        self.af_model = None
        self.esm_model = None
        self.pssm_logits = None

    @time_method()
    def setup(
        self,
        design_state: DesignState,
        debug: bool = False,
        force_model_setup: bool = False,
    ) -> None:
        """Set up models, trajectories, and loss functions."""
        self._setup_logging()

        if self.af_model is None or force_model_setup:
            with timer("Setup models", self._log, design_state.trajectory_data.timings):
                self._setup_models(design_state, debug=debug)

        with timer("Setup trajectory", self._log, design_state.trajectory_data.timings):
            self._setup_trajectory(design_state)

        with timer("Setup optimizer", self._log, design_state.trajectory_data.timings):
            self._setup_optimizer(design_state)

        with timer("Setup loss", self._log, design_state.trajectory_data.timings):
            self._setup_loss()

        # Precompile model to avoid JIT delays during design
        with timer("Precompile model", self._log, design_state.trajectory_data.timings):
            self._precompile_model()

    @time_method()
    def run(self, design_state: DesignState) -> DesignState:
        """Run the trajectory optimization."""
        # Perform initial design step
        with timer(
            "Design logits phase", self._log, design_state.trajectory_data.step_timings
        ):
            continue_signal = self._design_logits(design_state=design_state)

        if not continue_signal:
            self._handle_early_stopping(design_state)
            return design_state

        # Sample designs using PSSM from logits + bias
        with timer(
            "Sample designs PSSM phase",
            self._log,
            design_state.trajectory_data.step_timings,
        ):
            self._sample_designs_pssm(
                design_state=design_state,
                update_esm_bias=self.trajectory_config.update_esm_bias,
            )

        return design_state

    @time_method()
    def teardown(self, design_state: DesignState) -> None:
        """Clean up resources and update design state."""
        with timer(
            "Update design state", self._log, design_state.trajectory_data.timings
        ):
            self._update_design_state(design_state)

        self._stop_logging(design_state)
        self._save_configuration_data(design_state)

        self._log("Trajectory module teardown complete")

    @time_method()
    def _setup_optimizer(self, design_state: DesignState) -> None:
        """Configure and set up the optimizer based on TrajectoryConfig."""
        optimizer_type = self.trajectory_config.optimizer_type
        learning_rate = self.trajectory_config.optimizer_learning_rate

        # Log optimizer configuration
        self._log(f"Configuring optimizer: {optimizer_type} with lr={learning_rate}")

        # Extract optimizer parameters from trajectory config
        optimizer_params = {
            "b1": self.trajectory_config.optimizer_b1,
            "b2": self.trajectory_config.optimizer_b2,
            "eps": self.trajectory_config.optimizer_eps,
            "weight_decay": self.trajectory_config.optimizer_weight_decay,
        }
        
        # Add schedule-free specific parameters when relevant
        if "schedule_free" in optimizer_type:
            optimizer_params.update({
                "weight_lr_power": self.trajectory_config.optimizer_weight_lr_power,
                "warmup_steps": self.trajectory_config.optimizer_warmup_steps,
            })
        
        # Use unified optimizer setup method
        self.af_model.set_optimizer(
            optimizer=optimizer_type,
            learning_rate=learning_rate,
            **{k: v for k, v in optimizer_params.items() if v is not None}
        )
        
        self._log(f"Configured optimizer: {optimizer_type}")

    @time_method()
    def _setup_models(self, design_state: DesignState, debug: bool = False) -> None:
        """Initialize models needed for trajectory optimization."""
        with timer(
            "Initialize AF model", self._log, design_state.trajectory_data.timings
        ):
            self._log("Initializing AlphaFold model for design")
            self.af_model = AFModel(
                protocol="binder",
                debug=debug,
                data_dir=self.environment_config.af_params_dir,
                use_multimer=self.model_config.use_multimer_design,
                num_recycles=self.model_config.num_recycles_design,
                best_metric="loss",
            )

        if self.trajectory_config.update_esm_bias:
            with timer(
                "Initialize ESM model", self._log, design_state.trajectory_data.timings
            ):
                self._log("Initializing ESM model for bias updates")
                self.esm_model = PLM_MODELS[self.trajectory_config.plm_model](device="cpu")

    def _setup_loss(self) -> None:
        """Setup and configure loss functions."""
        # Configure main weight parameters
        self.af_model.opt["weights"].update(
            {
                "dgram_cce": 0.0,
                "rmsd": 0.0,
                "fape": 0.0,
                "plddt": self.loss_config.weights_plddt,
                "con": self.loss_config.weights_con_intra,
                "i_con": self.loss_config.weights_con_inter,
                "pae": self.loss_config.weights_pae_intra,
                "i_pae": self.loss_config.weights_pae_inter,
                "rg": self.loss_config.weights_rg,
                "hbond": self.loss_config.weights_hbond,
                "salt_bridge": self.loss_config.weights_salt_bridge,
                "seq_ent": self.loss_config.weights_seq_ent,
            }
        )

        # Configure contact parameters
        self.af_model.opt["con"].update(
            {
                "num": self.loss_config.intra_contact_number,
                "cutoff": self.loss_config.intra_contact_distance,
                "binary": False,
                "seqsep": 9,
            }
        )

        self.af_model.opt["i_con"].update(
            {
                "num": self.loss_config.inter_contact_number,
                "cutoff": self.loss_config.inter_contact_distance,
                "binary": False,
            }
        )

        # Add additional loss functions
        add_rg_loss(self.af_model, self.loss_config.weights_rg)
        add_i_ptm_loss(self.af_model, self.loss_config.weights_iptm)
        add_hbond_loss(self.af_model, self.loss_config.weights_hbond)
        add_salt_bridge_loss(self.af_model, self.loss_config.weights_salt_bridge)

    @time_method()
    def _setup_trajectory(self, design_state: DesignState) -> None:
        """Set up trajectory optimization parameters."""
        # Initialize trajectory seed and name if not present
        if not design_state.trajectory_data.seed:
            seed = np.random.randint(1e6, 1e7 - 1)
            design_state.trajectory_data.seed = seed
            self._log(f"Generated random seed for trajectory: {seed}", level="info")

        if not design_state.trajectory_data.trajectory_name:
            name = f"{design_state.template_data.target_name}_{design_state.trajectory_data.seed}"
            design_state.trajectory_data.trajectory_name = name
            self._log(f"Set trajectory name: {name}", level="info")

        # Create PDB file from template
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pdb") as template_pdb:
            # Write template_pdb to tmpfile as raw text
            template_pdb.write(design_state.template_data.template_pdb.encode())
            template_pdb.flush()
            pdb_filename = template_pdb.name

            # Configure masking options based on trajectory config
            rm_binder_template = (
                design_state.template_data.get_flex_pos(as_array=False)
                if self.trajectory_config.mask_binder_variable_template
                else False
            )

            rm_binder_seq = (
                True
                if self.trajectory_config.mask_binder_seq
                else design_state.template_data.get_flex_pos(as_array=False)
            )

            rm_binder_sc = (
                True
                if self.trajectory_config.mask_binder_sidechains
                else design_state.template_data.get_flex_pos(as_array=False)
            )

            if self.trajectory_config.mask_binder:
                rm_binder_template = True
                rm_binder_seq = True
                rm_binder_sc = True

            # Initialize AF model with the template
            
            # Now that we've renamed all chains in the PDB files to H, we use the proper chain IDs
            self.af_model._prep_binder(
                pdb_filename=pdb_filename,
                chain=design_state.template_data.target_chain, 
                binder_chain=design_state.template_data.binder_chain,
                binder_len=design_state.template_data.binder_len,
                hotspot=design_state.template_data.target_hotspot_residues,
                seed=design_state.trajectory_data.seed,
                rm_aa=self.trajectory_config.rm_aa,
                rm_target=self.trajectory_config.mask_target,
                rm_target_seq=self.trajectory_config.mask_target_seq,
                rm_target_sc=self.trajectory_config.mask_target_sidechains,
                rm_template_ic=True,
                rm_binder=rm_binder_template,
                rm_binder_seq=rm_binder_seq,
                rm_binder_sc=rm_binder_sc,
            )

        # Fix unmasked residues for pssm step
        self.af_model.opt["fix_binder_pos"] = (
            design_state.template_data.get_fix_pos() - 1
        )

        # Convert binder bias to sequence probabilities
        seq_probs = np.exp(design_state.template_data.binder_bias) / np.sum(
            np.exp(design_state.template_data.binder_bias), axis=1, keepdims=True
        )

        # Set initial sequence and bias
        self.af_model.set_seq(
            seq=seq_probs, bias=design_state.template_data.binder_bias
        )

    @time_method()
    def _precompile_model(self, design_state: Optional[DesignState] = None) -> None:
        """Precompile model with dummy inputs to avoid JIT delays during design."""
        self._log("Precompiling model for faster execution...")

        # Make a copy of current inputs to restore later
        original_inputs = self.af_model._inputs.copy()
        original_params = self.af_model._params.copy()

        # Run a dummy forward pass with backpropagation
        try:
            # Use a small number of iterations to trigger compilation
            self.af_model.design(
                iters=1,
                models=self.model_config.design_models,
                num_models=1,
            )
            self._log("Model precompilation complete")
        except Exception as e:
            self._log(
                f"Precompilation failed (but we can continue): {e}", level="warning"
            )
        finally:
            # Restore original state
            self.af_model._inputs = original_inputs
            self.af_model._params = original_params

    def _handle_early_stopping(self, design_state: DesignState) -> None:
        """Handle early stopping due to low iPTM."""
        self._log(
            "Early stopping design due to low iPTM. Exiting run phase.", level="warning"
        )
        design_state.trajectory_data.early_stop = True
        design_state.trajectory_data.trajectory_complete = True

    @time_method()
    def _design_logits(self, design_state: DesignState) -> bool:
        """Run the design optimization phase to generate logits."""
        # First part of design with early stopping check
        self.af_model.design(
            iters=int(
                self.trajectory_config.soft_iters
                * self.trajectory_config.early_stop_fraction
            ),
            soft=0,
            e_soft=self.trajectory_config.early_stop_fraction,
            models=self.model_config.design_models,
            num_models=1,
            sample_models=True,
            save_best=True,
        )

        # Early stopping check
        if self.trajectory_config.early_stopping:
            best_iptm = self.af_model._tmp["best"]["aux"]["i_ptm"]
            if best_iptm < self.trajectory_config.early_stop_iptm:
                self._log(
                    f"Early stopping design due to low iPTM: {best_iptm:.4f}",
                    level="warning",
                )
                return False

        # Continue with remaining soft iterations
        self.af_model.design(
            iters=int(
                self.trajectory_config.soft_iters
                * (1 - self.trajectory_config.early_stop_fraction)
            ),
            soft=self.trajectory_config.early_stop_fraction,
            e_soft=1,
            models=self.model_config.design_models,
            num_models=1,
            sample_models=True,
            save_best=True,
        )

        # Temperature-based design phase
        self.af_model.design(
            iters=self.trajectory_config.temp_iters,
            soft=1,
            temp=1,
            e_temp=1e-2,
            models=self.model_config.design_models,
            num_models=1,
            sample_models=True,
            save_best=True,
        )

        # Hard design phase
        if self.trajectory_config.hard_iters > 0:
            self.af_model.design(
                iters=self.trajectory_config.hard_iters,
                soft=1,
                hard=1,
                temp=1e-2,
                num_models=1,
                sample_models=True,
                dropout=False,
                save_best=True,
            )

        return True

    @time_method()
    def _sample_designs_pssm(
        self, design_state: DesignState, update_esm_bias: bool = True
    ) -> None:
        """Sample design sequences using PSSM-based generation."""
        if update_esm_bias and hasattr(self, "esm_model") and self.esm_model:
            last_seq = self.af_model.get_trajectory_seqs(as_str=True)[-1]
            self._log(f"Updating ESM bias from last sequence: {last_seq}")

            design_state.trajectory_data.updated_bias = generate_bias_unmasked(
                last_seq,
                model=self.esm_model,
                template_data=design_state.template_data,
                omit_aas=self.trajectory_config.rm_aa,
            )

            self.af_model.set_seq(
                seq=last_seq, bias=design_state.trajectory_data.updated_bias
            )

        # Get best PSSM logits
        self.pssm_logits = np.squeeze(
            self.af_model._tmp["best"]["aux"]["seq"]["logits"], axis=0
        )

        # Clear best to take only PSSM PDB
        self.af_model.clear_best()

        # Run PSSM-based design
        self.af_model.design_pssm_semigreedy(
            soft_iters=0,
            hard_iters=self.trajectory_config.pssm_iters,
            tries=self.trajectory_config.greedy_tries,
            models=self.model_config.design_models,
            num_models=1,
            sample_models=True,
            ramp_models=False,
            save_best=True,
            seq_logits=self.af_model.aux["seq"]["logits"],
        )

    @time_method()
    def _update_design_state(self, design_state: DesignState) -> None:
        """Update design state with trajectory results."""
        # Get trajectory metrics
        trajectory_metrics = self.af_model.get_trajectory_metrics()

        # Convert any numpy arrays to lists for serialization
        for step in trajectory_metrics:
            for key, value in step.items():
                if isinstance(value, np.ndarray):
                    step[key] = value.tolist()

        design_state.trajectory_data.metrics = trajectory_metrics

        # Get final sequences (unique PSSM sequences)
        pssm_seqs = self.af_model.get_trajectory_seqs(as_str=True)[
            -self.trajectory_config.pssm_iters :
        ]
        unique_pssm_seqs = list(set(pssm_seqs))
        design_state.trajectory_data.final_seqs = unique_pssm_seqs

        # Get best PDB and other outputs
        design_state.trajectory_data.best_pdb = self.af_model.save_pdb()
        design_state.trajectory_data.pssm_logits = self.pssm_logits
        design_state.trajectory_data.animated_trajectory = self.af_model.animate()

        design_state.trajectory_data.trajectory_complete = True