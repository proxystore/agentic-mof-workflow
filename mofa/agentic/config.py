from __future__ import annotations

import dataclasses
import functools
import pathlib

from mofa.model import LigandTemplate
from mofa.model import NodeDescription


@dataclasses.dataclass
class DatabaseConfig:
    run_dir: str
    mongo_host: str = "localhost"
    mongo_port: int = 27017


@dataclasses.dataclass
class GeneratorConfig:
    """Configuration for the generation tasks"""

    generator_path: pathlib.Path
    """Path to the base DiffLinker model"""
    model_dir: pathlib.Path
    """Dir for new DiffLinker models"""
    templates: list[LigandTemplate]
    """The templates being generated"""
    num_workers: int
    """Maximum workers to use for generation."""
    atom_counts: list[int]
    """Number of atoms within a linker to generate"""
    device: str
    """Torch device for inference."""
    num_samples: int
    """Number of molecules to generate at each size"""
    batch_size: int

    @functools.cached_property
    def anchor_types(self) -> set[str]:
        return {x.anchor_type for x in self.templates}


@dataclasses.dataclass
class TrainerConfig:
    """Configuration for retraining tasks"""

    num_epochs: int
    """Number of epochs to use for training"""
    minimum_train_size: int
    """Trigger retraining after these many computations have completed successfully"""
    maximum_train_size: int
    """How many of the top MOFs to train on"""
    best_fraction: float
    """Percentile of top MOFs to include in training set"""
    maximum_strain: float
    """Only use MOFs with strains below this value in training set"""
    config_path: str
    """Path to the generator training configuration"""
    retrain_dir: pathlib.Path
    """Retraining directory"""
    device: str
    """Torch device for training."""


@dataclasses.dataclass
class AssemblerConfig:
    max_queue_depth: int
    """Maximum ligands in assemble queue."""
    max_attempts: int
    """Maximum attempts to assemble a MOF."""
    min_ligand_candidates: int
    """Minimum number of candidates of each anchor needed before assembling MOFs"""
    num_mofs: int
    """Number of MOFs to create per assembly."""
    num_workers: int
    """Number of assembly workers."""
    node_templates: list[NodeDescription]
    """Nodes to be used for assembly."""
    run_dir: str


@dataclasses.dataclass
class ValidatorConfig:
    delete_finished: bool
    lammps_command: list[str]
    lammps_environ: dict[str, str]
    lmp_sims_root_path: str
    max_queue_depth: int
    """Maximum MOFs in validate queue."""
    num_workers: int
    """Number of validation workers."""
    report_frequency: int
    simulation_budget: int
    """Maximum number of simulations to perform before ending the workflow."""
    timesteps: int


@dataclasses.dataclass
class OptimizerConfig:
    cp2k_cmd: str
    cp2k_dir: pathlib.Path
    cp2k_steps: int
    num_workers: int
    run_dir: str


@dataclasses.dataclass
class EstimatorConfig:
    num_workers: int
    raspa_dir: pathlib.Path
    raspa_timesteps: int
    run_dir: str
