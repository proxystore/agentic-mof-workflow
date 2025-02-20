from __future__ import annotations

import pathlib
from collections.abc import Generator
from collections.abc import Sequence

import ase
import ray
from more_itertools import batched

from mofa.assembly.assemble import assemble_many
from mofa.assembly.validate import process_ligands
from mofa.generator import run_generator
from mofa.generator import train_generator
from mofa.model import LigandDescription
from mofa.model import LigandTemplate
from mofa.model import MOFRecord
from mofa.model import NodeDescription
from mofa.simulation.cp2k import compute_partial_charges
from mofa.simulation.cp2k import CP2KRunner
from mofa.simulation.lammps import LAMMPSRunner
from mofa.simulation.raspa import RASPARunner


@ray.remote(num_cpus=4, num_gpus=1)
def generate_ligands_task(  # noqa: PLR0913
    model: str | pathlib.Path,
    templates: Sequence[LigandTemplate],
    batch_size: int,
    n_atoms: int | str = 8,
    n_samples: int = 1,
    n_steps: int | None = None,
    device: str = "cpu",
) -> Generator[tuple[LigandDescription, ...], None, None]:
    generator = run_generator(
        model=model,
        templates=templates,
        n_atoms=n_atoms,
        n_samples=n_samples,
        n_steps=n_steps,
        device=device,
    )
    yield from batched(generator, batch_size)


@ray.remote(num_cpus=1)
def process_ligands_task(
    ligands: Sequence[LigandDescription],
) -> tuple[list[LigandDescription], list[dict]]:
    return process_ligands(ligands)


@ray.remote(num_cpus=1)
def assemble_mofs_task(
    ligand_options: dict[str, Sequence[LigandDescription]],
    nodes: Sequence[NodeDescription],
    to_make: int,
    attempts: int,
) -> list[MOFRecord]:
    return assemble_many(ligand_options, nodes, to_make, attempts)


@ray.remote(num_cpus=4, num_gpus=0.5)
def validate_structure_task(
    runner: LAMMPSRunner,
    mof: MOFRecord,
    timesteps: int,
    report_frequency: int,
) -> tuple[MOFRecord, list[ase.Atoms]]:
    return mof, runner.run_molecular_dynamics(
        mof,
        timesteps=timesteps,
        report_frequency=report_frequency,
    )


# TODO: figure out resource usage
@ray.remote(num_gpus=1, num_cpus=8)
def optimize_cells_task(
    runner: CP2KRunner,
    mof: MOFRecord,
    steps: int,
) -> tuple[str, pathlib.Path]:
    # Note: we don't use the atoms yet.
    _, path = runner.run_optimization(mof, steps=steps)
    return mof.name, path


@ray.remote(num_cpus=1)
def compute_partial_charges_task(
    name: str,
    cp2k_path: pathlib.Path,
) -> tuple[str, ase.Atoms]:
    atoms = compute_partial_charges(cp2k_path)
    return name, atoms


@ray.remote(num_cpus=1)
def estimate_adsorption_task(
    runner: RASPARunner,
    atoms: ase.Atoms,
    name: str,
    timesteps: int,
) -> tuple[str, float, float]:
    gas_ads_mean, gas_ads_std = runner.run_GCMC_single(
        atoms,
        name,
        timesteps=timesteps,
    )
    return name, gas_ads_mean, gas_ads_std


@ray.remote(num_cpus=8, num_gpus=1)
def retrain_task(  # noqa: PLR0913
    *,
    starting_model: str | pathlib.Path | None,
    run_directory: pathlib.Path,
    config_path: str | pathlib.Path,
    examples: list[MOFRecord],
    num_epochs: int = 10,
    device: str = "cpu",
    strategy: str | None = None,
) -> pathlib.Path:
    return train_generator(
        starting_model=starting_model,
        run_directory=run_directory,
        config_path=config_path,
        examples=examples,
        num_epochs=num_epochs,
        device=device,
        strategy=strategy,
    )
