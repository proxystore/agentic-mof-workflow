from __future__ import annotations

import pathlib
from collections.abc import Sequence

import ase
from more_itertools import batched
from parsl import python_app

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


@python_app(executors=["generator"])
def generate_ligands_task(  # noqa: PLR0913
    model: str | pathlib.Path,
    templates: Sequence[LigandTemplate],
    batch_size: int,
    n_atoms: int | str = 8,
    n_samples: int = 1,
    n_steps: int | None = None,
    device: str = "cpu",
) -> tuple[list[LigandDescription], str]:
    valid_ligands: list[LigandDescription] = []
    assert len(templates) == 1
    anchor_type = templates[0].anchor_type

    generator = run_generator(
        model=model,
        templates=templates,
        n_atoms=n_atoms,
        n_samples=n_samples,
        n_steps=n_steps,
        device=device,
    )
    for batch in batched(generator, batch_size):
        ligands, _ = process_ligands(batch)
        valid_ligands.extend(ligands)

    return ligands, anchor_type


@python_app
def assemble_mofs_task(
    ligand_options: dict[str, Sequence[LigandDescription]],
    nodes: Sequence[NodeDescription],
    to_make: int,
    attempts: int,
) -> list[MOFRecord]:
    return assemble_many(ligand_options, nodes, to_make, attempts)


@python_app(executors=["validator"])
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


@python_app
def optimize_cells_and_compute_charges_task(
    runner: CP2KRunner,
    mof: MOFRecord,
    steps: int,
    fmax: float = 1e-2,
) -> tuple[MOFRecord, ase.Atoms]:
    # Note: we don't use the atoms yet.
    _, path = runner.run_optimization(mof, steps=steps, fmax=fmax)
    atoms = compute_partial_charges(path)
    return mof, atoms


@python_app
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


@python_app(executors=["generator"])
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
