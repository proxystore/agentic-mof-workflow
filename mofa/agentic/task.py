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
from mofa.model import LigandDescription
from mofa.model import LigandTemplate
from mofa.model import MOFRecord
from mofa.model import NodeDescription
from mofa.simulation.lammps import LAMMPSRunner


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
