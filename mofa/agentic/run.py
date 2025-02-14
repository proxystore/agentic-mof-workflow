from __future__ import annotations

from datetime import datetime
import argparse
import pathlib
import logging
import hashlib
import json
import sys

from aeris.launcher.thread import ThreadLauncher
from aeris.exchange.thread import ThreadExchange
from aeris.manager import Manager
import ray
from rdkit import RDLogger
from openbabel import openbabel

from mofa.model import NodeDescription, LigandTemplate
from mofa.agentic.compute import COMPUTE_CONFIGS
from mofa.agentic.config import GeneratorConfig, ValidatorConfig, AssemblerConfig
from mofa.agentic.steering import Generator, Assembler, Validator

RDLogger.DisableLog("rdApp.*")
openbabel.obErrorLog.SetOutputLevel(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--simulation-budget",
        type=int,
        help="Number of simulations to submit before exiting",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    group = parser.add_argument_group(
        title="MOF Settings",
        description="Options related to the MOF type being generated",
    )
    group.add_argument("--node-path", required=True, help="Path to a node record")

    group = parser.add_argument_group(
        title="Generator Settings",
        description="Options related to how the generation is performed",
    )
    group.add_argument(
        "--ligand-templates",
        required=True,
        nargs="+",
        help="Path to YAML files containing a description of the ligands to be created",
    )
    group.add_argument(
        "--generator-path",
        required=True,
        help="Path to the PyTorch files describing model architecture and weights",
    )
    group.add_argument(
        "--molecule-sizes",
        nargs="+",
        type=int,
        default=list(range(6, 21)),
        help="Sizes of molecules we should generate",
    )
    group.add_argument(
        "--num-samples",
        type=int,
        default=16,
        help="Number of molecules to generate at each size",
    )
    group.add_argument(
        "--gen-batch-size",
        type=int,
        default=16,
        help="Number of ligands to stream per batch",
    )

    group = parser.add_argument_group(
        "Retraining Settings", description="How often to retain, what to train on, etc"
    )
    group.add_argument(
        "--generator-config-path",
        required=True,
        help="Path to the generator training configuration",
    )
    group.add_argument(
        "--retrain-freq",
        type=int,
        default=8,
        help="Trigger retraining after these many successful computations",
    )
    group.add_argument(
        "--maximum-train-size",
        type=int,
        default=256,
        help="Maximum number of MOFs to use for retraining",
    )
    group.add_argument(
        "--num-epochs", type=int, default=128, help="Number of training epochs"
    )
    group.add_argument(
        "--best-fraction",
        type=float,
        default=0.5,
        help="What percentile of MOFs to include in training",
    )
    group.add_argument(
        "--maximum-strain",
        type=float,
        default=0.5,
        help="Maximum strain allowed MOF used in training set",
    )

    group = parser.add_argument_group(
        title="Assembly Settings", description="Options related to MOF assembly"
    )
    group.add_argument(
        "--max-assemble-attempts",
        default=100,
        help="Maximum number of attempts to create a MOF",
    )
    group.add_argument(
        "--minimum-ligand-pool",
        type=int,
        default=4,
        help="Minimum number of ligands before MOF assembly",
    )

    group = parser.add_argument_group(
        title="Simulation Settings Settings",
        description="Options related to property calculations",
    )
    group.add_argument(
        "--md-timesteps",
        default=100000,
        help="Number of timesteps for the UFF MD simulation",
        type=int,
    )
    group.add_argument(
        "--md-snapshots",
        default=100,
        help="Maximum number of snapshots during MD simulation",
        type=int,
    )
    group.add_argument(
        "--retain-lammps",
        action="store_true",
        help="Keep LAMMPS output files after it finishes",
    )
    group.add_argument(
        "--dft-opt-steps",
        default=8,
        help="Maximum number of DFT optimization steps",
        type=int,
    )
    group.add_argument(
        "--raspa-timesteps",
        default=100000,
        help="Number of timesteps for GCMC computation",
        type=int,
    )

    group = parser.add_argument_group(
        title="Compute Settings", description="Compute environment configuration"
    )
    group.add_argument("--ray-address", required=True, help="Ray cluster address")
    group.add_argument(
        "--lammps-on-ramdisk",
        action="store_true",
        help="Write LAMMPS outputs to a RAM Disk",
    )
    group.add_argument(
        "--compute-config",
        choices=list(COMPUTE_CONFIGS),
        required=True,
        help="Configuration for the HPC system",
    )

    return parser.parse_args()


def configure_logging(run_dir: pathlib.Path, level: str) -> logging.Logger:
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)

    file_handler = logging.FileHandler(run_dir / "run.log")
    file_handler.setLevel(logging.DEBUG)

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
        handlers=[stream_handler, file_handler],
    )

    return logging.getLogger("main")


def main() -> int:
    args = parse_args()

    # Make the run directory
    params = args.__dict__.copy()
    start_time = datetime.utcnow()
    params_hash = hashlib.sha256(json.dumps(params).encode()).hexdigest()[:6]
    run_dir = pathlib.Path("run") / (
        f"agentic-{args.compute_config}-{start_time.strftime('%d%b%y%H%M%S')}-{params_hash}"
    )
    run_dir.mkdir(parents=True)

    # Save the run parameters to disk
    (run_dir / "params.json").write_text(json.dumps(params))

    logger = configure_logging(run_dir, args.log_level)
    logger.info("Loaded run params")
    logger.info("Created run directory: %s", run_dir)

    # Load the ligand descriptions
    templates = []
    for path in args.ligand_templates:
        template = LigandTemplate.from_yaml(path)
        templates.append(template)

    # Load the example MOF
    node_template = NodeDescription(
        **json.loads(pathlib.Path(args.node_path).read_text())
    )

    # Make all agent configs
    compute = COMPUTE_CONFIGS[args.compute_config]
    logger.info("Using compute config: %r", compute)

    generator_config = GeneratorConfig(
        generator_path=args.generator_path,
        templates=templates,
        num_workers=compute.num_generator_workers,
        atom_counts=args.molecule_sizes,
        batch_size=args.gen_batch_size,
        device=compute.torch_device,
        num_samples=args.num_samples,
    )
    assembler_config = AssemblerConfig(
        max_queue_depth=50 * compute.num_generator_workers,
        max_attempts=4,
        min_ligand_candidates=args.minimum_ligand_pool,
        num_mofs=min(compute.num_validator_workers + 4, 128),
        num_workers=compute.num_assembly_workers,
        node_templates=[node_template],
    )
    validator_config = ValidatorConfig(
        delete_finished=not args.retain_lammps,
        lammps_command=compute.lammps_cmd,
        lammps_environ=compute.lammps_env,
        lmp_sims_root_path=(
            "/dev/shm/lmp_run" if args.lammps_on_ramdisk else str(run_dir / "lmp_run")
        ),
        max_queue_depth=8 * compute.num_validator_workers,
        num_workers=compute.num_validator_workers,
        report_frequency=max(1, args.md_timesteps / args.md_snapshots),
        simulation_budget=args.simulation_budget,
        timesteps=args.md_timesteps,
    )
    logger.info("Initialized agent configs")

    ray.init(args.ray_address, configure_logging=False)

    with Manager(
        exchange=ThreadExchange(),
        launcher=ThreadLauncher(),
    ) as manager:
        # Register agents
        generator_id = manager.exchange.create_agent()
        assembler_id = manager.exchange.create_agent()
        validator_id = manager.exchange.create_agent()

        # Construct unbound handles to share with agent behaviors
        assembler_handle = manager.exchange.create_handle(assembler_id)
        validator_handle = manager.exchange.create_handle(validator_id)

        # Intialize agent behaviors
        generator_behavior = Generator(
            assembler=assembler_handle,
            config=generator_config,
            ray_address=args.ray_address,
        )
        assembler_behavior = Assembler(
            validator=validator_handle,
            config=assembler_config,
            ray_address=args.ray_address,
        )
        validator_behavior = Validator(
            assembler=assembler_handle,
            config=validator_config,
            ray_address=args.ray_address,
        )
        logger.info("Initialized agent behaviors")

        # Launch agents using preregistered IDs
        manager.launch(generator_behavior, agent_id=generator_id)
        manager.launch(assembler_behavior, agent_id=assembler_id)
        manager.launch(validator_behavior, agent_id=validator_id)

        try:
            manager.wait(validator_id)
        except KeyboardInterrupt:
            # Exiting the context manager will cause the agents to be shutdown.
            logger.info("Requesting validator to shutdown...")
            manager.shutdown(validator_id, blocking=True)

        logger.info("Shutting down remaining agents...")
    logger.info("All agents completed!")
    ray.shutdown()
    logger.info("Ray shutdown")


if __name__ == "__main__":
    raise SystemExit(main())
