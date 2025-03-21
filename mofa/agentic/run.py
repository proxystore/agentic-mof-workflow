from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import pathlib
import sys
import typing
from datetime import datetime

import parsl
from aeris.exchange.redis import RedisExchange
from aeris.launcher.executor import ExecutorLauncher
from aeris.launcher.thread import ThreadLauncher
from aeris.manager import Manager
from globus_compute_sdk import Executor
from openbabel import openbabel
from rdkit import RDLogger

from mofa.agentic.compute import COMPUTE_CONFIGS
from mofa.agentic.config import AssemblerConfig
from mofa.agentic.config import DatabaseConfig
from mofa.agentic.config import EstimatorConfig
from mofa.agentic.config import GeneratorConfig
from mofa.agentic.config import OptimizerConfig
from mofa.agentic.config import TrainerConfig
from mofa.agentic.config import ValidatorConfig
from mofa.agentic.parsl import get_parsl_config
from mofa.agentic.steering import Assembler
from mofa.agentic.steering import Database
from mofa.agentic.steering import Estimator
from mofa.agentic.steering import Generator
from mofa.agentic.steering import Optimizer
from mofa.agentic.steering import Validator
from mofa.model import LigandTemplate
from mofa.model import NodeDescription

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
    group.add_argument(
        "--node-path",
        required=True,
        help="Path to a node record",
    )

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
        "Retraining Settings",
        description="How often to retain, what to train on, etc",
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
        "--num-epochs",
        type=int,
        default=128,
        help="Number of training epochs",
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
        title="Assembly Settings",
        description="Options related to MOF assembly",
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
        title="Compute Settings",
        description="Compute environment configuration",
    )
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
    group.add_argument(
        "--cpu-endpoint",
        required=True,
        help="CPU agent endpoint",
    )
    group.add_argument(
        "--polaris-endpoint",
        required=True,
        help="Polaris agent endpoint",
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


def create_managers(
    cpu_endpoint: str,
    polaris_endpoint: str,
    logger: logging.Logger,
) -> typing.Generator[dict[str, Manager], None, None]:
    with contextlib.ExitStack() as stack:
        exchange = RedisExchange(
            host=os.environ["REDIS_HOST"],
            port=os.environ["REDIS_PORT"],
            password=os.environ["REDIS_PASS"],
        )

        cpu_launcher = ExecutorLauncher(Executor(cpu_endpoint))
        polaris_launcher = ExecutorLauncher(Executor(polaris_endpoint))
        thread_launcher = ThreadLauncher()

        cpu_manager = Manager(exchange=exchange, launcher=cpu_launcher)
        polaris_manager = Manager(exchange=exchange, launcher=polaris_launcher)
        thread_manager = Manager(exchange=exchange, launcher=thread_launcher)

        managers = {
            "cpu": cpu_manager,
            "polaris": polaris_manager,
            "thread": thread_manager,
        }
        for manager in managers.values():
            stack.enter_context(manager)

        logger.info("Initialized managers!")
        yield manager
        logger.info("Shutting down managers...")
    logger.info("All managers shutdown!")


def run(  # noqa: PLR0913
    *,
    managers: dict[str, Manager],
    database_config: DatabaseConfig,
    generator_config: GeneratorConfig,
    trainer_config: TrainerConfig,
    assembler_config: AssemblerConfig,
    validator_config: ValidatorConfig,
    optimizer_config: OptimizerConfig,
    estimator_config: EstimatorConfig,
    logger: logging.Logger,
) -> None:
    # Register agents; all managers use the same exchange
    database_id = managers["thread"].exchange.create_agent()
    generator_id = managers["thread"].exchange.create_agent()
    assembler_id = managers["thread"].exchange.create_agent()
    validator_id = managers["thread"].exchange.create_agent()
    optimizer_id = managers["thread"].exchange.create_agent()
    estimator_id = managers["thread"].exchange.create_agent()

    # Construct unbound handles to share with agent behaviors
    database_handle = managers["thread"].exchange.create_handle(database_id)
    assembler_handle = managers["thread"].exchange.create_handle(assembler_id)
    validator_handle = managers["thread"].exchange.create_handle(validator_id)
    generator_handle = managers["thread"].exchange.create_handle(generator_id)
    optimizer_handle = managers["thread"].exchange.create_handle(optimizer_id)
    estimator_handle = managers["thread"].exchange.create_handle(estimator_id)

    # Intialize agent behaviors
    generator_behavior = Generator(
        assembler=assembler_handle,
        config=generator_config,
        trainer=trainer_config,
    )
    assembler_behavior = Assembler(
        validator=validator_handle,
        config=assembler_config,
    )
    validator_behavior = Validator(
        assembler=assembler_handle,
        database=database_handle,
        optimizer=optimizer_handle,
        config=validator_config,
    )
    database_behavior = Database(
        generator_handle,
        config=database_config,
        trainer=trainer_config,
    )
    optimizer_behavior = Optimizer(
        estimator=estimator_handle,
        config=optimizer_config,
    )
    estimator_behavior = Estimator(
        database=database_handle,
        config=estimator_config,
    )
    logger.info("Initialized agent behaviors")

    # Launch agents using preregistered IDs
    managers["cpu"].launch(database_behavior, agent_id=database_id)
    managers["thread"].launch(generator_behavior, agent_id=generator_id)
    managers["cpu"].launch(assembler_behavior, agent_id=assembler_id)
    managers["thread"].launch(validator_behavior, agent_id=validator_id)
    managers["polaris"].launch(optimizer_behavior, agent_id=optimizer_id)
    managers["cpu"].launch(estimator_behavior, agent_id=estimator_id)

    try:
        managers["thread"].wait(validator_id)
    except KeyboardInterrupt:
        # Exiting the context manager will cause the agents to be shutdown.
        logger.info("Requesting validator to shutdown...")
        managers["thread"].shutdown(validator_id, blocking=True)


def main() -> int:
    args = parse_args()

    # Make the run directory
    params = args.__dict__.copy()
    start_time = datetime.utcnow().strftime("%d%b%y%H%M%S")
    params_hash = hashlib.sha256(json.dumps(params).encode()).hexdigest()[:6]
    run_dir = pathlib.Path("run") / (
        f"agentic-{args.compute_config}-{start_time}-{params_hash}"
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
        **json.loads(pathlib.Path(args.node_path).read_text()),
    )

    # Make all agent configs
    compute = COMPUTE_CONFIGS[args.compute_config]
    logger.info("Using compute config: %r", compute)

    ccloud_run_dir = pathlib.Path(f"/home/cc/mofa-runs/{start_time}")
    polaris_run_dir = pathlib.Path(
        f"/eagle/MOFA/jgpaul/scratch/mofa-runs/{start_time}",
    )

    database_config = DatabaseConfig(
        run_dir=str(ccloud_run_dir / "database"),
    )
    generator_config = GeneratorConfig(
        generator_path=args.generator_path,
        model_dir=run_dir / "models",
        templates=templates,
        num_workers=compute.num_generator_workers,
        atom_counts=args.molecule_sizes,
        batch_size=args.gen_batch_size,
        device=compute.torch_device,
        num_samples=args.num_samples,
    )
    trainer_config = TrainerConfig(
        maximum_train_size=args.maximum_train_size,
        num_epochs=args.num_epochs,
        minimum_train_size=args.retrain_freq,
        best_fraction=args.best_fraction,
        maximum_strain=args.maximum_strain,
        config_path=args.generator_config_path,
        retrain_dir=run_dir / "retraining",
        device=compute.torch_device,
    )
    assembler_config = AssemblerConfig(
        max_queue_depth=50 * compute.num_generator_workers,
        max_attempts=4,
        min_ligand_candidates=args.minimum_ligand_pool,
        num_mofs=min(compute.num_validator_workers + 4, 128),
        num_workers=compute.num_assembly_workers,
        node_templates=[node_template],
        run_dir=str(ccloud_run_dir / "assembler"),
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
    optimizer_config = OptimizerConfig(
        cp2k_cmd=compute.cp2k_cmd,
        cp2k_dir=run_dir / "cp2k-runs",
        cp2k_steps=args.dft_opt_steps,
        num_workers=compute.num_optimizer_workers,
        run_dir=str(polaris_run_dir / "optimizer"),
    )
    estimator_config = EstimatorConfig(
        num_workers=compute.num_estimator_workers,
        raspa_dir=run_dir / "raspa-runs",
        raspa_timesteps=args.raspa_timesteps,
        run_dir=str(ccloud_run_dir / "estimator"),
    )
    logger.info("Initialized agent configs")

    config = get_parsl_config("aurora", run_dir=str(run_dir / "parsl"))
    parsl.load(config)
    try:
        with create_managers(
            cpu_endpoint=args.cpu_endpoint,
            polaris_endpoint=args.polaris_endpoint,
            logger=logger,
        ) as managers:
            run(
                managers=managers,
                database_config=database_config,
                generator_config=generator_config,
                trainer_config=trainer_config,
                assembler_config=assembler_config,
                validator_config=validator_config,
                optimizer_config=optimizer_config,
                estimator_config=estimator_config,
                logger=logger,
            )
    except Exception:
        logger.exception("Workflow run failed!")
        raise
    finally:
        logger.info("Cleaning up Parsl DFK...")
        parsl.dfk().cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
