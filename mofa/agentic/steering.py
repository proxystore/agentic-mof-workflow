from __future__ import annotations

import contextlib
import dataclasses
import itertools
import logging
import pathlib
import os
import random
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from collections import defaultdict
from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future
from datetime import datetime
from queue import Empty
from queue import PriorityQueue
from queue import Queue
from threading import Event
from threading import Semaphore
from typing import Any

import ase
import parsl
import pymongo
from aeris.behavior import action
from aeris.behavior import Behavior
from aeris.behavior import loop
from aeris.logging import init_logging
from aeris.handle import Handle

from mofa import db as mofadb
from mofa.agentic.config import AssemblerConfig
from mofa.agentic.config import DatabaseConfig
from mofa.agentic.config import EstimatorConfig
from mofa.agentic.config import GeneratorConfig
from mofa.agentic.config import OptimizerConfig
from mofa.agentic.config import TrainerConfig
from mofa.agentic.config import ValidatorConfig
from mofa.agentic.parsl_config import get_parsl_config
from mofa.agentic.task import assemble_mofs_task
from mofa.agentic.task import estimate_adsorption_task
from mofa.agentic.task import generate_ligands_task
from mofa.agentic.task import optimize_cells_and_compute_charges_task
from mofa.agentic.task import retrain_task
from mofa.agentic.task import validate_structure_task
from mofa.model import LigandDescription
from mofa.model import MOFRecord
from mofa.scoring.geometry import LatticeParameterChange
from mofa.simulation.cp2k import CP2KRunner
from mofa.simulation.lammps import LAMMPSRunner
from mofa.simulation.raspa import RASPARunner
from mofa.utils.conversions import write_to_string

ACTION_TIMEOUT = 120


@dataclasses.dataclass(order=True)
class _Item:
    priority: float
    value: Any = dataclasses.field(compare=False)


class MOFABehavior(Behavior):
    def __init__(self, *, logger_name: str) -> None:
        self.logger = logging.getLogger(logger_name)

    @action
    def identity(self, x):
        return x

    def on_setup(self) -> None:
        pass

    def on_shutdown(self) -> None:
        pass


class Database(MOFABehavior):
    def __init__(
        self,
        generator: Handle[Generator],
        config: DatabaseConfig,
        trainer: TrainerConfig,
        **kwargs: Any,
    ) -> None:
        self.generator = generator
        self.config = config
        self.trainer = trainer

        self.lammps_completed = 0
        self.raspa_completed = 0

        super().__init__(logger_name="Database", **kwargs)

    def on_setup(self) -> None:
        super().on_setup()

        run_dir = pathlib.Path(self.config.run_dir)
        mongo_dir = run_dir / "db"
        mongo_dir.mkdir(parents=True)

        init_logging(
            logfile=os.path.join(self.config.run_dir, "log.txt"),
            color=False,
        )

        self.retrain = Event()

        self.mongo_proc = subprocess.Popen(
            f"mongod --wiredTigerCacheSizeGB 4 --dbpath {mongo_dir.absolute()} "
            f"--logpath {(run_dir / 'mongo.log').absolute()}".split(),
            stderr=(run_dir / "mongo.err").open("w"),
        )
        time.sleep(5)
        self.logger.info("Spawned MongoDB process (pid=%d)", self.mongo_proc.pid)

        self.client = pymongo.MongoClient(
            host=self.config.mongo_host,
            port=self.config.mongo_port,
        )
        self.collection = mofadb.initialize_database(self.client)

    def on_shutdown(self) -> None:
        self.mongo_proc.terminate()
        try:
            self.mongo_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.logger.exception("Timeout waiting for MongoDB shutdown. Killing...")
            self.mongo_proc.kill()
        else:
            self.logger.info("Shutdown MongoDB")

        self.client.close()

        super().on_shutdown()

    @loop
    def periodic_retrain(self, shutdown: Event) -> None:
        # Waits for sufficient records to be created and then invokes the
        # retrain action on the Generator agent with the top-performing
        # records (either by stability or capacity). This blocks on the
        # retraining so that multiple retrains are not performed at same time.
        last_train_size = 0

        while not shutdown.is_set():
            if not self.retrain.wait(timeout=1):
                continue

            # Determine how to select best MOFs
            if self.raspa_completed < self.trainer.minimum_train_size:
                sort_field = "structure_stability.uff"
                to_include = min(
                    int(self.lammps_completed * self.trainer.best_fraction),
                    self.trainer.maximum_train_size,
                )
                sort_order = pymongo.ASCENDING
            else:
                sort_field = "gas_storage.CO2"
                to_include = min(
                    int(self.raspa_completed * self.trainer.best_fraction),
                    self.trainer.maximum_train_size,
                )
                sort_order = pymongo.DESCENDING

            # Build the query
            query = defaultdict(dict)
            query[sort_field] = {"$exists": True}
            query["structure_stability.uff"] = {"$lt": self.trainer.maximum_strain}

            # Filter out the trajectory to save I/O
            cursor = (
                self.collection.find(
                    filter=query,
                    projection={"md_trajectory": 0},
                )
                .sort(sort_field, sort_order)
                .limit(to_include)
            )

            examples = []
            for record in cursor:
                record.pop("_id")
                record["times"] = {}
                record["md_trajectory"] = {}
                examples.append(MOFRecord(**record))

            if len(examples) == 0:
                self.logger.warning("No valid training examples in database")
                self.retrain.clear()
                continue
            if (
                len(examples) == last_train_size
                and len(examples) < self.trainer.maximum_train_size
            ):
                self.logger.info(
                    "Number of training samples has not changed since last "
                    "retraining. Skipping...",
                )
                self.retrain.clear()
                continue

            last_train_size = len(examples)

            future = self.generator.action("retrain", examples)
            self.retrain.clear()
            try:
                future.result()
            except Exception:
                self.logger.exception("Error in retrain action")

    @action
    def create_record(self, record: MOFRecord) -> None:
        # Create a MOF record. This is invoked by the validator to store
        # records for all validated MOFs.
        mofadb.create_records(self.collection, [record])
        self.logger.info("Created database record for %r", record.name)

        # This is true because the validate-structures task callback
        # is the only place where this action gets triggered, but if that
        # changes we'd need a better way to track.
        self.lammps_completed += 1
        if self.lammps_completed >= self.trainer.minimum_train_size:
            self.retrain.set()

    @action
    def update_record(self, record: MOFRecord) -> None:
        # Update a MOF record with the gas capacity computed by the Estimator.
        mofadb.update_records(self.collection, [record])
        self.logger.info("Updated database record for %r", record.name)

        # This is true because the estimate-adsorption task callback
        # is the only place where this action gets triggered, but if that
        # changes we'd need a better way to track.
        self.raspa_completed += 1
        if self.raspa_completed >= self.trainer.minimum_train_size:
            self.retrain.set()


class Generator(MOFABehavior):
    def __init__(
        self,
        assembler: Handle[Assembler],
        config: GeneratorConfig,
        trainer: TrainerConfig,
        **kwargs: Any,
    ) -> None:
        self.assembler = assembler
        self.config = config
        self.trainer = trainer

        self.initial_model_path = config.generator_path
        self.latest_model_path = config.generator_path
        self.model_iteration = 0

        self.generator_tasks: set[Future] = set()
        self.retrain_tasks: dict[int, Future] = {}

        super().__init__(logger_name="Generator", **kwargs)

    def on_setup(self) -> None:
        super().on_setup()

        self.generator_queue = Queue()
        self.generator_count = Semaphore(value=self.config.num_workers)
        self.retrain_lock = threading.Lock()

        tasks = list(
            itertools.product(
                range(len(self.config.templates)),
                self.config.atom_counts,
            ),
        )
        random.shuffle(tasks)
        for task in tasks:
            self.generator_queue.put(task)

    def on_shutdown(self) -> None:
        self.logger.warning(
            "There are %s remaining generate-ligands task(s) after shutdown",
            len(self.generator_tasks),
        )
        self.logger.warning(
            "There are %s remaining retrain task(s) after shutdown",
            len(self.retrain_tasks),
        )

        super().on_shutdown()

    @loop
    def submit_generate_ligands(self, shutdown: Event) -> None:
        # Submit generate_ligands_task while there are available resources
        # as indicated by the self.generator_count semaphore.
        while not shutdown.is_set():
            # Acquired workers are released when the generator tasks
            # raise StopIteration exceptions in submit_process_ligands.
            if not self.generator_count.acquire(timeout=1):
                continue

            try:
                ligand_id, size = self.generator_queue.get(timeout=1)
            except Empty:
                self.generator_count.release()
                continue

            ligand = self.config.templates[ligand_id]
            future = generate_ligands_task(
                model=self.latest_model_path,
                templates=[ligand],
                batch_size=self.config.batch_size,
                n_atoms=size,
                n_samples=self.config.num_samples,
                device=self.config.device,
            )
            future._id = uuid.uuid4()
            self.logger.info(
                "Submitted generate-ligands task (type=%s size=%d, "
                "samples=%d, model=%d)",
                ligand.anchor_type,
                size,
                self.config.num_samples,
                self.model_iteration,
            )
            self.logger.info("START generate-ligands %s", future._id)
            self.generator_tasks.add(future)
            future.add_done_callback(self._generate_ligands_callback)
            # Push this generation task back on the queue
            self.generator_queue.put((ligand_id, size))

    def _generate_ligands_callback(self, future) -> None:
        self.generator_count.release()
        self.generator_tasks.remove(future)
        self.logger.info("END generate-ligands %s", future._id)
        try:
            valid_ligands, anchor_type = future.result()
        except Exception:
            self.logger.exception("Failure in generate-ligands task")
            return

        self.logger.info(
            "Received generate-ligands batch (valid=%s)",
            len(valid_ligands),
        )

        if len(valid_ligands) == 0:
            return

        action_future = self.assembler.action(
            "submit_ligands",
            anchor_type,
            valid_ligands,
        )
        try:
            action_future.result(timeout=ACTION_TIMEOUT)
        except TimeoutError:
            self.logger.warning(
                "Timeout in submit-ligands action. Is the assembler actor alive?",
            )
        except Exception:
            self.logger.exception("Error in submit-ligands action")
        else:
            self.logger.info("Submitted ligands to assembler")

    def _retrain(self, examples: list[MOFRecord]) -> None:
        version = self.model_iteration + 1
        retrain_dir = self.trainer.retrain_dir / f"model-v{version}"
        retrain_dir.mkdir(parents=True)

        # future = retrain_task(
        #     starting_model=self.initial_model_path,
        #     run_directory=retrain_dir,
        #     config_path=self.trainer.config_path,
        #     examples=examples,
        #     num_epochs=self.trainer.num_epochs,
        #     device=self.trainer.device,
        # )
        # self.retrain_tasks[version] = future
        # self.logger.info(
        #     "Submitted retrain task (version=%d, path=%s)",
        #     version,
        #     retrain_dir,
        # )

        uid = uuid.uuid4()
        self.logger.info("START retrain %s", uid)
        import time

        time.sleep(100)
        self.logger.info("END retrain %s", uid)
        # try:
        #     path = future.result()
        # except Exception:
        #     self.logger.exception("Error in retrain task")
        # else:
        #     new_model_path = self.config.model_dir / f"model-v{version}.ckpt"
        #     self.config.model_dir.mkdir(exist_ok=True, parents=True)
        #     shutil.copyfile(path, new_model_path)
        #     self.model_iteration = version
        #     self.latest_model_path = new_model_path
        #     self.logger.info(
        #         "Received retrain task (version=%d, path=%s)",
        #         version,
        #         new_model_path,
        #     )
        # finally:
        #     shutil.rmtree(retrain_dir)

    @action
    def retrain(self, examples: list[MOFRecord]) -> None:
        with self.retrain_lock:
            return self._retrain(examples)


class Assembler(MOFABehavior):
    def __init__(
        self,
        *,
        validator: Handle[Validator],
        config: AssemblerConfig,
        **kwargs: Any,
    ) -> None:
        self.config = config
        self.validator = validator
        super().__init__(logger_name="Assembler", **kwargs)

    def on_setup(self) -> None:
        super().on_setup()

        init_logging(
            logfile=os.path.join(self.config.run_dir, "log.txt"),
            color=False,
        )

        config = get_parsl_config(
            "local",
            self.config.run_dir,
            workers_per_node=self.config.num_workers,
        )
        parsl.load(config)

        self.assembly_queues = defaultdict(
            lambda: deque(maxlen=self.config.max_queue_depth),
        )
        self.assembly_count = threading.Semaphore(
            value=self.config.num_workers,
        )
        self.assembly_tasks: set[Future] = set()
        self.enabled = threading.Event()
        self.enabled.set()

    def on_shutdown(self) -> None:
        self.logger.warning(
            "There are %s remaining assemble-ligands task(s) after shutdown",
            len(self.assembly_tasks),
        )

        parsl.dfk().cleanup()
        super().on_shutdown()

    @action
    def submit_ligands(
        self,
        anchor_type: str,
        valid_ligands: Sequence[LigandDescription],
    ) -> None:
        # Submit a batch of ligands to the assembler's queue for MOF assembly.
        self.assembly_queues[anchor_type].extend(valid_ligands)
        self.logger.info(
            "Added ligands to assembly queue (anchor_type=%s, count=%d)",
            anchor_type,
            len(valid_ligands),
        )

    @loop
    def submit_assembly(self, shutdown: Event) -> None:
        # Pull from assembly queue and submit assembly tasks. Works with
        # the process_assembly loop to determine max number of assembly
        # tasks to run at any point.
        while not shutdown.is_set():
            if not self.enabled.wait(timeout=1):
                continue

            # Acquired workers are released in the callback on the task future.
            if not self.assembly_count.acquire(timeout=1):
                continue

            # TODO: handle dictionary resizing here in a better fashion than
            # copying the dict
            candidates = sum(len(q) for q in self.assembly_queues.copy().values())
            if (
                candidates < self.config.min_ligand_candidates
                # assemble_many() requires 2 COO and 1 cyano ligands
                or "COO" not in self.assembly_queues
                or "cyano" not in self.assembly_queues
            ):
                self.assembly_count.release()
                continue

            ligand_options = {k: list(v) for k, v in self.assembly_queues.items()}
            future = assemble_mofs_task(
                ligand_options=ligand_options,
                nodes=self.config.node_templates,
                to_make=self.config.num_mofs,
                attempts=self.config.max_attempts,
            )
            future._id = uuid.uuid4()
            self.logger.info("START assemble-mofs %s", future._id)
            self.logger.info("Submitted assemble-mofs task")
            self.assembly_tasks.add(future)
            future.add_done_callback(self._assembly_task_callback)

    def _assembly_task_callback(self, future: Future) -> None:
        self.logger.info("END assemble-mofs %s", future._id)
        self.assembly_count.release()
        self.assembly_tasks.remove(future)
        try:
            mofs = future.result()
        except Exception:
            self.logger.exception("Failure in assemble-mofs task")
            return

        self.logger.info("Received assemble-mofs batch (count=%d)", len(mofs))
        action_future = self.validator.action("submit_mofs", mofs)
        try:
            action_future.result(timeout=ACTION_TIMEOUT)
        except TimeoutError:
            self.logger.warning(
                "Timeout in submit-mofs action. Is the validator actor alive?",
            )
        except Exception:
            self.logger.exception("Error in submit-mofs action")
        else:
            self.logger.info("Submitted mofs to validator")

    @action
    def enable_assembly(self) -> None:
        self.logger.info("Enabling MOF assembly")
        self.enabled.set()

    @action
    def disable_assembly(self) -> None:
        self.logger.info("Disabling MOF assembly")
        self.enabled.clear()


class Validator(MOFABehavior):
    def __init__(
        self,
        assembler: Handle[Assembler],
        database: Handle[Database],
        optimizer: Handle[Optimizer],
        config: ValidatorConfig,
        **kwargs: Any,
    ) -> None:
        self.assembler = assembler
        self.database = database
        self.optimizer = optimizer
        self.config = config
        self.runner = LAMMPSRunner(
            lammps_command=config.lammps_command,
            lmp_sims_root_path=config.lmp_sims_root_path,
            # lammps_environ={'OMP_NUM_THREADS': '8', **config.lammps_environ},
            lammps_environ=config.lammps_environ.copy(),
            delete_finished=config.delete_finished,
            timeout=None,
        )
        self.simulations_budget = config.simulation_budget
        self.queue_threshold = max(
            self.config.num_workers,
            int(0.8 * self.config.max_queue_depth),
        )

        super().__init__(logger_name="Validator", **kwargs)

    def on_setup(self) -> None:
        super().on_setup()

        self.assembler_enabled = True
        self.validator_queue: deque[MOFRecord] = deque(
            maxlen=self.config.max_queue_depth,
        )
        self.validator_count = Semaphore(value=self.config.num_workers)
        self.validator_tasks: set[Future] = set()

        self.simulations_completed = 0

        self.process_queue: Queue[tuple[MOFRecord, list[ase.Atoms]]] = Queue()
        self.processed_mofs = 0

    def on_shutdown(self) -> None:
        self.logger.info(
            "There are %d remaining validate-structures task(s) after shutdown",
            len(self.validator_tasks),
        )

        super().on_shutdown()

    def _check_queue_depth(self, timeout: float) -> None:
        # TODO: lock this to prevent race conditions?
        if len(self.validator_queue) > self.queue_threshold and self.assembler_enabled:
            self.assembler.action("disable_assembly").result(timeout=timeout)
            self.assembler_enabled = False
            self.logger.info("Disabled assembly because queue is full")
        elif (
            len(self.validator_queue) < self.queue_threshold
            and not self.assembler_enabled
        ):
            self.assembler.action("enable_assembly").result(timeout=timeout)
            self.assembler_enabled = True
            self.logger.info("Enabled assembly because queue is low")

    @action
    def submit_mofs(self, mofs: Sequence[MOFRecord]) -> None:
        # Submit a MOF to the Validator's queue for validation (lammps).
        for mof in mofs:
            self.validator_queue.append(mof)
        self.logger.info(
            "Added mofs to validation queue (count=%d)",
            len(mofs),
        )
        # self._check_queue_depth(timeout=ACTION_TIMEOUT)

    @loop
    def monitor_budget(self, shutdown: Event) -> None:
        while not shutdown.is_set():
            if self.simulations_completed < self.simulations_budget:
                time.sleep(1)
                continue

            self.logger.info(
                "Simulation budget reached! Shutting down validator...",
            )
            with contextlib.suppress(TimeoutError):
                self.assembler.action("disable_assembly").result(timeout=ACTION_TIMEOUT)
            shutdown.set()

    @loop
    def submit_validation(self, shutdown: Event) -> None:
        # Pull from the validation queue and submit validation tasks. Works
        # with the process_validation loop to determine the max number of
        # assembly tasks to run at any point.
        while not shutdown.is_set():
            self._check_queue_depth(timeout=ACTION_TIMEOUT)

            # Acquired workers are released in the callback on the task future.
            if not self.validator_count.acquire(timeout=1):
                continue

            try:
                record = self.validator_queue.pop()
            except IndexError:
                self.validator_count.release()
                time.sleep(1)
                continue

            future = validate_structure_task(
                runner=self.runner,
                mof=record,
                timesteps=self.config.timesteps,
                report_frequency=self.config.report_frequency,
            )
            future._id = uuid.uuid4()
            self.logger.info("START validate-structures %s", future._id)
            self.logger.info(
                "Submitted validate-structures task (name=%s)",
                record.name,
            )
            self.validator_tasks.add(future)
            future.add_done_callback(self._validate_task_callback)

    @loop
    def process_validation(self, shutdown: Event) -> None:
        while not shutdown.is_set():
            try:
                record, frames = self.process_queue.get(timeout=1)
            except Empty:
                continue

            self.logger.info(
                "Computing lattice strain (name=%s, frames=%d, queue_size=%d)",
                record.name,
                len(frames),
                self.process_queue.qsize(),
            )
            # Compute the lattice strain
            scorer = LatticeParameterChange()
            frames_vasp = [write_to_string(f, "vasp") for f in frames]
            record.md_trajectory["uff"] = frames_vasp
            strain = scorer.score_mof(record)
            record.structure_stability["uff"] = strain
            record.times["md-done"] = datetime.now()
            self.logger.info(
                "Computed lattice strain (name=%s, strain=%.1f)",
                record.name,
                100 * strain,
            )
            self.optimizer.action("submit_mof", record).result(timeout=ACTION_TIMEOUT)
            self.logger.info("Submitted structure to optimizer (name=%s)", record.name)
            self.database.action("create_record", record).result(timeout=ACTION_TIMEOUT)
            self.logger.info("Submitted record to database (name=%s)", record.name)

    def _validate_task_callback(self, future: Future) -> None:
        self.logger.info("END validate-structures %s", future._id)
        self.validator_count.release()
        self.validator_tasks.remove(future)
        try:
            record, frames = future.result()
        except Exception as e:
            self.logger.warning("Failure in validate-structures task: %s", e)
            self.logger.debug(
                "Failure traceback: %r\n%s",
                e,
                traceback.format_exc(),
            )
            return

        self.simulations_completed += 1
        self.logger.info(
            "Received validate-structures result (name=%s, budget-left=%d)",
            record.name,
            self.simulations_budget - self.simulations_completed,
        )
        self.process_queue.put((record, frames))


class Optimizer(MOFABehavior):
    def __init__(
        self,
        estimator: Handle[Estimator],
        config: OptimizerConfig,
        **kwargs: Any,
    ) -> None:
        self.estimator = estimator
        self.config = config
        super().__init__(logger_name="Optimizer", **kwargs)

    def on_setup(self) -> None:
        super().on_setup()

        init_logging(
            logfile=os.path.join(self.config.run_dir, "log.txt"),
            color=False,
        )

        config = get_parsl_config(
            "polaris",
            self.config.run_dir,
        )
        parsl.load(config)

        self.cp2k_runner = CP2KRunner(
            cp2k_invocation=self.config.cp2k_cmd,
            run_dir=pathlib.Path(self.config.cp2k_dir),
        )
        self.records: dict[str, MOFRecord] = {}
        self.optimizer_count = threading.Semaphore(self.config.num_workers)
        self.optimize_queue = PriorityQueue()
        self.optimize_tasks: set[Future] = set()

    def on_shutdown(self) -> None:
        self.logger.warning(
            "There are %s remaining optimize-cells task(s) after shutdown",
            len(self.optimize_tasks),
        )
        parsl.dfk().cleanup()
        super().on_shutdown()

    @action
    def submit_mof(self, record: MOFRecord) -> None:
        # Submit a MOF to the Optimizer's queue for optimization.
        priority = record.structure_stability["uff"]
        self.optimize_queue.put(_Item(priority, record))
        self.logger.info("Added mof to optimizer queue (name=%s)", record.name)

    @loop
    def submit_optimization(self, shutdown: Event) -> None:
        # Pull from the optimization queue and submit optimization tasks
        # (cp2k). Works with the process_optimization loops to determine
        # the maximum number of tasks to submit.
        while not shutdown.is_set():
            if not self.optimizer_count.acquire(timeout=1):
                continue

            try:
                item = self.optimize_queue.get(timeout=1)
                record = item.value
            except Empty:
                self.optimizer_count.release()
                continue

            if record.name in self.records:
                self.optimizer_count.release()
                continue

            self.records[record.name] = record

            future = optimize_cells_and_compute_charges_task(
                runner=self.cp2k_runner,
                mof=record,
                steps=self.config.cp2k_steps,
                fmax=0.1,
            )
            future._id = uuid.uuid4()
            self.logger.info("START optimize-cells %s", future._id)
            self.logger.info(
                "Submitted optimize-cells task (name=%s)",
                record.name,
            )
            self.optimize_tasks.add(future)
            future.add_done_callback(self._optimize_cells_task_callback)

    def _optimize_cells_task_callback(self, future) -> None:
        self.logger.info("END optimize-cells %s", future._id)
        self.optimizer_count.release()
        self.optimize_tasks.remove(future)
        try:
            record, atoms = future.result()
        except Exception:
            self.logger.exception("Failure in optimize-cells task")
            return

        self.logger.info("Completed optimize-cells task (name=%s)", record.name)

        action_future = self.estimator.submit("submit_atoms", record, atoms)
        try:
            action_future.result(timeout=ACTION_TIMEOUT)
        except TimeoutError:
            self.logger.warning(
                "Timeout in submit-atoms action. Is the estimator alive?",
            )
        except Exception:
            self.logger.exception("Error in submit-atoms action.")
        else:
            self.logger.info("Submitted mofs to validator.")


class Estimator(MOFABehavior):
    def __init__(
        self,
        database: Handle[Database],
        config: EstimatorConfig,
        **kwargs: Any,
    ) -> None:
        self.database = database
        self.config = config
        super().__init__(logger_name="Estimator", **kwargs)

    def on_setup(self) -> None:
        super().on_setup()

        init_logging(
            logfile=os.path.join(self.config.run_dir, "log.txt"),
            color=False,
        )

        config = get_parsl_config(
            "local",
            self.config.run_dir,
            workers_per_node=self.config.num_workers,
        )
        parsl.load(config)

        self.raspa_runner = RASPARunner(
            raspa_sims_root_path=pathlib.Path(self.config.raspa_dir),
        )
        self.estimate_queue: Queue[tuple[str, ase.Atoms]] = Queue()
        self.estimate_tasks: set[Future] = {}
        self.records: dict[str, MOFRecord] = {}

    def on_shutdown(self) -> None:
        self.logger.warning(
            "There are %s remaining estimate-adsorption task(s) after shutdown",
            len(self.estimate_tasks),
        )
        parsl.dfk().cleanup()
        super().on_shutdown()

    @action
    def submit_atoms(self, record: MOFRecord, atoms: ase.Atoms) -> None:
        self.estimate_queue.put((record.name, atoms))
        self.records[record.name] = record
        self.logger.info("Added atoms to estimator queue (name=%s)", record.name)

    @loop
    def submit_estimation(self, shutdown: Event) -> None:
        while not shutdown.is_set():
            try:
                name, atoms = self.estimate_queue.get(timeout=1)
            except Empty:
                continue

            future = estimate_adsorption_task(
                self.raspa_runner,
                atoms,
                name,
                timesteps=self.config.raspa_timesteps,
            )
            future._id = uuid.uuid4()
            self.logger.info("START estimate-adsorption %s", future._id)
            self.logger.info("Submitted estimate-adsorption task (name=%s)", name)
            self.estimate_tasks.add(future)
            future.add_done_callback(self._estimate_adsorption_task_callback)

    def _estimate_adsorption_task_callback(self, future) -> None:
        self.logger.info("END estimate-adsorption %s", future._id)
        self.estimate_tasks.remove(future)
        try:
            name, storage_mean, storage_std = future.result()
        except Exception:
            self.logger.exception("Failure in estimate-adsorption task")
            return

        self.logger.info("Completed estimate-adsorption task (name=%s)", name)

        record = self.records[name]
        record.gas_storage["C02"] = storage_mean
        record.times["raspa-done"] = datetime.now()

        self.database.action("update_record", record).result(timeout=ACTION_TIMEOUT)
