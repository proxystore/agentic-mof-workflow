from __future__ import annotations

import contextlib
import itertools
import logging
import random
import shutil
import threading
import time
import traceback
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
import pymongo
import ray
from aeris.behavior import action
from aeris.behavior import Behavior
from aeris.behavior import loop
from aeris.handle import Handle

from mofa import db as mofadb
from mofa.agentic.config import AssemblerConfig
from mofa.agentic.config import GeneratorConfig
from mofa.agentic.config import OptimizerConfig
from mofa.agentic.config import TrainerConfig
from mofa.agentic.config import ValidatorConfig
from mofa.agentic.task import assemble_mofs_task
from mofa.agentic.task import compute_partial_charges_task
from mofa.agentic.task import estimate_adsorption_task
from mofa.agentic.task import generate_ligands_task
from mofa.agentic.task import optimize_cells_task
from mofa.agentic.task import process_ligands_task
from mofa.agentic.task import retrain_task
from mofa.agentic.task import validate_structure_task
from mofa.model import LigandDescription
from mofa.model import MOFRecord
from mofa.scoring.geometry import LatticeParameterChange
from mofa.simulation.cp2k import CP2KRunner
from mofa.simulation.lammps import LAMMPSRunner
from mofa.simulation.raspa import RASPARunner
from mofa.utils.conversions import write_to_string

_ray_init_lock = threading.Lock()


class MOFABehavior(Behavior):
    def __init__(self, *, logger_name: str) -> None:
        self.logger = logging.getLogger(logger_name)

    def setup(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


class Database(MOFABehavior):
    def __init__(
        self,
        generator: Handle[Generator],
        trainer: TrainerConfig,
        *,
        mongo_host: str = "localhost",
        mongo_port: int = 27017,
        **kwargs: Any,
    ) -> None:
        self.generator = generator
        self.trainer = trainer
        self.mongo_host = mongo_host
        self.mongo_port = mongo_port

        self.lammps_completed = 0
        self.raspa_completed = 0

        super().__init__(logger_name="Database", **kwargs)

    def setup(self) -> None:
        super().setup()

        self.retrain = Event()

        self.client = pymongo.MongoClient(
            host=self.mongo_host,
            port=self.mongo_port,
        )
        self.collection = mofadb.initialize_database(self.client)

    def shutdown(self) -> None:
        self.client.close()

        super().shutdown()

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
        if self.lammps_completed > self.trainer.minimum_train_size:
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
        if self.raspa_completed > self.trainer.minimum_train_size:
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

        self.generator_tasks: set[ray.ObjectRef] = set()
        self.process_tasks: dict[Future, ray.ObjectRef] = {}
        self.retrain_tasks: dict[int, ray.ObjectRef] = {}

        super().__init__(logger_name="Generator", **kwargs)

    def setup(self) -> None:
        super().setup()

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

    def shutdown(self) -> None:
        cancelled = 0
        for task in self.generator_tasks:
            ray.cancel(task)
            cancelled += 1
        self.logger.info(
            "Cancelled %s generate-ligands task(s) after shutdown",
            cancelled,
        )

        cancelled = 0
        for task in self.process_tasks.values():
            ray.cancel(task)
            cancelled += 1
        self.logger.info(
            "Cancelled %s process-ligands task(s) after shutdown",
            cancelled,
        )

        cancelled = 0
        for task in self.retrain_tasks.values():
            ray.cancel(task)
            cancelled += 1
        self.logger.info(
            "Cancelled %s retrain task(s) after shutdown",
            cancelled,
        )

        super().shutdown()

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
            task = generate_ligands_task.remote(
                batch_size=self.config.batch_size,
                model=self.latest_model_path,
                templates=[ligand],
                n_atoms=size,
                n_samples=self.config.num_samples,
                device=self.config.device,
            )
            self.logger.info(
                "Submitted generate-ligands task (type=%s size=%d, "
                "samples=%d, model=%d)",
                ligand.anchor_type,
                size,
                self.config.num_samples,
                self.model_iteration,
            )
            self.generator_tasks.add(task)
            # Push this generation task back on the queue
            self.generator_queue.put((ligand_id, size))

    @loop
    def submit_process_ligands(self, shutdown: Event) -> None:
        # Passes batches from generator tasks to process ligands tasks
        while not shutdown.is_set():
            # Check which generator tasks have ready batches
            ready, _ = ray.wait(list(self.generator_tasks), timeout=1)
            for task in ready:
                try:
                    batch_ref = next(task)
                except StopIteration:
                    # Task is finished so let another task be submitted.
                    self.generator_count.release()
                    self.generator_tasks.remove(task)
                    self.logger.info("Completed generate-ligands task")
                else:
                    # TODO: this could be passed as ref to next task
                    batch = ray.get(batch_ref)
                    self.logger.info(
                        "Received generate-ligands batch (size=%d)",
                        len(batch),
                    )
                    next_task = process_ligands_task.remote(batch)
                    self.logger.info(
                        "Submitted process-ligands task (size=%d)",
                        len(batch),
                    )
                    future = next_task.future()
                    self.process_tasks[future] = next_task
                    future.add_done_callback(self._process_ligands_callback)

    def _process_ligands_callback(self, future) -> None:
        self.process_tasks.pop(future)
        try:
            valid_ligands, all_ligands = future.result()
        except ray.exceptions.TaskCancelledError:
            self.logger.warning("Cancelled process-ligands task")
        except Exception:
            self.logger.exception("Failure in process-ligands task")
            return

        self.logger.info(
            "Received process-ligands batch (valid=%s, rate=%.2f%%)",
            len(valid_ligands),
            100 * (len(valid_ligands) / len(all_ligands)),
        )

        if len(valid_ligands) == 0:
            return

        assert len(all_ligands) > 0
        # All ligands come from the same batch so they were created by the
        # same model and therefore have the same anchor type.
        anchor_type = all_ligands[0]["anchor_type"]

        action_future = self.assembler.action(
            "submit_ligands",
            anchor_type,
            valid_ligands,
        )
        try:
            action_future.result(timeout=5)
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

        task = retrain_task.remote(
            starting_model=self.initial_model_path,
            run_directory=retrain_dir,
            config_path=self.trainer.config_path,
            examples=examples,
            num_epochs=self.trainer.num_epochs,
            device=self.trainer.device,
        )
        future = task.future()
        self.retrain_tasks[version] = task
        self.logger.info(
            "Submitted retrain task (version=%d, path=%s)",
            version,
            retrain_dir,
        )

        try:
            path = future.result()
        except ray.exceptions.TaskCancelledError:
            self.logger.warning("Cancelled retrain task")
        except Exception:
            self.logger.exception("Error in retrain task")
        else:
            new_model_path = self.config.model_dir / f"model-v{version}.ckpt"
            self.config.model_dir.mkdir(exist_ok=True, parents=True)
            shutil.copyfile(path, new_model_path)
            self.model_iteration = version
            self.latest_model_path = new_model_path
            self.logger.info(
                "Received retrain task (version=%d, path=%s)",
                version,
                new_model_path,
            )
        finally:
            shutil.rmtree(retrain_dir)

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

    def setup(self) -> None:
        super().setup()

        self.assembly_queues = defaultdict(
            lambda: deque(maxlen=self.config.max_queue_depth),
        )
        self.assembly_count = threading.Semaphore(
            value=self.config.num_workers,
        )
        self.assembly_tasks = {}
        self.enabled = threading.Event()

    def shutdown(self) -> None:
        cancelled = 0
        for task in self.assembly_tasks.values():
            ray.cancel(task)
            cancelled += 1
        self.logger.info(
            "Cancelled %s assemble-ligands task(s) after shutdown",
            cancelled,
        )

        super().shutdown()

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
            task = assemble_mofs_task.remote(
                ligand_options=ligand_options,
                nodes=self.config.node_templates,
                to_make=self.config.num_mofs,
                attempts=self.config.max_attempts,
            )
            self.logger.info("Submitted assemble-mofs task")
            future = task.future()
            self.assembly_tasks[future] = task
            future.add_done_callback(self._assembly_task_callback)

    def _assembly_task_callback(self, future: Future) -> None:
        self.assembly_count.release()
        self.assembly_tasks.pop(future)
        try:
            mofs = future.result()
        except ray.exceptions.TaskCancelledError:
            self.logger.warning("Cancelled assemble-mofs task")
        except Exception:
            self.logger.exception("Failure in assemble-mofs task")
            return

        self.logger.info("Received assemble-mofs batch (count=%d)", len(mofs))
        action_future = self.validator.action("submit_mofs", mofs)
        try:
            action_future.result(timeout=5)
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
            lammps_environ=config.lammps_environ.copy(),
            delete_finished=config.delete_finished,
        )
        self.simulations_budget = config.simulation_budget
        self.queue_threshold = max(
            self.config.num_workers,
            int(0.8 * self.config.max_queue_depth),
        )

        super().__init__(logger_name="Validator", **kwargs)

    def setup(self) -> None:
        super().setup()

        self.assembler_enabled = False

        self.validator_queue: deque[MOFRecord] = deque(
            maxlen=self.config.max_queue_depth,
        )
        self.validator_count = Semaphore(value=self.config.num_workers)
        self.validator_tasks = {}

        self.simulations_completed = 0

        self.process_queue: Queue[tuple[MOFRecord, list[ase.Atoms]]] = Queue()
        self.processed_mofs = 0

    def shutdown(self) -> None:
        cancelled = 0
        for task in self.validator_tasks.values():
            ray.cancel(task)
            cancelled += 1
        self.logger.info(
            "Cancelled %d validate-structures task(s) after shutdown",
            cancelled,
        )

        super().shutdown()

    def _check_queue_depth(self, timeout: float) -> None:
        # TODO: lock this to prevent race conditions?
        if len(self.validator_queue) > self.queue_threshold and self.assembler_enabled:
            self.assembler.action("disable_assembly").result(timeout=timeout)
            self.assembler_enabled = False
        elif (
            len(self.validator_queue) < self.queue_threshold
            and not self.assembler_enabled
        ):
            self.assembler.action("enable_assembly").result(timeout=timeout)
            self.assembler_enabled = True

    @action
    def submit_mofs(self, mofs: Sequence[MOFRecord]) -> None:
        # Submit a MOF to the Validator's queue for validation (lammps).
        for mof in mofs:
            self.validator_queue.append(mof)
        self.logger.info(
            "Added mofs to validation queue (count=%d)",
            len(mofs),
        )
        self._check_queue_depth(timeout=5)

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
                self.assembler.action("disable_assembly").result(timeout=5)
            shutdown.set()

    @loop
    def submit_validation(self, shutdown: Event) -> None:
        # Pull from the validation queue and submit validation tasks. Works
        # with the process_validation loop to determine the max number of
        # assembly tasks to run at any point.
        while not shutdown.is_set():
            self._check_queue_depth(timeout=5)

            # Acquired workers are released in the callback on the task future.
            if not self.validator_count.acquire(timeout=1):
                continue

            try:
                record = self.validator_queue.pop()
            except IndexError:
                self.validator_count.release()
                continue

            task = validate_structure_task.remote(
                runner=self.runner,
                mof=record,
                timesteps=self.config.timesteps,
                report_frequency=self.config.report_frequency,
            )
            self.logger.info(
                "Submitted validate-structures task (name=%s)",
                record.name,
            )
            future = task.future()
            self.validator_tasks[future] = task
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
            self.optimizer.action("submit_mof", record).result(timeout=5)
            self.database.action("create_record", record).result(timeout=5)

    def _validate_task_callback(self, future: Future) -> None:
        self.validator_count.release()
        self.validator_tasks.pop(future)
        try:
            record, frames = future.result()
        except ray.exceptions.TaskCancelledError:
            self.logger.warning("Cancelled validate-structures task")
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
        database: Handle[Database],
        config: OptimizerConfig,
        **kwargs: Any,
    ) -> None:
        self.database = database
        self.config = config
        super().__init__(logger_name="Optimizer", **kwargs)

    def setup(self) -> None:
        super().setup()

        self.cp2k_runner = CP2KRunner(
            cp2k_invocation=self.config.cp2k_cmd,
            run_dir=self.config.cp2k_dir,
        )
        self.raspa_runner = RASPARunner(
            raspa_sims_root_path=self.config.raspa_dir,
        )

        self.records: dict[str, MOFRecord] = {}
        self.optimizer_count = threading.Semaphore(self.config.num_workers)
        self.optimize_queue = PriorityQueue()

        self.optimize_tasks = {}
        self.compute_tasks = {}
        self.estimate_tasks = {}

    def shutdown(self) -> None:
        cancelled = 0
        for task in self.optimize_tasks.values():
            ray.cancel(task)
            cancelled += 1
        self.logger.info(
            "Cancelled %s optimize-cells task(s) after shutdown",
            cancelled,
        )

        cancelled = 0
        for task in self.compute_tasks.values():
            ray.cancel(task)
            cancelled += 1
        self.logger.info(
            "Cancelled %s compute-partial-charges task(s) after shutdown",
            cancelled,
        )

        cancelled = 0
        for task in self.estimate_tasks.values():
            ray.cancel(task)
            cancelled += 1
        self.logger.info(
            "Cancelled %s estimate-adsorption task(s) after shutdown",
            cancelled,
        )

        super().shutdown()

    @action
    def submit_mof(self, record: MOFRecord) -> None:
        # Submit a MOF to the Optimizer's queue for optimization.
        priority = record.structure_stability["uff"]
        self.optimize_queue.put((priority, record))
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
                _, record = self.optimize_queue.get(timeout=1)
            except Empty:
                self.optimizer_count.release()
                continue

            if record.name in self.records:
                self.optimizer_count.release()
                continue

            self.records[record.name] = record

            task = optimize_cells_task.remote(
                runner=self.cp2k_runner,
                mof=record,
                steps=self.config.cp2k_steps,
            )
            self.logger.info(
                "Submitted optimize-cells task (name=%s)",
                record.name,
            )
            future = task.future()
            self.optimize_tasks[future] = task
            future.add_done_callback(self._optimize_cells_task_callback)

    def _optimize_cells_task_callback(self, future) -> None:
        self.optimizer_count.release()
        self.optimize_tasks.pop(future)
        try:
            name, cp2k_path = future.result()
        except ray.exceptions.TaskCancelledError:
            self.logger.warning("Cancelled optimize-cells task")
        except Exception:
            self.logger.exception("Failure in optimize-cells task")
            return

        self.logger.info("Completed optimize-cells task (name=%s)", name)

        task = compute_partial_charges_task.remote(cp2k_path)
        self.logger.info("Submitted compute-partial-charges task (name=%s)", name)
        future = task.future()
        self.compute_tasks[future] = task
        future.add_done_callback(self._compute_partial_charges_task_callback)

    def _compute_partial_charges_task_callback(self, future) -> None:
        self.compute_tasks.pop(future)
        try:
            name, atoms = future.result()
        except ray.exceptions.TaskCancelledError:
            self.logger.warning("Cancelled compute-partial-charges task")
        except Exception:
            self.logger.exception("Failure in compute-partial-charges task")
            return

        self.logger.info("Completed compute-partial-charges task (name=%s)", name)

        task = estimate_adsorption_task(
            self.raspa_runner,
            atoms,
            name,
            timesteps=self.config.raspa_timesteps,
        )
        self.logger.info("Submitted estimate-adsorption task (name=%s)", name)
        future = task.future()
        self.estimate_tasks[future] = task
        future.add_done_callback(self._estimate_adsorption_task_callback)

    def _estimate_adsorption_task_callback(self, future) -> None:
        self.estimate_tasks.pop(future)
        try:
            name, storage_mean, storage_std = future.result()
        except ray.exceptions.TaskCancelledError:
            self.logger.warning("Cancelled estimate-adsorption task")
        except Exception:
            self.logger.exception("Failure in estimate-adsorption task")
            return

        self.logger.info("Completed estimate-adsorption task (name=%s)", name)

        record = self.records[name]
        record.gas_storage["C02"] = storage_mean
        record.times["raspa-done"] = datetime.now()

        self.database.action("update_record", record).result(timeout=5)
