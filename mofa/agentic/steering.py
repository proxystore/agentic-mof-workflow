from __future__ import annotations

import contextlib
import itertools
import logging
import random
import threading
import time
import traceback
from collections import defaultdict
from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future
from datetime import datetime
from queue import Empty
from queue import Queue
from threading import Event
from threading import Semaphore
from typing import Any

import ase
import ray
from aeris.behavior import action
from aeris.behavior import Behavior
from aeris.behavior import loop
from aeris.handle import Handle

from mofa.agentic.config import AssemblerConfig
from mofa.agentic.config import GeneratorConfig
from mofa.agentic.config import ValidatorConfig
from mofa.agentic.task import assemble_mofs_task
from mofa.agentic.task import generate_ligands_task
from mofa.agentic.task import process_ligands_task
from mofa.agentic.task import validate_structure_task
from mofa.model import LigandDescription
from mofa.model import MOFRecord
from mofa.scoring.geometry import LatticeParameterChange
from mofa.simulation.lammps import LAMMPSRunner
from mofa.utils.conversions import write_to_string

_ray_init_lock = threading.Lock()


class MOFABehavior(Behavior):
    def __init__(self, *, logger_name: str, ray_address: str) -> None:
        self.logger = logging.getLogger(logger_name)
        self.ray_address = ray_address

    def setup(self) -> None:
        pass
        # with _ray_init_lock:
        #     if not ray.is_initialized():
        #         ray.init(self.ray_address, configure_logging=False)

    def shutdown(self) -> None:
        pass
        # with _ray_init_lock:
        #     if ray.is_initialized():
        #         ray.shutdown()


class Database(MOFABehavior):
    def __init__(self, generator: Handle[Generator], **kwargs: Any) -> None:
        super().__init__(logger_name="Database", **kwargs)

    @loop
    def periodic_retrain(self, shutdown: Event) -> None:
        # Waits for sufficient records to be created and then invokes the
        # retrain action on the Generator agent with the top-performing
        # records (either by stability or capacity). This blocks on the
        # retraining so that multiple retrains are not performed at same time.
        ...

    @action
    def create_record(self) -> None:
        # Create a MOF record. This is invoked by the validator to store
        # records for all validated MOFs.
        ...

    @action
    def update_record(self) -> None:
        # Update a MOF record with the gas capacity computed by the Estimator.
        ...


class Generator(MOFABehavior):
    def __init__(
        self,
        assembler: Handle[Assembler],
        config: GeneratorConfig,
        **kwargs: Any,
    ) -> None:
        self.assembler = assembler
        self.config = config

        self.initial_model_path = config.generator_path
        self.latest_model_path = config.generator_path

        self.generator_tasks: set[ray.ObjectRef] = set()
        self.process_tasks: dict[Future, ray.ObjectRef] = {}

        super().__init__(logger_name="Generator", **kwargs)

    def setup(self) -> None:
        super().setup()

        self.generator_queue = Queue()
        self.generator_count = Semaphore(value=self.config.num_workers)

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
            "Cancelled %s generate-ligands task(s) after shutdown", cancelled,
        )

        cancelled = 0
        for task in self.process_tasks.values():
            ray.cancel(task)
            cancelled += 1
        self.logger.info(
            "Cancelled %s process-ligands task(s) after shutdown", cancelled,
        )

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
                "Submitted generate-ligands task (type=%s size=%d, samples=%d)",
                ligand.anchor_type,
                size,
                self.config.num_samples,
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
                        "Received generate-ligands batch (size=%d)", len(batch),
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

    @action
    def retrain(self) -> None:
        # Retrain the model on this set of data and store the retrained
        # for use by the generate_ligands loop (via a callback on the future)
        # of the retraining tasks. This action does not block on training.
        ...


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
            "Cancelled %s assemble-ligands task(s) after shutdown", cancelled,
        )

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
            candidates = sum(
                len(q) for q in self.assembly_queues.copy().values()
            )
            if (
                candidates < self.config.min_ligand_candidates
                # assemble_many() requires 2 COO and 1 cyano ligands
                or "COO" not in self.assembly_queues
                or "cyano" not in self.assembly_queues
            ):
                self.assembly_count.release()
                continue

            ligand_options = {
                k: list(v) for k, v in self.assembly_queues.items()
            }
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
        config: ValidatorConfig,
        **kwargs: Any,
    ) -> None:
        self.assembler = assembler
        self.config = config
        self.runner = LAMMPSRunner(
            lammps_command=config.lammps_command,
            lmp_sims_root_path=config.lmp_sims_root_path,
            lammps_environ=config.lammps_environ,
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

    def _check_queue_depth(self, timeout: float) -> None:
        # TODO: lock this to prevent race conditions?
        if (
            len(self.validator_queue) > self.queue_threshold
            and self.assembler_enabled
        ):
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
            "Added mofs to validation queue (count=%d)", len(mofs),
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
                "Submitted validate-structures task (name=%s)", record.name,
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

    def _validate_task_callback(self, future: Future) -> None:
        self.validator_count.release()
        self.validator_tasks.pop(future)
        try:
            record, frames = future.result()
        except Exception as e:
            self.logger.warning("Failure in validate-structures task: %s", e)
            self.logger.debug(
                "Failure traceback: %r\n%s", e, traceback.format_exc(),
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
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(logger_name="Optimizer", **kwargs)

    @action
    def submit_mof(self) -> None:
        # Submit a MOF to the Optimizer's queue for optimization.
        ...

    @loop
    def submit_optimization(self, shutdown: Event) -> None:
        # Pull from the optimization queue and submit optimization tasks
        # (cp2k). Works with the process_optimization loops to determine
        # the maximum number of tasks to submit.
        ...

    @loop
    def process_optimization(self, shutdown: Event) -> None:
        # Waits for optimization tasks to complete and submits those MOFs
        # for adsorption estimation.
        ...

    @loop
    def process_estimation(self, shutdown: Event) -> None:
        # Waits for estimation tasks to complete and updates the MOF records
        # in the Database agent.
        ...
