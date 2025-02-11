from __future__ import annotations

import collections
import itertools
import threading
import logging
from concurrent.futures import Future
from queue import Queue, Empty
from threading import Semaphore, Event
from typing import Sequence

import ray
from aeris.behavior import Behavior, loop, action
from aeris.handle import Handle

from mofa.agentic.config import AssemblyConfig, GeneratorConfig, ValidatorConfig
from mofa.agentic.task import (
    assemble_mofs_task,
    validate_structure_task,
    process_ligands_task,
    generate_ligands_task,
)
from mofa.model import MOFRecord, LigandDescription
from mofa.simulation.lammps import LAMMPSRunner


logger = logging.getLogger(__name__)


class Database(Behavior):
    def __init__(self, generator: Handle[Generator]) -> None: ...

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


class Generator(Behavior):
    def __init__(
        self,
        assembler: Handle[Assembler],
        config: GeneratorConfig,
    ) -> None:
        self.assembler = assembler
        self.config = config

        self.initial_model_path = config.generator_path
        self.lastest_model_path = config.generator_path

        tasks = list(
            itertools.product(range(len(config.templates)), config.atom_counts)
        )
        tasks.shuffle()
        self.generator_queue = Queue(tasks)
        self.generator_count = Semaphore(value=config.num_workers)
        self.generator_tasks: set[ray.ObjectRef] = set()
        self.process_tasks: dict[Future, ray.ObjectRef] = {}

    @loop
    def submit_generate_ligands(self, shutdown: Event) -> None:
        # Submit generate_ligands_task while there are available resources
        # as indicated by the self.generator_count semaphore.
        while not shutdown.is_set():
            # Acquired workers are released when the generator tasks
            # raise StopIteration exceptions in submit_process_ligands.
            if not self.generator_count.acquire(timeout=1):
                continue

            ligand_id, size = self.generator_queue.get()
            ligand = self.config.templates[ligand_id]

            task = ray.remote(
                generate_ligands_task,
                batch_size=self.config.batch_size,
                model=self.latest_model_path,
                templates=[ligand],
                n_atoms=size,
                n_samples=self.config.num_samples,
                device=self.config.device,
            )
            self.generator_tasks.put(task)
            # Push this generation task back on the queue
            self.generate_queue.append((ligand_id, size))

    @loop
    def submit_process_ligands(self, shutdown: Event) -> None:
        # Passes batches from generator tasks to process ligands tasks
        while not shutdown.is_set():
            # Check which generator tasks have ready batches
            ready, _ = ray.wait(self.generator_tasks)
            for task in ready:
                try:
                    batch = next(task)
                except StopIteration:
                    # Task is finished so let another task be submitted.
                    self.generator_count.release()
                else:
                    task = process_ligands_task(batch).remote()
                    future = task.future()
                    self.process_tasks[future] = task
                    future.add_done_callback(self._process_ligands_callback)

    def _process_ligands_callback(self, future) -> None:
        self.process_tasks.pop(future)
        try:
            valid_ligands, all_ligands = future.result()
        except Exception:
            logger.exception("Error in process ligands task")

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
            action_future.result()
        except Exception:
            logger.exception("Error requesting submit_ligands action")

    @action
    def retrain(self) -> None:
        # Retrain the model on this set of data and store the retrained
        # for use by the generate_ligands loop (via a callback on the future)
        # of the retraining tasks. This action does not block on training.
        ...


class Assembler(Behavior):
    def __init__(
        self,
        *,
        validator: Handle[Validator],
        config: AssemblyConfig,
    ) -> None:
        self.assembly_queues = collections.defaultdict(
            lambda: collections.deque(maxlen=config.max_queue_depth),
        )
        self.assembly_count = threading.Semaphore(value=config.num_workers)
        self.assembly_tasks = {}

    @action
    def submit_ligands(
        self,
        anchor_type: str,
        valid_ligands: Sequence[LigandDescription],
    ) -> None:
        # Submit a batch of ligands to the assembler's queue for MOF assembly.
        self.assembly_queues[anchor_type].extend(valid_ligands)

    @loop
    def submit_assembly(self, shutdown: Event) -> None:
        # Pull from assembly queue and submit assembly tasks. Works with
        # the process_assembly loop to determine max number of assembly
        # tasks to run at any point.
        while not shutdown.is_set():
            # Acquired workers are released in the callback on the task future.
            if not self.assembly_count.acquire(timeout=1):
                continue

            candidates = sum(len(q) for q in self.assembly_queues.values())
            if candidates < self.config.min_ligand_candidates:
                self.assembly_count.release()
                continue

            task = assemble_mofs_task(
                dict((k, list(v)) for k, v in self.assembly_queue.items()),
                nodes=self.config.node_templates,
                to_make=self.config.num_mofs,
                attemps=self.config.max_attempts,
            ).remote()
            future = task.future()
            self.assembly_tasks[future] = task
            future.add_done_callback(self._assembly_task_callback)

    def _assembly_task_callback(self, future: Future) -> None:
        self.assembly_count.release()
        self.assembly_tasks.pop(future)
        try:
            mofs = future.result()
        except Exception:
            logger.exception("Error in assembly task")

        action_future = self.validator.action("submit_mofs", mofs)
        try:
            action_future.result()
        except Exception:
            logger.exception("Error request submit_mofs action")


class Validator(Behavior):
    def __init__(self, config: ValidatorConfig) -> None:
        self.config = config
        self.runner = LAMMPSRunner(
            lammps_command=config.lammps_command,
            lmp_sims_root_path=config.lmp_sims_root_path,
            lammps_environ=config.lammps_environ,
            delete_finished=config.delete_finished,
        )

        self.validator_queue = Queue(maxsize=config.max_queue_depth)
        self.validator_count = Semaphore(value=config.num_workers)
        self.validator_tasks = {}

        self.process_queue = Queue()

    @action
    def submit_mof(self, mofs: Sequence[MOFRecord]) -> None:
        # Submit a MOF to the Validator's queue for validation (lammps).
        for mof in mofs:
            self.validator_queue.put(mof)

    @loop
    def submit_validation(self, shutdown: Event) -> None:
        # Pull from the validation queue and submit validation tasks. Works
        # with the process_validation loop to determine the max number of
        # assembly tasks to run at any point.
        while not shutdown.is_set():
            # Acquired workers are released in the callback on the task future.
            if not self.validator_count.acquire(timeout=1):
                continue

            try:
                mof = self.validator_queue.get(timeout=1)
            except Empty:
                self.assembly_count.release()

            task = validate_structure_task(
                runner=self.runner,
                mof=mof,
                timesteps=self.config.timesteps,
                report_frequency=self.config.report_frequency,
            ).remote()
            future = task.future()
            self.validator_tasks[future] = task
            future.add_done_callback(self._validate_task_callback)

    @loop
    def process_validation(self, shutdown: Event) -> None:
        while not shutdown.is_set():
            try:
                _ = self.process_queue.get(timeout=1)
            except Empty:
                continue

            # Compute the lattice strain
            # TODO: figure this out and log
            # scorer = LatticeParameterChange()
            # traj_vasp = [write_to_string(t, "vasp") for t in traj]
            # strain = scorer.score_mof(record)

    def _validate_task_callback(self, future: Future) -> None:
        self.validator_count.release()
        self.validator_tasks.pop(future)
        try:
            result = future.result()
        except Exception:
            logger.exception("Error in validation task")
        self.process_queue.put(result)


class Optimizer(Behavior):
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
