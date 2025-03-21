from __future__ import annotations

from typing import Any

from parsl.addresses import address_by_hostname
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import MpiExecLauncher
from parsl.providers import LocalProvider
from parsl.providers import PBSProProvider


def get_parsl_config(name: str, run_dir: str, **kwargs: Any) -> Config:
    if name == "aurora":
        return get_aurora_config(run_dir)
    elif name == "local":
        return get_local_config(run_dir, **kwargs)
    elif name == "polaris":
        return get_polaris_config(run_dir)
    else:
        raise AssertionError(f"Unknown Parsl config name: {name}.")


def get_aurora_config(run_dir: str) -> Config:
    tile_names = [f"{gid}.{tid}" for gid in range(6) for tid in range(2)]
    user_opts = {
        "worker_init": (
            "module load frameworks; "
            "conda activate /flare/proxystore/jgpaul/agentic-mof-workflow; "
            f"cd {run_dir}"
        ),
        "scheduler_options": "#PBS -l filesystems=home:flare",
        "account": "proxystore",
        "queue": "debug",
        "walltime": "1:00:00",
        "nodes_per_block": 1,
        "cpus_per_node": 208,
    }

    executor = HighThroughputExecutor(
        available_accelerators=tile_names,
        max_workers_per_node=len(tile_names),
        cpu_affinity=(
            "list:0-7,104-111:8-15,112-119:16-23,120-127"
            ":24-31,128-135:32-39,136-143:40-47,144-151"
            ":52-59,156-163:60-67,164-171:68-75,172-179"
            ":76-83,180-187:84-91,188-195:92-99,196-203"
        ),
        prefetch_capacity=0,
        provider=PBSProProvider(
            account=user_opts["account"],
            queue=user_opts["queue"],
            worker_init=user_opts["worker_init"],
            walltime=user_opts["walltime"],
            scheduler_options=user_opts["scheduler_options"],
            launcher=MpiExecLauncher(bind_cmd="--cpu-bind", overrides="--ppn 1"),
            select_options="",
            nodes_per_block=user_opts["nodes_per_block"],
            init_blocks=0,
            min_blocks=0,
            max_blocks=1,
            cpus_per_node=user_opts["cpus_per_node"],
        ),
    )

    return Config(
        executor=[executor],
        run_dir=run_dir,
        initialize_logging=False,
        retries=0,
        app_cache=False,
    )


def get_local_config(
    run_dir: str,
    workers_per_node: int,
) -> Config:
    executor = HighThroughputExecutor(
        label="htex-local",
        max_workers_per_node=workers_per_node,
        address=address_by_hostname(),
        cores_per_worker=1,
        provider=LocalProvider(init_blocks=1, max_blocks=1),
    )
    return Config(
        executors=[executor],
        run_dir=run_dir,
        initialize_logging=True,
        retries=0,
        app_cache=False,
    )


def get_polaris_config(run_dir: str) -> Config:
    user_opts = {
        "worker_init": (
            "module use /soft/modulefiles; module load conda; "
            "conda activate /eagle/MOFA/jgpaul/agentic-mof-workflow/env; "
            f"cd {run_dir}"
        ),
        "scheduler_options": "#PBS -l filesystems=home:eagle",
        "account": "proxystore",
        "queue": "debug-scaling",
        "walltime": "1:00:00",
        "nodes_per_block": 3,
        "cpus_per_node": 32,
        "available_accelerators": 4,
    }

    executor = HighThroughputExecutor(
        label="polaris-htex",
        heartbeat_period=15,
        heartbeat_threshold=120,
        worker_debug=True,
        available_accelerators=user_opts["available_accelerators"],
        max_workers_per_node=user_opts["available_accelerators"],
        # This give optimal binding of threads to GPUs on a Polaris node
        cpu_affinity="list:24-31,56-63:16-23,48-55:8-15,40-47:0-7,32-39",
        prefetch_capacity=0,
        provider=PBSProProvider(
            launcher=MpiExecLauncher(
                bind_cmd="--cpu-bind",
                overrides="--depth=64 --ppn 1",
            ),
            account=user_opts["account"],
            queue=user_opts["queue"],
            select_options="ngpus=4",
            # PBS directives (header lines)
            scheduler_options=user_opts["scheduler_options"],
            # Command to be run before starting a worker, such as:
            worker_init=user_opts["worker_init"],
            # number of compute nodes allocated for each block
            nodes_per_block=user_opts["nodes_per_block"],
            init_blocks=0,
            min_blocks=0,
            max_blocks=1,  # Can increase more to have more parallel jobs
            cpus_per_node=user_opts["cpus_per_node"],
            walltime=user_opts["walltime"],
        ),
    )

    return Config(
        executors=[executor],
        run_dir=run_dir,
        initialize_logging=True,
        retries=0,
        app_cache=False,
    )
