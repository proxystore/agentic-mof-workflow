from __future__ import annotations

import dataclasses


class ComputeConfig:
    num_generator_workers: int
    num_assembly_workers: int
    num_validator_workers: int
    num_optimizer_workers: int
    num_estimator_workers: int
    torch_device: str
    cp2k_cmd: str
    lammps_cmd: tuple[str, ...]
    lammps_env: dict[str, str]


@dataclasses.dataclass(kw_only=True)
class PolarisSingleNode(ComputeConfig):
    # retrain tasks get 1 GPU and 8 CPUs
    num_retrain_workers = 1
    # generate-ligands tasks get 1 GPU and 4 CPUs
    num_generator_workers = 2
    # assemble-ligands tasks get 1 CPU
    num_assembly_workers = 8
    # validate-structures get 0.5 GPUs and 2 CPUs
    num_validator_workers = 1
    # optimize-cells get 1 GPU and 8 CPUs
    num_optimizer_workers = 0
    num_estimator_workers = 2
    torch_device = "cuda"
    cp2k_cmd = (
        "module restore && "
        "mpiexec -n 1 --ppn 1 --env OMP_NUM_THREADS=8 --hosts $HOSTNAME "
        # "--cpu-bind depth --depth 8 "
        # "/lus/eagle/projects/ExaMol/cp2k-2024.1/set_affinity_gpu_polaris.sh "
        "/lus/eagle/projects/ExaMol/cp2k-2024.1/exe/local_cuda/cp2k_shell.psmp "
    )
    lammps_cmd = (
        "/eagle/MOFA/jgpaul/lammps/build-gpu-nompi-mixed/lmp",
        "-sf",
        "gpu",
        "-pk",
        "gpu",
        "1",
    )
    lammps_env: dict[str, str] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(kw_only=True)
class FederatedConfig(ComputeConfig):
    # Generator, retrain, and validators share a single debug aurora node
    # with 12 GPUs
    num_retrain_workers = 1
    num_generator_workers = 3
    num_validator_workers = 8

    # Assembly and optimizer share the chameleon node
    num_assembly_workers = 64
    num_estimator_workers = 16

    # Optimizer workers have a single polaris debug job with four GPUs
    num_optimizer_workers = 4

    torch_device = "xpu"
    # Runs on Polaris
    cp2k_cmd = (
        "module restore && "
        "mpiexec -n 1 --ppn 1 --env OMP_NUM_THREADS=8 --hosts $HOSTNAME "
        # "--cpu-bind depth --depth 8 "
        # "/lus/eagle/projects/ExaMol/cp2k-2024.1/set_affinity_gpu_polaris.sh "
        "/lus/eagle/projects/ExaMol/cp2k-2024.1/exe/local_cuda/cp2k_shell.psmp "
    )
    # TODO: aurora version
    lammps_cmd = (
        "/eagle/MOFA/jgpaul/lammps/build-gpu-nompi-mixed/lmp",
        "-sf",
        "gpu",
        "-pk",
        "gpu",
        "1",
    )
    lammps_env: dict[str, str] = dataclasses.field(default_factory=dict)


COMPUTE_CONFIGS = {
    "federated": FederatedConfig(),
    "polaris-single": PolarisSingleNode(),
}
