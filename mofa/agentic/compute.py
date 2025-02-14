import dataclasses


class ComputeConfig:
    num_generator_workers: int
    num_assembly_workers: int
    num_validator_workers: int
    torch_device: str
    lammps_cmd: tuple[str]
    lammps_env: dict[str, str]


@dataclasses.dataclass(kw_only=True)
class PolarisSingleNode(ComputeConfig):
    # generate-ligands tasks get 1 GPU and 4 CPUs
    num_generator_workers = 1
    # assemble-ligands tasks get 1 CPU
    num_assembly_workers = 8
    # validate-structures get 0.5 GPUs and 2 CPUs
    num_validator_workers = 6
    torch_device = "cuda"
    lammps_cmd = (
        "/eagle/MOFA/jgpaul/lammps/build-gpu-nompi-mixed/lmp -sf gpu -pk gpu 1"
    ).split()
    lammps_env = {}


COMPUTE_CONFIGS = {
    "polaris-single": PolarisSingleNode(),
}
