import dataclasses


@dataclasses.dataclass
class ComputeConfig:
    num_generator_workers: int
    num_assembly_workers: int
    num_validator_workers: int
    torch_device: str
    lammps_cmd: tuple[str]
    lammps_env: dict[str, str]


@dataclasses.dataclass
class PolarisSingleNode(ComputeConfig):
    num_generator_workers = 1
    num_assembly_workers = 8
    num_validator_workers = 3
    torch_device = "cuda"
    lammps_cmd = ("lmp_serial",)
    lammps_env = {}


COMPUTE_CONFIGS = {
    "polaris-single": PolarisSingleNode(),
}
