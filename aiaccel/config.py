from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from importlib.resources import read_text
from pathlib import Path
from typing import Any, List, Optional, Union

from omegaconf import OmegaConf
from omegaconf.base import Container
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

from aiaccel.common import resource_type_abci, resource_type_local, resource_type_memory, resource_type_mpi, goal_maximize, goal_minimize

class ResourceType(Enum):
    abci: str = resource_type_abci
    local: str = resource_type_local
    python_local: str = resource_type_memory
    mpi: str = resource_type_mpi

    @classmethod
    def _missing_(cls, value: Any) -> Any | None:
        value = value.lower()
        for member in cls:
            if member.value == value:
                return member
        return None


class OptimizerDirection(Enum):
    minimize: str = goal_minimize
    maximize: str = goal_maximize

    @classmethod
    def _missing_(cls, value: Any) -> Any | None:
        value = value.lower()
        for member in cls:
            if member.value == value:
                return member
        return None


@dataclass
class GenericConfig:
    workspace: str
    job_command: str
    python_file: str
    function: str
    enabled_variable_name_argumentation: bool
    main_loop_sleep_seconds: Union[float, int]
    logging_level: str
    venv_dir: Optional[str]
    aiaccel_dir: Optional[str]


@dataclass
class ResourceConifig:
    type: ResourceType
    num_workers: int
    mpi_npernode: Optional[int]
    mpi_bat_rt_type: Optional[str]
    mpi_bat_rt_num: Optional[int]
    mpi_bat_h_rt: Optional[str]
    mpi_bat_file: Optional[str]
    mpi_hostfile: Optional[str]
    mpi_gpu_mode: Optional[bool]
    mpi_bat_make_file: Optional[bool]


@dataclass
class AbciConifig:
    group: str
    job_script_preamble: str
    job_script_preamble_path: Optional[str]
    job_execution_options: Optional[str]


@dataclass
class ParameterConfig:
    name: str
    type: str
    lower: Union[float, int]
    upper: Union[float, int]
    # initial: Optional[Union[None, float, int, str, List[Union[float, int]]]]  # Unions of containers are not supported
    initial: Optional[Any]
    choices: Optional[List[Union[float, int, str]]]
    sequence: Optional[List[Union[float, int]]]
    log: Optional[bool]
    base: Optional[int]
    step: Optional[Union[int, float]]
    num_numeric_choices: Optional[int]


@dataclass
class OptimizeConifig:
    search_algorithm: str
    goal: List[OptimizerDirection]
    trial_number: int
    rand_seed: int
    sobol_scramble: bool
    grid_accept_small_trial_number: bool
    grid_sampling_method: str
    parameters: List[ParameterConfig]


@dataclass
class JobConfig:
    job_timeout_seconds: float
    max_failure_retries: int
    trial_id_digits: int


@dataclass
class ConditionConfig:
    loop: int
    minimum: Union[float, int]
    maximum: Union[float, int]
    passed: Optional[bool]
    best: Optional[Union[float, int]]


@dataclass
class Config:
    generic: GenericConfig
    resource: ResourceConifig
    ABCI: AbciConifig
    optimize: OptimizeConifig
    job_setting: JobConfig
    clean: Optional[bool]
    resume: Optional[Union[None, int]]
    config_path: Optional[Union[None, Path, str]]


def set_structs_false(conf: Container) -> None:
    OmegaConf.set_struct(conf, False)
    if hasattr(conf, "__iter__"):
        for item in conf:
            if isinstance(conf.__dict__["_content"], dict):
                set_structs_false(conf.__dict__["_content"][item])


def load_config(config_path: Path | str) -> DictConfig:
    """
    Load any configuration files, return the DictConfig object.
    Args:
        config_path (str): A path to a configuration file.
    Returns:
        config: DictConfig object
    """
    path = Path(config_path).resolve()

    if not path.exists():
        raise ValueError("Config is not found.")

    base = OmegaConf.structured(Config)
    default = OmegaConf.create(read_text("aiaccel", "default_config.yaml"))
    customize = OmegaConf.load(path)
    customize.config_path = str(path)
    if not isinstance(customize.optimize.goal, ListConfig):
        customize.optimize.goal = ListConfig([customize.optimize.goal])

    config: Union[ListConfig, DictConfig] = OmegaConf.merge(base, default)
    set_structs_false(config)
    config = OmegaConf.merge(config, customize)
    if not isinstance(config, DictConfig):
        raise RuntimeError("The configuration is not a DictConfig object.")
    return config


def is_multi_objective(config: DictConfig) -> bool:
    """Is the multi-objective option set in the configuration.

    Args:
        config (Config): A configuration object.

    Returns:
        bool: Is the multi--objective option set in the configuration or not.
    """
    return isinstance(config.optimize.goal, ListConfig) and len(config.optimize.goal) > 1
