from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf.dictconfig import DictConfig

from aiaccel.storage import Storage
from aiaccel.util import TrialId
from aiaccel.workspace import Workspace


class AiaccelCore(object):
    def __init__(self, config: DictConfig, module_name: str) -> None:
        self.config = config
        self.workspace = Workspace(self.config.generic.workspace)
        self.goals = [item.value for item in self.config.optimize.goal]
        self.logger: Any = None
        self.fh: Any = None
        self.ch: Any = None
        self.ch_formatter: Any = None
        self.loop_count = 0
        self.seed = self.config.optimize.rand_seed
        self.storage = Storage(self.workspace.storage_file_path)
        self.trial_id = TrialId(self.config)
        # TODO: Separate the generator if don't want to affect randomness each other.
        self._rng = np.random.RandomState(self.seed)
        self.module_name = module_name

        self.storage.variable.register(
            process_name=self.module_name, labels=["native_random_state", "numpy_random_state", "state"]
        )

    def set_storage(self, storage: Storage):
        self.storage = storage
        self.storage.variable.register(
            process_name=self.module_name, labels=[
                "native_random_state", "numpy_random_state", "state"
            ]
        )

    def set_logger(self, logger_name: str, logfile: Path, file_level: int, stream_level: int, module_type: str) -> None:
        """Set a default logger options.

        Args:
            logger_name (str): A name of a logger.
            logfile (Path): A path to a log file.
            file_level (int): A logging level for a log file output. For
                example logging.DEBUG
            stream_level (int): A logging level for a stream output.
            module_type (str): A module type of a caller.

        Returns:
            None
        """
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(logfile, mode="w")
        fh_formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(filename)-12s line " "%(lineno)-4s %(message)s")
        fh.setFormatter(fh_formatter)
        fh.setLevel(file_level)

        ch = logging.StreamHandler()
        ch_formatter = logging.Formatter(f"{module_type} %(levelname)-8s %(message)s")
        ch.setFormatter(ch_formatter)
        ch.setLevel(stream_level)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def _serialize(self, trial_id: int) -> None:
        """Serialize this module.

        Returns:
            None
        """
        self.storage.variable.d["state"].set(trial_id, self)

        # random state
        self.storage.variable.d["numpy_random_state"].set(trial_id, self.get_numpy_random_state())

    def _deserialize(self, trial_id: int) -> None:
        """Deserialize this module.

        Returns:
            None
        """
        print(self.storage.variable.d["state"])
        print(type(self.storage.variable.d["state"]))
        print(self.storage.variable.d["state"].get(trial_id))
        self.__dict__.update(self.storage.variable.d["state"].get(trial_id).__dict__.copy())

        # random state
        self.set_numpy_random_state(self.storage.variable.d["numpy_random_state"].get(trial_id))

    def write_random_seed_to_debug_log(self) -> None:
        """Writes the random seed to the logger as debug information."""
        self.logger.debug(f"create numpy random generator by seed: {self.seed}")

    def get_numpy_random_state(
        self,
    ) -> dict[str, Any] | tuple[str, np.ndarray[Any, np.dtype[np.uint32]], int, int, float]:
        """Gets random state.

        Returns:
            dict[str, Any] | tuple[str, ndarray[Any, dtype[uint32]], int, int, float]: A tuple representing the
                internal state of the generator if legacy is True. If legacy is False, or the BitGenerator is not
                MT19937, then state is returned as a dictionary.
        """
        return self._rng.get_state()

    def set_numpy_random_state(self, state: Any) -> None:
        """Gets random state.

        Args:
            state (dict[str, Any] | tuple[str, ndarray[Any, np.dtype[uint32]], int, int, float]): A tuple or dictionary
                representing the internal state of the generator.
        """
        self._rng.set_state(state)

    def resume(self) -> None:
        """When in resume mode, load the previous
                optimization data in advance.

        Args:
            None

        Returns:
            None
        """
        if self.config.resume is not None and self.config.resume > 0:
            self._deserialize(self.config.resume)

    def check_error(self) -> bool:
        return True

    def __getstate__(self) -> dict[str, Any]:
        obj = self.__dict__.copy()
        del obj["storage"]
        del obj["config"]
        return obj
