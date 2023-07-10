from __future__ import annotations

import copy
from typing import Any

from numpy import str_
from omegaconf.dictconfig import DictConfig

from aiaccel.config import is_multi_objective
from aiaccel.parameter import HyperParameterConfiguration, is_categorical, is_ordinal, is_uniform_float, is_uniform_int
from aiaccel.util import str_to_logging_level
from aiaccel.module import AiaccelCore
from aiaccel.parameter import HyperParameterConfiguration
from aiaccel.storage import Storage
from aiaccel.util import str_to_logging_level


class AbstractOptimizer(AiaccelCore):
    """An abstract class for Optimizer classes.

    Args:
        config (DictConfig): A DictConfig object which contains optimization
            settings specified by the configuration file and the command line
            options.

    Attributes:
        options (dict[str, str | int | bool]): A dictionary containing
            command line options.
        num_ready (int): A ready number of hyperparameters.
        num_running (int): A running number of hyperprameters.
        num_finished (int): A finished number of hyperparameters.
        num_of_generated_parameter (int): A number of generated hyperparamters.
        all_parameters_generated (bool): Whether all parameters are generated.
            True if all parameters are generated.
        params (HyperParameterConfiguration): A loaded parameter configuration
            object.
        trial_id (TrialId): A TrialId object.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config, "optimizer")
        self.set_logger(
            "root.optimizer",
            self.workspace.log / self.config.logger.file.optimizer,
            str_to_logging_level(self.config.logger.log_level.optimizer),
            str_to_logging_level(self.config.logger.stream_level.optimizer),
            "Optimizer",
        )

        self.trial_number = self.config.optimize.trial_number
        self.num_of_generated_parameter = 0
        self.params = HyperParameterConfiguration(self.config.optimize.parameters)
        self._initial_process()

    def _initial_process(self):
        storage = Storage(self.workspace.storage_file_path)
        self.set_storage(storage)
        self.write_random_seed_to_debug_log()
        self.resume()

    def register_new_parameters(self, params: list[dict[str, float | int | str]]) -> None:
        """Create hyper parameter files.

        Args:
            params (list[dict[str, float | int | str]]): A list of hyper
                parameter dictionaries.

        Returns:
            None

        Note:
            ::

                param = {
                    'parameter_name': ...,
                    'type': ...,
                    'value': ...
                }

        """
        self.storage.hp.set_any_trial_params(trial_id=self.trial_id.get(), params=params)
        self.storage.trial.set_any_trial_state(trial_id=self.trial_id.get(), state="ready")
        self.num_of_generated_parameter += 1

    def generate_initial_parameter(self) -> Any:
        """Generate a list of initial parameters.

        Returns:
            list[dict[str, float | int | str]]: A created list of initial
            parameters.
        """
        sample = self.params.sample(self._rng, initial=True)
        new_params = []

        for s in sample:
            new_param = {"parameter_name": s["name"], "type": s["type"], "value": s["value"]}
            new_params.append(new_param)

        return new_params

    def generate_parameter(self) -> Any:
        """Generate a list of parameters.

        Raises:
            NotImplementedError: Causes when the inherited class does not
                implement.

        Returns:
            list[dict[str, float | int | str]] | None: A created list of
            parameters.
        """
        raise NotImplementedError

    def generate_new_parameter(self) -> list[dict[str, float | int | str]] | None:
        """Generate a list of parameters.

        Returns:
            list[dict[str, float | int | str]] | None: A created list of
            parameters.
        """
        if self.num_of_generated_parameter == 0:
            new_params = self.cast(self.generate_initial_parameter())
        else:
            new_params = self.cast(self.generate_parameter())

        return new_params

    def run_optimizer_multiple_times(self, available_pool_size) -> None:
        if available_pool_size <= 0:
            return
        for _ in range(available_pool_size):
            if new_params := self.generate_new_parameter():
                self.register_new_parameters(new_params)
                self.trial_id.increment()
                self._serialize(self.trial_id.integer)

    def resume(self) -> None:
        """When in resume mode, load the previous optimization data in advance.

        Args:
            None

        Returns:
            None
        """
        if self.config.resume is not None and self.config.resume > 0:
            self.storage.rollback_to_ready(self.config.resume)
            self.storage.delete_trial_data_after_this(self.config.resume)
            self.trial_id.initial(num=self.config.resume)
            self._deserialize(self.config.resume)
            self.trial_number = self.config.optimize.trial_number

    def cast(self, params: list[dict[str, Any]]) -> list[Any] | None:
        """Casts types of parameter values to appropriate tepes.

        Args:
            params (list | None): list of parameters.

        Raises:
            ValueError: Occurs if any of parameter value could not be casted.

        Returns:
            list | None: A list of parameters with casted values. None if given
            `params` is None.
        """
        if params is None or len(params) == 0:
            return params

        casted_params = []

        for param in params:
            _param = copy.deepcopy(param)
            param_type = _param["type"]
            param_value = _param["value"]

            # None: str to NoneType
            if type(_param["value"]) in [str, str_]:
                if _param["value"].lower() == "none":
                    _param["value"] = None
                    _param["type"] = str(type(None))

            try:
                if is_categorical(param_type) or is_ordinal(param_type):
                    casted_params.append(_param)
                    continue
                if is_uniform_float(param_type):
                    _param["value"] = float(param_value)
                if is_uniform_int(param_type):
                    _param["value"] = int(param_value)
                casted_params.append(_param)

            except ValueError as e:
                raise ValueError(e)

        return casted_params

    def get_any_trial_objective(self, trial_id: int) -> Any:
        """Get any trial result.

            if the objective is multi-objective, return the list of objective.

        Args:
            trial_id (int): Trial ID.

        Returns:
            Any: Any trial result.
        """

        objective = self.storage.result.get_any_trial_objective(trial_id)
        if objective is None:
            return None

        if is_multi_objective(self.config):
            return objective
        else:
            return objective[0]
