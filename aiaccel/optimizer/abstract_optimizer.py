from __future__ import annotations

from typing import Any

from omegaconf.dictconfig import DictConfig

from aiaccel.config import is_multi_objective
from aiaccel.module import AiaccelCore
from aiaccel.parameter import HyperParameterConfiguration
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
            self.workspace.log / "optimizer.log",
            str_to_logging_level(self.config.generic.logging_level),
            str_to_logging_level(self.config.generic.logging_level),
            "Optimizer",
        )

        self.trial_number = self.config.optimize.trial_number
        self.num_of_generated_parameter = 0
        self.params = HyperParameterConfiguration(self.config.optimize.parameters)
        self.write_random_seed_to_debug_log()

    def register_new_parameters(self, params: list[dict[str, float | int | str]], state: str = "ready") -> None:
        """Create hyper parameter files.

        Args:
            params (list[dict[str, float | int | str]]): A list of hyper
                parameter dictionaries.

        Returns:
            None

        Note:
            ::

                param = {
                    'name': ...,
                    'type': ...,
                    'value': ...
                }

        """
        self.storage.hp.set_any_trial_params(trial_id=self.trial_id.get(), params=params)
        self.storage.trial.set_any_trial_state(trial_id=self.trial_id.get(), state=state)
        self.num_of_generated_parameter += 1
        self.logger.debug(f"generated parameters: {params}")

    def generate_initial_parameter(self) -> Any:
        """Generate a list of initial parameters.

        Returns:
            list[dict[str, float | int | str]]: A created list of initial
            parameters.
        """
        sample = self.params.sample(self._rng, initial=True)
        new_params = []

        for s in sample:
            new_param = {"name": s["name"], "type": s["type"], "value": s["value"]}
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
            new_params = self.generate_initial_parameter()
        else:
            new_params = self.generate_parameter()

        return new_params

    def run_optimizer_multiple_times(self, available_pool_size) -> None:
        if available_pool_size <= 0:
            return
        for _ in range(available_pool_size):
            if new_params := self.generate_new_parameter():
                self.register_new_parameters(new_params)
                self.trial_id.increment()
                self.serialize(self.trial_id.integer)

    def resume(self) -> None:
        """When in resume mode, load the previous optimization data in advance.

        Args:
            None

        Returns:
            None
        """
        self.logger.info(f"Resume mode: {self.config.resume}")
        self.trial_id.initial(num=self.config.resume)
        super().deserialize(self.config.resume)
        self.trial_number = self.config.optimize.trial_number

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

    def get_any_trial_params(self, trial_id: int) -> list[dict[str, float | int | str]]:
        """Get any trial parameters.

        Args:
            trial_id (int): Trial ID.

        Returns:
            list[dict[str, float | int | str]]: Any trial parameters.
        """
        return self.storage.hp.get_any_trial_params_dict(trial_id)

    def is_error_free(self) -> bool:
        """Check if all trials are error free.

        Returns:
            bool: True if all trials are error free.
        """
        error_trial_ids = self.storage.error.get_error_trial_id()
        for trial_id in error_trial_ids:
            error_message = self.storage.error.get_any_trial_error(trial_id=trial_id)
            self.logger.error(error_message)
            return False
        return True