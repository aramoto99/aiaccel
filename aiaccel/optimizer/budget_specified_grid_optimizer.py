from __future__ import annotations

from numpy import nan as np_nan
from omegaconf.dictconfig import DictConfig

from aiaccel.optimizer._grid_point_generator import GridPointGenerator
from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer


class BudgetSpecifiedGridOptimizer(AbstractOptimizer):
    """An optimizer class with grid search algorithm.

    Args:
        options (dict[str, str | int | bool]): A dictionary containing command line options.


    Raises:
        ValueError: Causes when the number of trials is smaller than the least
            space size determined by the parameters of which the number of
            choices is specified.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)

        accept_small_trial_number = self.config.optimize.grid_accept_small_trial_number

        if accept_small_trial_number:
            self.logger.warning(
                'The option "accept_small_trial_number" is valid. ' "Some of grid points may be not seached."
            )

        try:
            self._grid_point_generator = GridPointGenerator(
                self.config.optimize.trial_number,
                self.params.get_parameter_list(),
                sampling_method=self.config.optimize.grid_sampling_method.upper(),
                rng=self._rng,
                accept_small_trial_number=accept_small_trial_number,
            )
        except ValueError as exception:
            self.logger.error(exception)
            raise exception

    def generate_parameter(self) -> list[dict[str, float | int | str]] | None:
        """Generates parameters.

        Returns:
            list[dict[str, float | int | str]] | None: A list of new parameters.
        """
        if self._grid_point_generator.all_grid_points_generated():
            self.logger.info("Generated all of parameters.")
            return None

        new_params: list[dict[str, float | int | str]] = []
        for param, value in zip(self.params.get_parameter_list(), self._grid_point_generator.get_next_grid_point()):
            new_params.append({"name": param.name, "type": param.type, "value": value})
        return new_params

    def generate_initial_parameter(self) -> list[dict[str, float | int | str]]:
        """Generates initial parameters.

        Raises:
            ValueError: Causes when initial parameter could not be generated.

        Returns:
            list[dict[str, float | int | str]]: A list of new parameters.
        """

        for hyperparameter in self.params.get_parameter_list():
            if hyperparameter.initial is not None:
                self.logger.warning(
                    "Initial values cannot be specified for grid search. " "The set initial value has been invalidated."
                )
        if initial_parameter := self.generate_parameter():
            return initial_parameter
        else:
            raise ValueError("Initial parameter could not be generated.")

    def nan_parameter(self) -> list[dict[str, float | int | str]]:
        """Returns a parameter with nan values.

        Returns:
            list[dict[str, float | int | str]]: A list of new parameters.
        """
        return [{"name": param.name, "type": param.type, "value": np_nan} for param in self.params.get_parameter_list()]

    def run_optimizer(self) -> None:
        if new_params := self.generate_new_parameter():
            self.register_new_parameters(self.convert_type_by_config(new_params))
            self.trial_id.increment()
            self.serialize(self.trial_id.integer)
        else:
            self.all_parameters_generated = True
            self.logger.info("Generated all of parameters.")
            # self.register_new_parameters(self.convert_type_by_config(self.nan_parameter()), state="finished")
            # self.storage.result.set_any_trial_objective(trial_id=self.trial_id.integer, objective=[np_nan])
            # self.trial_id.increment()
            # self.serialize(self.trial_id.integer)
