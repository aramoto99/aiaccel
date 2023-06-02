from __future__ import annotations

import copy
from typing import Any

import numpy as np
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

from aiaccel.config import is_multi_objective
from aiaccel.converted_parameter import ConvertedParameterConfiguration
from aiaccel.optimizer._nelder_mead import NelderMead
from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer
from aiaccel.parameter import HyperParameter


class NelderMeadOptimizer(AbstractOptimizer):
    """An optimizer class with nelder mead algorithm.

    Args:
        config (DictConfig): A DictConfig object which contains optimization
            settings specified by the configuration file and the command line
            options.

    Attributes:
        nelder_mead (NelderMead): A class object implementing Nelder-Mead
            method.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.params: ConvertedParameterConfiguration = ConvertedParameterConfiguration(
            self.params, convert_log=True, convert_int=True, convert_choices=True, convert_sequence=True
        )
        self.n_params = len(self.params.get_parameter_list())
        self.bdrys = np.array([[p.lower, p.upper] for p in self.params.get_parameter_list()])
        self.n_dim = len(self.bdrys)
        self.store = {
            "reflect": None,
            "inside_contract": None,
            "outside_contract": None,
            "expand": None,
            "shrink": None
        }
        self.nelder_mead: Any = None
        if is_multi_objective(self.config):
            raise NotImplementedError("Nelder-Mead optimizer does not support multi-objective optimization.")

    def create_initial_values(self, initial_parameters: list[dict[str, Any]]) -> np.ndarray[Any, Any]:
        initial_values = [
            [self.create_initial_value(initial_parameters, dim, num_of_initials) for dim in range(self.n_params)]
            for num_of_initials in range(self.n_dim + 1)
        ]
        return np.array(initial_values)

    def create_initial_value(self, initial_parameters: Any, dim: int, num_of_initials: int) -> Any:
        params = self.params.get_parameter_list()
        val = params[dim].sample(rng=self._rng)["value"]
        if initial_parameters is None:
            val = params[dim].sample(rng=self._rng)["value"]
            return val

        val = initial_parameters[dim]["value"]
        if not isinstance(val, (list, ListConfig)):
            initial_parameters[dim]["value"] = [val]

        vals = initial_parameters[dim]["value"]
        assert isinstance(vals, (list, ListConfig))
        if num_of_initials < len(vals):
            val = initial_parameters[dim]["value"][num_of_initials]
            return val
        else:
            val = params[dim].sample(rng=self._rng)["value"]
            return val

    def generate_initial_parameter(self) -> list[dict[str, float | int | str]] | None:
        """Generate initial parameters.

        Returns:
            list[dict[str, float | int | str]] | None: A list of new
            parameters. None if `self.nelder_mead` is already defined.
        """
        initial_parameters = super().generate_initial_parameter()
        initial_parameters = self.create_initial_values(initial_parameters)
        self.logger.debug(f"initial_parameters: {initial_parameters}")
        if self.nelder_mead is not None:
            return None
        self.nelder_mead = NelderMead(initial_parameters=initial_parameters)
        return self.generate_parameter()

    def nelder_mead_main(self) -> list[Any] | None:
        """Nelder Mead's main module.

        Args:
            None

        Returns:
            searched_params (list): Result of optimization.
        """
        searched_params = self.nelder_mead.search()
        if searched_params is None:
            self.logger.info("generate_parameter(): reached to max iteration.")
            return None
        if len(searched_params) == 0:
            return None
        return searched_params

    def generate_parameter(self) -> list[dict[str, float | int | str]] | None:
        """Generate parameters.

        Returns:
            list[dict[str, float | int | str]] | None: A list of created
            parameters.

        Raises:
            TypeError: Causes when an invalid parameter type is set.
        """
        new_params = self.nelder_mead_main()
        if new_params is None:
            return None
        return new_params

    def out_of_boundary(self, y):  # TODO: fix
        for yi, b in zip(y, self.bdrys):
            if b[0] <= yi <= b[1]:
                pass
            else:
                return True
        return False