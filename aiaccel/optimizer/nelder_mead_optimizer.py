from __future__ import annotations

import copy
from typing import Any

import numpy as np
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

from aiaccel.config import is_multi_objective
from aiaccel.converted_parameter import ConvertedParameterConfiguration
from aiaccel.optimizer import AbstractOptimizer, NelderMead
from aiaccel.optimizer._nelder_mead import NelderMead
from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer
from aiaccel.util.buffer import Buffer


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
        self.base_params = self.params.get_empty_parameter_dict()
        self.n_params = len(self.params.get_parameter_list())
        self.param_names = self.params.get_parameter_names()
        self.bdrys = np.array([[p.lower, p.upper] for p in self.params.get_parameter_list()])
        self.n_dim = len(self.bdrys)
        self.n_vertices = self.n_dim + 1
        self.nelder_mead: Any = None
        self.range_of_trials = [0] * 2
        if is_multi_objective(self.config):
            raise NotImplementedError("Nelder-Mead optimizer does not support multi-objective optimization.")
        self.single_or_multiple_trial_params = []
        self.buff = Buffer(["finished"])
        self.buff.d["finished"].set_max_len(2)

    def new_finished(self) -> list[int]:
        finished = self.storage.get_finished()
        if len(finished) == 0:
            return []
        self.buff.d["finished"].add(self.storage.get_finished())
        if self.buff.d["finished"].Len >= 2:
            pre = self.buff.d["finished"].Pre
            now = self.buff.d["finished"].Now
            return list(set(pre) ^ set(now))
        else:
            return []

    def convert_ndarray_to_parameter(self, multiple_parameter_lists: list[np.ndarray]) -> list[dict[str, float | int | str]]:
        """Convert a list of numpy.ndarray to a list of parameters.
        """
        num_of_lists = len(multiple_parameter_lists)
        trial_params = []
        for i in range(num_of_lists):
            new_params = copy.deepcopy(self.base_params)
            params = multiple_parameter_lists[i]
            for name, value in zip(self.param_names, params):
                for new_param in new_params:
                    if new_param["name"] == name:
                        new_param["value"] = value
            trial_params.append(new_params)
        return trial_params

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
        self.nelder_mead = NelderMead(n_dim=self.n_dim, initial_parameters=initial_parameters)
        return self.generate_parameter()

    def generate_parameter(self) -> list[dict[str, float | int | str]] | None:
        """Generate parameters.

        Returns:
            list[dict[str, float | int | str]] | None: A list of created
            parameters.

        Raises:
            TypeError: Causes when an invalid parameter type is set.
        """


        nm_state = self.nelder_mead.get_state()
        if nm_state in {
            "wait_initialize_completed", "wait_reflect_completed", "wait_expand_completed",
            "wait_inside_contract_completed", "wait_outside_contract_completed", "wait_shrink_completed"
        }:
            if not(self.storage.get_num_ready() == 0 and self.storage.get_num_running() == 0):
                return None  # wait
            n_waits = self.nelder_mead.get_n_waits()
            current_trial_id = self.trial_id.integer
            print(f"current_trial_id: {current_trial_id}")
            print(f"n_waits: {n_waits}")
            if current_trial_id >= n_waits:
                self.range_of_trials[0] = current_trial_id - n_waits
                self.range_of_trials[1] = current_trial_id
                print(f"self.range_of_trials: {self.range_of_trials}")
                trials = self.get_n_trials(self.range_of_trials[0], self.range_of_trials[1])
                xs, ys = self.create_simplex(trials)
                if n_waits == self.n_vertices:
                    self.nelder_mead.update(xs, ys)
                    if nm_state == "wait_initialize_completed":
                        self.nelder_mead.after_initialize()
                    elif nm_state == "wait_shrink_completed":
                        self.nelder_mead.aftter_shrink()
                else:
                    print(f"xs: {xs}, ys: {ys}")
                    for i in range(n_waits):
                        self.nelder_mead.pop()
                        self.nelder_mead.push(xs[i], ys[i])
                    if nm_state == "wait_reflect_completed":
                        self.nelder_mead.after_reflect()
                    elif nm_state == "wait_expand_completed":
                        self.nelder_mead.after_expand()
                    elif nm_state == "wait_inside_contract_completed":
                        self.nelder_mead.after_inside_contract()
                    elif nm_state == "wait_outside_contract_completed":
                        self.nelder_mead.after_outside_contract()
        elif nm_state == "reflect":
            ...
        elif nm_state == "expand":
            ...
        elif nm_state == "inside_contract":
            ...
        elif nm_state == "outside_contract":
            ...
        elif nm_state == "shrink":
            ...

        searched_params = self.nelder_mead_main()
        trial_params: list[dict] = self.convert_ndarray_to_parameter(searched_params)
        for param in trial_params:
            self.single_or_multiple_trial_params.append(param)
    
        if len(self.single_or_multiple_trial_params) == 0:
            return None

        new_params = self.single_or_multiple_trial_params.pop(0)
        self.logger.debug(f"new_params: {new_params}")
        return new_params

    def get_n_trials(self, trial_id_1: int, trial_id_2: int) -> list[dict[str, Any]]:
        trials = []
        if trial_id_1 == trial_id_2:
            trials.append(self.storage.get_hp_dict(trial_id_1))
            print(self.self.storage.get_finished())
            print(f"trials: {trials}")
        else:
            target_trial_ids = list(range(trial_id_1, trial_id_2))
            for trial_id in target_trial_ids:
                trials.append(self.storage.get_hp_dict(trial_id))
        return trials

    def create_simplex(self, trials: list[dict[str, Any]]) -> list[Any]:
        xs = []
        ys = []
        for trial in trials:
            params = trial["parameters"]
            values = []
            for param in params:
                values.append(param['value'])
            xs.append(values)
            ys.append(trial["result"][0])  # trial["result"] is a list of objective values
        return np.array([xs])[0], np.array([ys][0])

    def to_ndarray_xs_and_y(self, trial: dict) -> list[Any]:
        params = trial["parameters"]
        xs = []
        y = trial["result"][0]  # trial["result"] is a list of objective values
        for param in params:
            xs.append(param['value'])
        return np.array([xs])[0], np.array([y][0])

    def nelder_mead_main(self) -> list[Any] | None:
        """Nelder Mead's main module.

        Args:
            None

        Returns:
            searched_params (list): Result of optimization.
        """

        searched_params: np.ndarray = self.nelder_mead.main()
        # print(f"searched_params: {searched_params}, state: {self.nelder_mead.get_state()}")
        if searched_params is None:
            return []
        if len(searched_params) == 0:
            return []
        # else:
        #     self.nelder_mead.reset()
        return searched_params

    def out_of_boundary(self, y):  # TODO: fix
        for yi, b in zip(y, self.bdrys):
            if b[0] <= yi <= b[1]:
                pass
            else:
                return True
        return False
