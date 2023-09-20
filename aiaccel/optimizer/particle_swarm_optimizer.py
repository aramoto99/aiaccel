from __future__ import annotations

import copy
import numpy as np
import string

from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

from typing import Any, Dict, List

from aiaccel.common import goal_maximize
from aiaccel.config import is_multi_objective
from aiaccel.converted_parameter import ConvertedParameterConfiguration
from aiaccel.optimizer import AbstractOptimizer


name_rng = np.random.RandomState()


class Value:
    def __init__(self, id: str, value: Any) -> None:
        self.id: str = id
        self.value: Any = value


class Particle:
    def __init__(self, num_dimensions: int, initial_position: list[float] | None = None) -> None:
        if initial_position is None:
            self.position = np.random.rand(num_dimensions)
        else:
            self.position = np.array(initial_position)
        self.position = np.array(initial_position)
        self.velocity = np.zeros(num_dimensions)   # 初期速度はゼロ
        self.best_position = self.position.copy()  # 最良位置は初期位置と同じ
        self.best_value = np.inf  # 最良評価値は無限大で初期化
        self.value = None         # 評価値はまだ計算していないためNone

        self.inertia_weight = 0.9    # 慣性重み
        self.cognitive_weight = 2.0  # 認知重み
        self.social_weight = 2.0     # 社会的重み

        self.id: str = self.generate_random_name()
        self.count_eval = 0

    @property
    def coordinates(self) -> np.ndarray[Any, Any]:
        return self.position

    def generate_random_name(self, length: int = 10) -> str:
        if length < 1:
            raise ValueError("Name length should be greater than 0.")
        rands = [name_rng.choice(list(string.ascii_letters + string.digits))[0] for _ in range(length)]
        return "".join(rands)

    def update_velocity(self, global_best_position: np.ndarray):
        # 速度の更新を実装する
        self.velocity = self.inertia_weight * self.velocity + self.cognitive_weight * np.random.rand() \
            * (self.best_position - self.position) + self.social_weight \
            * np.random.rand() * (global_best_position - self.position)

    def update_position(self):
        # 位置の更新を実装する
        self.position += self.velocity

    def initial_best_value(self, value):
        self.best_value = value

    def set_value(self, value: Any) -> None:
        self.value = value


class Aggregate:
    def __init__(self, partical_coordinates: np.ndarray[Any, Any], goals: list[str]) -> None:
        self.n_dim = partical_coordinates.shape[1]
        self.goals = goals
        self.global_best_position: ndarray | None = None
        self.global_best_value = None
        self.particles: list[Particle] = []
        for xs in partical_coordinates:
            self.particles.append(Particle(self.n_dim, xs))
        if self.goals[0] == 'minimize':
            for i in range(len(self.particles)):
                self.particles[i].initial_best_value(np.inf)
        else:
            for i in range(len(self.particles)):
                self.particles[i].initial_best_value(-np.inf)

    def get_particle_coordinates(self) -> np.ndarray[Any, Any]:
        return np.array([p.xs for p in self.particles])

    def set_value(self, particle_id: str, value: Any) -> bool:
        for p in self.particles:
            if p.id == particle_id:
                p.set_value(value)
                p.count_eval += 1
                return True
        return False

    def update_global_best_position(self) -> None:
        if self.goals[0] == 'minimize':
            for particle in self.particles:
                if self.global_best_value is None or particle.best_value < self.global_best_value:
                    self.global_best_value = particle.best_value
                    self.global_best_position = particle.best_position
        else:
            for particle in self.particles:
                if self.global_best_value is None or particle.best_value > self.global_best_value:
                    self.global_best_value = particle.best_value
                    self.global_best_position = particle.best_position

    def move(self) -> None:
        self.update_global_best_position()
        for p in self.particles:
            p.update_velocity(self.global_best_position)
            p.update_position()
        return self.particles


class ParticleSwarm:
    def __init__(self, initial_parameters: np.ndarray[Any, Any], goals: list) -> None:
        self.aggregate = Aggregate(initial_parameters, goals)
        self.state = "initialize"
        self.eval_completed_count = 0

    # def initialize(self) -> list[Particle]:
    #     return self.particles

    def change_state(self, state: str) -> None:
        self.state = state

    def get_state(self) -> str:
        return self.state

    def initialize(self) -> list[Particle]:
        return self.aggregate.particles

    def after_initialize(self, yis: list[Value]) -> None:
        for y in yis:
            self.aggregate.set_value(y.id, y.value)
        self.change_state("move")

    def move(self) -> list[Particle]:
        self.aggregate.move()
        return self.aggregate.get_particle_coordinates()

    def after_evaluate(self, yis: list[Value]) -> None:
        for y in yis:
            self.aggregate.set_value(y.id, y.value)
        self.change_state("move")

    def search(self) -> list[Particle]:
        if self.state == "initialize":
            xs = self.initialize()
            self.change_state("initialize_pending")
            return xs

        elif self.state == "initialize_pending":
            return []

        elif self.state == "move":
            xs = self.aggregate.move()
            self.change_state("evaluate_pending")
            return xs

        elif self.state == "evaluate_pending":
            return []

        else:
            raise ValueError(f"{self.state} Unknown state.")


class ParticleSwarmOptimizer(AbstractOptimizer):
    """An optimizer class with particle swarm optimization algorithm.
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
        # self.n_particle = self.config.n_particle
        self.n_particle = 10  # TODO: 仮置き
        self.particle_swarm: Any = None
        self.completed_trial_ids: list[int] = []
        self.single_or_multiple_trial_params: list[Particle] = []
        self.map_trial_id_and_particle_id: dict[int, str] = {}
        if is_multi_objective(self.config):
            raise NotImplementedError("partcle swarm optimizer does not support multi-objective optimization.")

    def convert_ndarray_to_parameter(self, ndarray: np.ndarray[Any, Any]) -> list[dict[str, float | int | str]]:
        """Convert a list of numpy.ndarray to a list of parameters."""
        new_params = copy.deepcopy(self.base_params)
        for name, value, b in zip(self.param_names, ndarray, self.bdrys):
            for new_param in new_params:
                if new_param["name"] == name:
                    new_param["value"] = value
                if b[0] <= value <= b[1]:
                    new_param["out_of_boundary"] = False
                else:
                    new_param["out_of_boundary"] = True
        return new_params

    def _generate_initial_parameter(self, initial_parameters: Any, dim: int, num_of_initials: int) -> Any:
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

    def generate_initial_parameter(self) -> list[Any]:
        initial_parameters = super().generate_initial_parameter()
        initial_parameters = np.array(
            [[self._generate_initial_parameter(initial_parameters, dim, num_of_initials) for dim in range(self.n_params)]
                for num_of_initials in range(self.n_particle)])
        if self.particle_swarm is not None:
            return None
        self.particle_swarm = ParticleSwarm(
            initial_parameters=initial_parameters, goals=self.goals)
        return self.generate_parameter()

    def generate_parameter(self) -> list[dict[str, float | int | str]]:
        searched_params: list[Particle] = self.particle_swarm_main()
        for searched_param in searched_params:
            self.single_or_multiple_trial_params.append(searched_param)
        if len(self.single_or_multiple_trial_params) == 0:
            return None
        new_params: Particle = self.single_or_multiple_trial_params.pop(0)
        self.map_trial_id_and_particle_id[self.trial_id.integer] = new_params.id
        new_param = self.convert_ndarray_to_parameter(new_params.coordinates)
        return new_param

    def new_finished(self) -> list[int]:
        finished = self.storage.get_finished()
        return list(set(finished) ^ set(self.completed_trial_ids))

    def particle_swarm_main(self):
        ps_state = self.particle_swarm.get_state()
        if ps_state in {"initialize_pending", "evaluate_pending"}:
            new_finished = self.new_finished()
            if len(new_finished) == self.n_particle:
                values = []
                for trial_id in new_finished:
                    self.completed_trial_ids.append(trial_id)
                    particle_id = self.map_trial_id_and_particle_id[trial_id]
                    objective = self.storage.result.get_any_trial_objective(trial_id)[0]
                    if self.goals[0] == goal_maximize:
                        objective *= -1.0
                    values.append(Value(id=particle_id, value=objective))
                if ps_state == "initialize_pending":
                    self.particle_swarm.after_initialize(values)
                elif ps_state == "evaluate_pending":
                    self.particle_swarm.after_evaluate(values)
        elif ps_state == "initialize":
            ...
        elif ps_state == "move":
            ...
        else:
            raise NotImplementedError(f"Invalid state: {ps_state}")  # not reachable
        searched_params = self.particle_swarm.search()
        return searched_params
