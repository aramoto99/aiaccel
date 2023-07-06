from typing import Any, Dict

from storage import Storage

#
# (WIP) This is a snippet of aiaccel/trial.py
# 

class Objective():
    def __init__(self, storage: Storage, trial_id: int):
        self.storage = storage
        self.trial_id = trial_id

    def set_value(self, value: Any) -> None:
        self.storage.result.set_any_trial_objective(trial_id=self.trial_id, objective=value)

    def get_value(self) -> Any:
        value = self.storage.result.get_any_trial_objective(trial_id=self.trial_id)
        return value


class Parameter():
    def __init__(self, name: str, value: Any):
        self.name = name
        self.value = value

    def get_name(self) -> str:
        return self.name

    def get_value(self) -> Any:
        return self.value


class ParameterList():
    def __init__(self):
        self.params = []

    def add(self, param: Parameter) -> None:
        self.params.append(param)

    def get(self) -> list[Parameter]:
        return self.params

    def get_by_name(self, name: str) -> Parameter:
        for param in self.params:
            if param.get_name() == name:
                return param
        return None


class State():
    def __init__(self, storage: Storage, trial_id: int):
        self.storage = storage
        self.trial_id = trial_id

    def set_state(self, state: str) -> None:
        self.storage.trial.set_any_trial_state(
            trial_id=self.trial_id,
            state=state
        )

    def get_state(self) -> str:
        state = self.storage.trial.get_any_trial_state(trial_id=self.trial_id)
        return state


class Trial():
    def __init__(
        self,
        storage: Storage,
        trial_id: int,
        params: Dict[str, Any],
        out_of_boundary: bool = True,
        objective: Any = None
    ):
        self.storage = storage
        self.trial_id = trial_id

        self.state = State(self.storage, self.trial_id)
        self.params = params
        self.out_of_boundary = out_of_boundary
        self.objective = objective

    def set_objective(self, objective: Any) -> None:
        self.objective = objective

    def change_state(self, state: str) -> None:
        self.state = state

    def get_state(self) -> str:
        return self.state