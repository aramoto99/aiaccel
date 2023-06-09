from typing import Any, Dict


class Trial():
    def __init__(self, trial_id: int, params: Dict[str, Any], out_of_boundary: bool, objective: Any):
        self.trial_id = trial_id
        self.params = params
        self.out_of_boundary = out_of_boundary
        self.objective = objective