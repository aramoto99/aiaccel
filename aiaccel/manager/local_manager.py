from __future__ import annotations

from typing import Any

from aiaccel.manager.abstract_manager import AbstractManager
from aiaccel.manager.job.model.local_model import LocalModel


class LocalManager(AbstractManager):
    """A manager class running on a local computer."""

    def create_model(self) -> LocalModel:
        """Creates model object of state machine.

        Returns:
            LocalModel: Model object.
        """
        return LocalModel()

    def get_any_trial_xs(self, trial_id: int) -> dict[str, Any] | None:
        """Gets a parameter list of specific trial ID from Storage object.

        Args:
            trial_id (int): Trial ID.

        Returns:
            dict | None: A dictionary of parameters. None if the parameter
                specified by the given trial ID is not registered.
        """
        params = self.storage.hp.get_any_trial_params(trial_id=trial_id)
        if params is None:
            return {}

        xs = {}
        for param in params:
            xs[param.param_name] = param.param_value

        return xs
