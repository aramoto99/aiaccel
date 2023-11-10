from __future__ import annotations

from aiaccel.manager.abstract_manager import AbstractManager
from aiaccel.manager.job.model.abci_model import AbciModel


class AbciManager(AbstractManager):
    """A manager class running on ABCI environment."""

    def create_model(self) -> AbciModel:
        """Creates model object of state machine.

        Returns:
            AbciModel: Model object.
        """
        return AbciModel()
