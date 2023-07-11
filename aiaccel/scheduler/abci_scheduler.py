from __future__ import annotations

from aiaccel.scheduler.abstract_scheduler import AbstractScheduler
from aiaccel.scheduler.job.model.abci_model import AbciModel


class AbciScheduler(AbstractScheduler):
    """A scheduler class running on ABCI environment."""

    def create_model(self) -> AbciModel:
        """Creates model object of state machine.

        Returns:
            AbciModel: Model object.
        """
        return AbciModel()
