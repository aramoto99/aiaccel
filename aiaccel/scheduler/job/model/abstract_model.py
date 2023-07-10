from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiaccel.scheduler import Job


class AbstractModel(object):
    state: str
    expire: Callable[[Any], Any]
    next: Callable[[Any], Any]

    # ready
    def before_ready(self, obj: Job) -> None:
        obj.logger.debug(f"before_ready: {obj.trial_id}")
        self.runner_create(obj)

    def after_ready(self, obj: Job) -> None:
        obj.logger.debug(f"after_ready: {obj.trial_id}")
        obj.write_start_time_to_storage()
        self.job_submitted(obj)

    def runner_create(self, obj: Job) -> None:
        ...

    def job_submitted(self, obj: Job) -> None:
        ...

    # running
    def before_running(self, obj: Job) -> None:
        obj.logger.debug(f"before_running: {obj.trial_id}")
        obj.write_state_to_storage("running")

    def after_running(self, obj: Job) -> None:
        ...

    def conditions_job_finished(self, obj: Job) -> bool:
        objective = obj.storage.result.get_any_trial_objective(
            trial_id=obj.trial_id)
        return (objective is not None)

    # finished
    def before_finished(self, obj: Job) -> None:
        obj.logger.debug(f"before_finished: {obj.trial_id}")
        obj.write_state_to_storage("finished")
        obj.write_end_time_to_storage()

    def after_finished(self, obj: Job) -> None:
        obj.logger.debug(f"after_finished: {obj.trial_id}")
        obj.write_job_success_or_failed_to_storage()

    ...
