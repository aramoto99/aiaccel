from __future__ import annotations

from subprocess import PIPE, Popen
from typing import TYPE_CHECKING

from aiaccel.abci import create_abci_batch_file, create_qsub_command
from aiaccel.scheduler.job.model.abstract_model import AbstractModel
from aiaccel.util import OutputHandler

if TYPE_CHECKING:
    from aiaccel.scheduler.job import Job


class AbciModel(AbstractModel):
    def runner_create(self, obj: Job) -> None:
        runner_file_path = obj.workspace.get_runner_file(obj.trial_id)
        create_abci_batch_file(
            trial_id=obj.trial_id,
            param_content=obj.content,
            storage_file_path=obj.workspace.storage_file_path,
            error_file_path=obj.workspace.get_error_output_file(obj.trial_id),
            config_file_path=obj.config.config_path,
            runner_file_path=runner_file_path,
            job_script_preamble=obj.config.ABCI.job_script_preamble,
            command=obj.config.generic.job_command,
            dict_lock=obj.workspace.lock,
        )

    def job_submitted(self, obj: Job) -> None:
        runner_file_path = obj.workspace.get_runner_file(obj.trial_id)
        runner_command = create_qsub_command(obj.config, runner_file_path)

        obj.logger.info(f'runner command: {" ".join(runner_command)}')
        obj.proc = Popen(runner_command, stdout=PIPE, stderr=PIPE)

        obj.th_oh = OutputHandler(obj.proc)
        obj.th_oh.start()
