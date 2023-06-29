from __future__ import annotations

import re
from pathlib import Path
from subprocess import PIPE, Popen
from typing import TYPE_CHECKING, Any

from aiaccel.abci import create_qsub_command
from aiaccel.scheduler.job.model.abstract_model import AbstractModel
from aiaccel.util import OutputHandler, file_create

if TYPE_CHECKING:
    from aiaccel.scheduler.job import Job


class AbciModel(AbstractModel):
    def runner_create(self, obj: Job) -> None:
        runner_file_path = obj.workspace.get_runner_file(obj.trial_id)
        self.create_abci_batch_file(
            trial_id=obj.trial_id,
            param_content=obj.content,
            storage_file_path=obj.workspace.storage_file_path,
            error_file_path=obj.workspace.get_error_output_file(obj.trial_id),
            config_file_path=obj.config.config_path,
            runner_file_path=runner_file_path,
            job_script_preamble=obj.config.ABCI.job_script_preamble,
            command=obj.config.generic.job_command,
            enable_command_argument=obj.config.generic.enable_command_argument,
            dict_lock=obj.workspace.lock,
        )

    def job_submitted(self, obj: Job) -> None:
        runner_file_path = obj.workspace.get_runner_file(obj.trial_id)
        runner_command = create_qsub_command(obj.config, runner_file_path)

        obj.logger.info(f'runner command: {" ".join(runner_command)}')
        obj.proc = Popen(runner_command, stdout=PIPE, stderr=PIPE)

        obj.th_oh = OutputHandler(obj.proc)
        obj.th_oh.start()

    def generate_command_line(self, command: str, args: list[str]) -> str:
        return f'{command} {" ".join(args)}'

    def generate_param_args(self, params: list[dict[str, Any]]) -> str:
        param_args = ""
        for param in params:
            if "name" in param.keys() and "value" in param.keys():
                param_args += f'--{param["name"]}=${param["name"]} '
        return param_args

    def create_abci_batch_file(
        self,
        trial_id: int,
        param_content: dict[str, Any],
        storage_file_path: Path | str,
        error_file_path: Path | str,
        config_file_path: Path | str,
        runner_file_path: Path,
        job_script_preamble: Path | str | None,
        command: str,
        enable_command_argument: bool,
        dict_lock: Path,
    ) -> None:
        """Create a ABCI batch file.

        The 'job_script_preamble' is a base of the ABCI batch file. At first, loads
        'job_script_preamble', and adds the 'commands' to the loaded contents. Finally,
        writes the contents to 'runner_file_path'.

        Args:
            trial_id (int): A trial id.
            param_content (dict): A dictionary of parameters.
            storage_file_path (Path | str): A path of a db file.
            error_file_path (Path | str): A path of a error file.
            config_file_path (Path | str): A path of a config file.
            runner_file_path (Path): A path of a creating file.
            job_script_preamble (str): A wrapper file of ABCI batch file.
            command (str): A command to execute.
            dict_lock (Path): A directory to store lock files.

        Returns:
            None
        """

        commands = re.split(" +", command)
        if enable_command_argument:
            for param in param_content["parameters"]:
                if "name" in param.keys() and "value" in param.keys():
                    commands.append(f'--{param["name"]}=${param["name"]}')
            commands.append(f"--trial_id={str(trial_id)}")
            commands.append("--config=$config_file_path")
        else:
            for param in param_content["parameters"]:
                if "name" in param.keys() and "value" in param.keys():
                    commands.append(f'${param["name"]}')
            commands.append(str(trial_id))
            commands.append("$config_file_path")
        commands.append("2>")
        commands.append("$error_file_path")

        set_retult = self.generate_command_line(
            command="aiaccel-set-result",
            args=[
                "--storage_file_path=$storage_file_path",
                "--trial_id=$trial_id",
                "--config=$config_file_path",
                "--objective=$ys",
                "--error=$error",
                "--returncode=$returncode",
                self.generate_param_args(param_content["parameters"]),
            ],
        )

        set_retult_no_error = self.generate_command_line(
            command="aiaccel-set-result",
            args=[
                "--storage_file_path=$storage_file_path",
                "--trial_id=$trial_id",
                "--config=$config_file_path",
                "--objective=$ys",
                "--returncode=$returncode",
                self.generate_param_args(param_content["parameters"]),
            ],
        )

        main_parts = [
            f"trial_id={str(trial_id)}",
            f"config_file_path={str(config_file_path)}",
            f"storage_file_path={str(storage_file_path)}",
            f"error_file_path={str(error_file_path)}",
            # 'start_time=`date "+%Y-%m-%d %H:%M:%S"`',
            f'result=`{" ".join(commands)}`',
            "returncode=$?",
            'ys=$(echo $result | tr -d "[],")',
            "error=`cat $error_file_path`",
            # 'end_time=`date "+%Y-%m-%d %H:%M:%S"`',
            'if [ -n "$error" ]; then',
            "\t" + set_retult,
            "else",
            "\t" + set_retult_no_error,
            "fi",
        ]

        script = ""
        # preamble
        if job_script_preamble is not None:
            with open(job_script_preamble, "r") as f:
                lines = f.read().splitlines()
                if len(lines) > 0:
                    for line in lines:
                        script += line + "\n"
        script += "\n"
        # parameters
        for param in param_content["parameters"]:
            if "name" in param.keys() and "value" in param.keys():
                script += f'{param["name"]}={param["value"]}' + "\n"
        script += "\n"
        # main
        for s in main_parts:
            script += s + "\n"

        file_create(runner_file_path, script, dict_lock)
