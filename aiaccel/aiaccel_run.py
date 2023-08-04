from __future__ import annotations

import os
import shutil
import logging
import sys
import traceback
from argparse import ArgumentParser
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from aiaccel.common import datetime_format
from aiaccel.config import load_config
from aiaccel.parameter import (CategoricalParameter, FloatParameter, HyperParameterConfiguration, IntParameter,
                               OrdinalParameter)
from aiaccel.util.data_type import str_or_float_or_int
from aiaccel.optimizer import create_optimizer
from aiaccel.storage import Storage
from aiaccel.workspace import Workspace
from aiaccel.cli.set_result import write_results_to_database
from aiaccel.common import datetime_format
from aiaccel.cli import CsvWriter
from aiaccel.util import create_yaml


def set_logging_file_for_trial_id(workspace: Path, trial_id: int) -> None:
    log_dir = workspace / "log"
    log_path = log_dir / f"job_{trial_id}.log"
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    logging.basicConfig(filename=log_path, level=logging.DEBUG, force=True)


def cast_y(y_value: Any, y_data_type: str | None) -> float | int | str:
    """Casts y to the appropriate data type.

    Args:
        y_value (Any): y value to be casted.
        y_data_type (str | None): Name of data type of objective value.

    Returns:
        float | int | str: Casted y value.

    Raises:
        TypeError: Occurs when given `y_data_type` is other than `float`,
                `int`, or `str`.
    """
    if y_data_type is None:
        y = y_value
    elif y_data_type.lower() == "float":
        y = float(y_value)
    elif y_data_type.lower() == "int":
        y = int(float(y_value))
    elif y_data_type.lower() == "str":
        y = str(y_value)
    else:
        TypeError(f"{y_data_type} cannot be specified")

    return y


class CommandLineArgs:
    def __init__(self) -> None:
        self.parser = ArgumentParser()
        self.parser.add_argument("--trial_id", type=int, required=False)
        self.parser.add_argument("--config", type=str, required=False)
        self.args = self.parser.parse_known_args()[0]
        self.trial_id = None
        self.config_path = None
        self.config = None
        if self.args.trial_id is not None:
            self.trial_id = self.args.trial_id
        if self.args.config is not None:
            self.config_path = Path(self.args.config).resolve()
            self.config = load_config(self.config_path)
            self.parameters_config = HyperParameterConfiguration(self.config.optimize.parameters)
            for p in self.parameters_config.get_parameter_list():
                if isinstance(p, FloatParameter):
                    self.parser.add_argument(f"--{p.name}", type=float)
                elif isinstance(p, IntParameter):
                    self.parser.add_argument(f"--{p.name}", type=int)
                elif isinstance(p, CategoricalParameter):
                    self.parser.add_argument(f"--{p.name}", type=str_or_float_or_int)
                elif isinstance(p, OrdinalParameter):
                    self.parser.add_argument(f"--{p.name}", type=str_or_float_or_int)
                else:
                    raise ValueError(f"Unknown parameter type: {p.type}")
            self.args = self.parser.parse_known_args()[0]
        else:
            unknown_args_list = self.parser.parse_known_args()[1]
            for unknown_arg in unknown_args_list:
                if unknown_arg.startswith("--"):
                    name = unknown_arg.replace("--", "")
                    self.parser.add_argument(f"--{name}", type=str_or_float_or_int)
            self.args = self.parser.parse_known_args()[0]

    def get_xs_from_args(self) -> dict[str, Any]:
        xs = vars(self.args)
        delete_keys = ["trial_id", "config"]
        for key in delete_keys:
            if key in xs.keys():
                del xs[key]
        return xs


class Run:
    """An Interface between user program or python function object.

    Args:
        config_path (str | Path | None, optional): A path to configration file.
            Defaults to None.

    Attributes:
        args (dict): A dictionary object which contains command line arguments
            given by aiaccel.
        trial_id (int): Trial Id.
        config_path (Path): A Path object which points to the
            configuration file.
        config (Config): A Config object.
        workspace (Path): A Path object which points to the workspace.

    Examples:
        *User program* ::

            from aiaccel.util import aiaccel

            run = aiaccel.Run()
            run.execute_and_report("execute user_program")

        Note that `execute user_program` is a command to execute a user
        program.
        See :doc:`../examples/wrapper_sample`.

        *Python function* ::

            from aiaccel.util import aiaccel

            def func(p: dict[str, Any]) -> float:
                # Write your operation to calculate objective value.

                return objective_y

            if __name__ == "__main__":
                run = aiaccel.Run()
                run.execute_and_report(func)
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config_path = None
        self.config = None
        self.workspace = None

        self.args = CommandLineArgs()
        self.config_path = self.args.config_path or config_path
        self.config = self.args.config
        if self.config is not None:
            self.workspace = Path(self.config.generic.workspace).resolve()

    def execute(
        self,
        func: Callable[[dict[str, float | int | str]], float],
        xs: "dict[str, float | int | str]",
        y_data_type: "str | None",
    ) -> Any:
        """Executes the target function.

        Args:
            func (Callable[[dict[str, float | int | str]], float]):
                User-defined python function.
            trial_id (int): Trial ID.
            y_data_type (str | None): Name of data type of objective value.

        Returns:
            tuple[dict[str, float | int | str] | None, float | int | str | None, str]:
                A dictionary of parameters, a casted objective value, and error
                string.
        """
        if self.workspace is not None and self.args.trial_id is not None:
            set_logging_file_for_trial_id(self.workspace, self.args.trial_id)

        y = None
        err = ""

        try:
            y = cast_y(func(xs), y_data_type)
        except BaseException:
            err = str(traceback.format_exc())
            y = None
        else:
            err = ""
        return xs, y, err

    def execute_and_report(
        self, func: Callable[[dict[str, float | int | str]], float], y_data_type: str | None = None
    ) -> None:
        """Executes the target function and report the results.

        Args:
            func (Callable[[dict[str, float | int | str]], float]):
                User-defined python function.
            y_data_type (str | None, optional): Name of data type of
                objective value. Defaults to None.

        Examples:
         ::

            from aiaccel.util import aiaccel

            def func(p: dict[str, Any]) -> float:
                # Write your operation to calculate objective value.

                return objective_y

            if __name__ == "__main__":
                run = aiaccel.Run()
                run.execute_and_report(func)
        """

        xs = self.args.get_xs_from_args()
        y: Any = None
        _, y, err = self.execute(func, xs, y_data_type)

        self.report(y, err)

    def report(self, y: Any, err: str) -> None:
        """Save the results to a text file.

        Args:
            y (Any): Objective value.
            err (str): Error string.
        """

        if y is not None:
            sys.stdout.write(f"{y}\n")
        if err != "":
            sys.stderr.write(f"{err}\n")
            exit(1)


class Run2(Run):
    def __init__(self):
        super().__init__()
        self.optimizer = None
        self.storage = None

    def load_config(self, config_path: str | Path, workspace_clean: bool = True) -> None:
        if self.config_path is not None:
            logging.warning(
                "If the configuration file path is specified as an optional argument, \
                it will take priority over any other settings."
            )
        else:
            self.config_path = Path(config_path).resolve()
        self.config = load_config(self.config_path)
        self.workspace = Workspace(self.config.generic.workspace)
        if workspace_clean:
            self.workspace.clean()
        self.workspace.create()

    def get_resume_trial_id(self) -> int | None:
        resume_trial_id: int | None = self.storage.state.get_first_trial_id_for_resume()
        num_ready, num_running, num_finished = self.storage.get_num_running_ready_finished()
        if num_finished == 0 or resume_trial_id is None: return
        if (
            resume_trial_id == self.config.optimize.trial_number or
            num_finished == self.config.optimize.trial_number
        ):
            raise ValueError(
                f"Trial number {resune_trial_id} is already finished. \
                Please remove work irectory: {self.self.workspace.path}."
            )
        if resume_trial_id == 0:
            return num_finished - 1
        else:
            return resune_trial_id

    def resume(self, trial_id: int) -> None:
        self.storage.rollback_to_ready(trial_id)
        self.storage.delete_trial_data_after_this(trial_id)
        self.optimizer.resume()

    def load_optimizer(self) -> None:
        resume_trial_id = self.get_resume_trial_id()
        self.config.resume = resume_trial_id
        self.optimizer = create_optimizer(self.config.optimize.search_algorithm)(self.config)
        self.storage = Storage(self.workspace.storage_file_path)
        if isinstance(resume_trial_id, int) and resume_trial_id > 0:
            self.resume(resume_trial_id)

    def optimize(self) -> None:
        if self.storage is None:
            self.storage = Storage(self.workspace.storage_file_path)
        if self.optimizer is None:
            self.load_optimizer()
        self.optimizer.run_optimizer()

    def execute_and_report(
        self,
        trial_id: int,
        func: Callable[[dict[str, float | int | str]], float],
        y_data_type: str | None = None
    ) -> None:
        if self.storage.state.get_any_trial_state(trial_id) == "finished":
            logging.warning(f"Trial ID {trial_id} is already finished.")
            return
        start_time = datetime.now().strftime(datetime_format)
        xs = self.storage.hp.get_any_trial_params_dict(trial_id)
        y: Any = None
        self.storage.state.set_any_trial_state(trial_id=trial_id, state="running")
        _, y, err = self.execute(func, xs, y_data_type)
        end_time = datetime.now().strftime(datetime_format)

        if isinstance(y, list):
            ys = [yi for yi in y]
        else:
            ys = [y]

        self.report(trial_id, ys, err, start_time, end_time)

    def report(self, trial_id: int, ys: list[Any], err: str, start_time: str, end_time: str) -> None:
        """Saves results in the Storage object.

        Args:
            trial_id (int): Trial ID.
            xs (dict): A dictionary of parameters.
            y (Any): Objective value.
            err (str): Error string.
            start_time (str): Execution start time.
            end_time (str): Execution end time.
        """

        write_results_to_database(
            storage_file_path=self.workspace.storage_file_path,
            trial_id=trial_id,
            objective=ys,
            start_time=start_time,
            end_time=end_time,
            error=err,
            returncode=None
        )
        self.storage.state.set_any_trial_state(trial_id=trial_id, state="finished")

    def get_total_seconds(self):
        time_format = "%m/%d/%Y %H:%M:%S"
        start_time = self.storage.timestamp.get_any_trial_start_time(trial_id=0)
        if start_time is not None:
            start_time = datetime.strptime(start_time, time_format)
        end_time = self.storage.timestamp.get_any_trial_end_time(trial_id=self.config.optimize.trial_number - 1)
        if end_time is not None:
            end_time = datetime.strptime(end_time, time_format)
        total_seconds = None
        if start_time is not None and end_time is not None:
            total_seconds = (end_time - start_time).total_seconds()
        return total_seconds

    def evaluate(self) -> None:
        """Evaluate the result of optimization.

        Returns:
            None
        """

        goals = [item.value for item in self.config.optimize.goal]
        best_trial_ids, _ = self.storage.get_best_trial(goals)
        if best_trial_ids is None:
            self.logger.error(f"Failed to output {self.workspace.final_result_file}.")
            return
        hp_results = []
        for best_trial_id in best_trial_ids:
            hp_results.append(self.storage.get_hp_dict(best_trial_id))
        create_yaml(self.workspace.final_result_file, hp_results, self.workspace.lock)

    def on_all_finish_processing(self):

        total_seconds = self.get_total_seconds()

        csv_writer = CsvWriter(self.config)
        csv_writer.create()

        print("moving...")
        dst = self.workspace.move_completed_data()
        if dst is None:
            print("Moving data is failed.")
            return

        shutil.copy(Path(self.config_path), dst / self.config_path.name)

        if os.path.exists(self.workspace.final_result_file):
            with open(self.workspace.final_result_file, "r") as f:
                final_results: list[dict[str, Any]] = yaml.load(f, Loader=yaml.UnsafeLoader)

            for i, final_result in enumerate(final_results):
                best_id = final_result["trial_id"]
                best_value = final_result["result"][i]
                if best_id is not None and best_value is not None:
                    print(f"Best trial [{i}] : {best_id}")
                    print(f"\tvalue : {best_value}")
        print(f"result file : {dst}/{'results.csv'}")
        print(f"Total time [s] : {round(total_seconds)}")
        print("Done.")

    @contextmanager
    def train(self):
        yield
        self.evaluate()
        self.on_all_finish_processing()
