from __future__ import annotations

import os
import shutil
import logging
import sys
import threading
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from aiaccel.common import datetime_format
from aiaccel.config import load_config
from aiaccel.optimizer import create_optimizer
from aiaccel.storage import Storage
from aiaccel.workspace import Workspace
from aiaccel.cli.set_result import write_results_to_database
from aiaccel.cli import CsvWriter
from aiaccel.util import cast_y, create_yaml


class Study():
    """An Interface between user program or python function object.

    Example:

        import aiaccel

        def main(p):
            y = (p["x1"]**2) - (4.0 * p["x1"]) + (p["x2"]**2) - p["x2"] - (p["x1"] * p["x2"])
            return float(y)

        if __name__ == "__main__":
            run = aiaccel.Run2("./config.yaml")
            run.optimize(main, n_trials=1)
    """
    def __init__(self, config_path: str | Path) -> None:
        self.config = None
        self.workspace = None
        self.optimizer = None
        self.storage = None
        self.config_path = Path(config_path).resolve()
        self.load_config(config_path)
        self.load_optimizer()

    @property
    def trial_number(self) -> int:
        if self.config is None:
            raise ValueError("config is None")
        return self.config.optimize.trial_number

    @property
    def num_workers(self) -> int:
        if self.config is None:
            raise ValueError("config is None")
        return self.config.resource.num_workers

    @property
    def best_value(self) -> float | int:
        self.optimizer.get_best_value()

    def load_config(self, config_path: str | Path) -> None:
        self.config = load_config(self.config_path)
        self.workspace = Workspace(self.config.generic.workspace)
        self.workspace.create()

    def load_optimizer(self) -> None:
        if self.config is None:
            raise ValueError("config is None")
        if self.optimizer is not None:
            return
        self.optimizer = create_optimizer(self.config.optimize.search_algorithm)(self.config)
        self.storage = Storage(self.workspace.storage_file_path)

    def get_resume_trial_id(self) -> int | None:
        resume_trial_id: int | None = self.storage.state.get_first_trial_id_for_resume()
        num_ready, num_running, num_finished = self.storage.get_num_running_ready_finished()
        if num_finished == 0 or resume_trial_id is None: return
        if (resume_trial_id == self.trial_number or num_finished == self.trial_number):
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

    def optimize(
        self,
        func: Callable[[dict[str, float | int | str]], float],
        n_trials: int | None = None,
    ) -> None:
        if self.storage is None:
            self.storage = Storage(self.workspace.storage_file_path)
        if self.optimizer is None:
            self.load_optimizer()

        n_count = 0
        while True:
            trial_id = self.optimizer.get_trial_id()
            if n_count == 0 and trial_id > 0:
                self.config.resume = self.optimizer.get_trial_id()
                self.optimizer.resume()
            self.optimizer.run_optimizer()
            self.execute_and_report(trial_id, func)
            n_count += 1
            if n_trials is None:
                if trial_id >= self.trial_number - 1:
                    break
            else:
                if n_count >= n_trials:
                    break

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
        y = None; err = ""
        try:
            y = cast_y(func(xs), y_data_type)
        except BaseException:
            err = str(traceback.format_exc())
            y = None
        else:
            err = ""
        return xs, y, err

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

    def execute_and_report(
        self,
        trial_id: int,
        func: Callable[[dict[str, float | int | str]], float],
        y_data_type: str | None = None
    ) -> None:
        if self.storage.state.get_any_trial_state(trial_id) == "finished":
            self.optimizer.logger.warning(f"Trial ID {trial_id} is already finished.")
            return
        start_time = datetime.now().strftime(datetime_format)
        xs = self.storage.hp.get_any_trial_params_dict(trial_id)
        y: Any = None
        self.storage.state.set_any_trial_state(trial_id=trial_id, state="running")
        _, y, err = self.execute(func, xs, y_data_type)
        self.storage.state.set_any_trial_state(trial_id=trial_id, state="finished")
        end_time = datetime.now().strftime(datetime_format)
        if isinstance(y, list):
            ys = [yi for yi in y]
        else:
            ys = [y]
        self.report(trial_id, ys, err, start_time, end_time)

    def get_total_seconds(self) -> float | None:
        """Get the total seconds of optimization.
        """
        time_format = "%m/%d/%Y %H:%M:%S"
        start_time = self.storage.timestamp.get_any_trial_start_time(trial_id=0)
        if start_time is not None:
            start_time = datetime.strptime(start_time, time_format)
        end_time = self.storage.timestamp.get_any_trial_end_time(trial_id=self.storage.current_max_trial_number())
        if end_time is not None:
            end_time = datetime.strptime(end_time, time_format)
        total_seconds = None
        if start_time is not None and end_time is not None:
            total_seconds = (end_time - start_time).total_seconds()
        return total_seconds

    def evaluate(self) -> None:
        """Evaluate the result of optimization.
        """
        goals = [item.value for item in self.config.optimize.goal]
        best_trial_ids, _ = self.storage.get_best_trial(goals)
        if best_trial_ids is None:
            logging.error(f"Failed to output {self.workspace.final_result_file}.")
            return
        hp_results = []
        for best_trial_id in best_trial_ids:
            hp_results.append(self.storage.get_hp_dict(best_trial_id))
        create_yaml(self.workspace.final_result_file, hp_results, self.workspace.lock)

    def show_result(self):
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
        print("For more details, execute the following command.")
        print(f"aiaccel-view --config {self.config_path}")


def create_study(config_path: str | Path) -> Study:
    if not Path(config_path).exists():
        raise FileNotFoundError(f"{config_path} is not found.")
    study = Study(config_path)
    return study
