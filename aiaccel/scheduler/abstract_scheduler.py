from __future__ import annotations

from typing import Any

from omegaconf.dictconfig import DictConfig

from aiaccel.module import AbstractModule
from aiaccel.optimizer import AbstractOptimizer
from aiaccel.scheduler.job.job import Job
from aiaccel.scheduler.job.model.local_model import LocalModel
from aiaccel.util import Buffer, create_yaml, str_to_logging_level


class AbstractScheduler(AbstractModule):
    """An abstract class for AbciScheduler and LocalScheduler.

    Args:
        options (dict[str, str | int | bool]): A dictionary containing
            command line options.

    Attributes:
        options (dict[str, str | int | bool]): A dictionary containing
            command line options.
            to select hyper parameters from a parameter pool.
        available_resource (int): An available current resource number.
        jobs (list[dict]): A list to store job dictionaries.
        max_resource (int): A max resource number.
        stats (list[dict]): A list of current status which is updated using ps
            command or qstat command.
    """

    def __init__(self, config: DictConfig, optimizer: AbstractOptimizer) -> None:
        super().__init__(config, "scheduler")
        self.set_logger(
            "root.scheduler",
            self.workspace.log / self.config.logger.file.scheduler,
            str_to_logging_level(self.config.logger.log_level.scheduler),
            str_to_logging_level(self.config.logger.stream_level.scheduler),
            "Scheduler",
        )
        self.optimizer = optimizer
        self.max_resource = self.config.resource.num_node
        self.trial_number = self.config.optimize.trial_number
        self.num_ready = 0
        self.num_running = 0
        self.num_finished = 0
        self.available_pool_size = 0
        self.available_resource = self.max_resource
        self.stats: list[Any] = []
        self.jobs: list[Any] = []
        self.job_status: dict[Any, Any] = {}
        self.start_trial_id = self.config.resume if self.config.resume is not None else 0
        self.buff = Buffer([trial_id for trial_id in range(self.start_trial_id, self.trial_number)])
        for trial_id in range(self.start_trial_id, self.config.optimize.trial_number):
            self.buff.d[trial_id].set_max_len(2)
        self.all_parameters_generated = False

    def updata_num_ready_running_finished(self) -> None:
        self.num_ready = len(self.storage.trial.get_ready())
        self.num_running = len(self.storage.trial.get_running())
        self.num_finished = len(self.storage.trial.get_finished())
        self.available_pool_size = self.max_resource - self.num_running

    def get_num_ready(self) -> int:
        return self.num_ready

    def get_num_running(self) -> int:
        return self.num_running

    def get_num_finished(self) -> int:
        return self.num_finished

    def get_available_pool_size(self) -> int:
        return self.available_pool_size

    def change_state_finished_trials(self) -> None:
        """Create finished hyper parameter files if result files can be found
        and running files are in running directory.

        Returns:
            None
        """
        runnings = self.storage.trial.get_running()
        result_names = self.storage.result.get_result_trial_id_list()
        if result_names is None:
            return
        for running in runnings:
            if running in result_names:
                self.storage.trial.set_any_trial_state(trial_id=running, state="finished")

    def start_job(self, trial_id: int) -> Any:
        """Start a new job.

        Args:
            hp (Path): A parameter file path

        Returns:
            Job | None: A reference for created job. It returns None if
            specified hyper parameter file already exists.
        """
        trial_ids = [job.trial_id for job in self.jobs]
        if trial_id not in trial_ids:
            job = Job(self.config, self, self.create_model(), trial_id)
            self.jobs.append(job)
            self.logger.debug(f"Submit a job: {str(trial_id)}")
            job.main()
            return job
        else:
            self.logger.error(f"Specified trial {trial_id} is already running ")
            return None

    def update_resource(self) -> None:
        """Update an available current resource number.

        Returns:
            None
        """
        state_names = [
            "Init",
            "RunnerReady",
            "RunnerChecking",
            "RunnerConfirmed",
            "RunnerFailed",
            "RunnerFailure",
            "Scheduling",
        ]

        succeed_jobs = [job for job in self.jobs if job.get_state_name() == "Success"]
        ready_jobs = [job for job in self.jobs if job.get_state_name() in state_names]
        num_running_jobs = len(self.jobs) - len(ready_jobs) - len(succeed_jobs)
        self.available_resource = max(0, self.max_resource - num_running_jobs)

    def pre_process(self) -> None:
        """Pre-procedure before executing processes.

        Returns:
            None
        """
        self.write_random_seed_to_debug_log()
        self.resume()

        self.change_state_finished_trials()

        runnings = self.storage.trial.get_running()
        for running in runnings:
            job = self.start_job(running)
            self.logger.info(f"restart hp files in previous running directory: {running}")

            while job.get_state_name() != "Scheduling":
                job.maie()
            job.schedule()

    def post_process(self) -> None:
        """Post-procedure after executed processes.

        Returns:
            None
        """
        self.logger.info("Scheduler finished.")

    def search_hyperparameters(self, num_ready, num_running, num_finished) -> None:
        """Start hyper parameter optimization.

        Returns:
            None
        """
        sum_status = num_ready + num_running + num_finished
        if sum_status >= self.trial_number and not self.all_parameters_generated:
            self.all_parameters_generated = True
            self.available_pool_size = 0
            self.logger.info(f"trial_number: {self.trial_number}, ready: {num_ready}, running: {num_running}, finished: {num_finished}")
            self.logger.info("All parameters are generated.")
        elif (self.trial_number - sum_status) < self.max_resource:
            self.available_pool_size = self.trial_number - sum_status
        else:
            self.available_pool_size = self.max_resource - num_running - num_ready

        if self.available_pool_size == 0:
            return

        if (
            not self.all_parameters_processed(num_ready, num_running) and
            not self.all_parameters_registered(num_ready, num_running, num_finished)
        ):
            self.optimizer.run_optimizer_multiple_times(self.available_pool_size)

    def inner_loop_main_process(self) -> bool:
        """A main loop process. This process is repeated every main loop.

        Returns:
            bool: The process succeeds or not. The main loop exits if failed.
        """

        self.num_ready, self.num_running, self.num_finished = self.storage.get_num_running_ready_finished()
        self.search_hyperparameters(self.num_ready, self.num_running, self.num_finished)
        if self.num_finished >= self.config.optimize.trial_number:
            return False

        readies = self.storage.trial.get_ready()
        # find a new hp
        for ready in readies:
            if ready not in [job.trial_id for job in self.jobs]:
                self.start_job(ready)

        scheduled_candidates = []
        for job in self.jobs:
            if job.get_state_name() == "Scheduling":
                scheduled_candidates.append(job)

        if len(scheduled_candidates) > 0:
            for job in scheduled_candidates:
                if job.get_state_name() == "Scheduling":
                    self._serialize(job.trial_id)
                    job.schedule()
                    self.logger.debug(f"trial id: {job.trial_id} has been scheduled.")
                    scheduled_candidates.remove(job)

        for job in self.jobs:
            job.main()
            # Only log if the state has changed.
            if job.trial_id in self.buff.d.keys():
                self.buff.d[job.trial_id].Add(job.get_state_name())
                if self.buff.d[job.trial_id].has_difference():
                    self.logger.info(f"name: {job.trial_id}, state: {job.get_state_name()}")

        self.update_resource()

        if self.all_done() is True:
            return False

        return True

    def parse_trial_id(self, command: str) -> str | None:
        """Parse a command string and extract an unique name.

        Args:
            command (str): A command string from ps command.

        Returns:
            str: An unique name.
        """
        pass

    def check_error(self) -> bool:
        # Check state machin
        jobstates = self.storage.jobstate.get_all_trial_jobstate()
        for trial_id in self.job_status.keys():
            for jobstate in jobstates:
                if jobstate["trial_id"] == trial_id and "failure" in jobstate["jobstate"].lower():
                    self.logger.info(
                        f"Job: {trial_id} is Failed.({self.job_status[trial_id]})\n"
                        f"This is a fatal internal error. "
                        f"Please review the configuration file. "
                        f"In particular, we recommend that you "
                        f"review the following items: "
                        f"{'cancel_timeout'}, "
                        f"{'expire_timeout'}, "
                        f"{'finished_timeout'}, "
                        f"{'job_timeout'}, "
                        f"{'kill_timeout'}, "
                        f"{'batch_job_timeout'}, "
                        f"{'runner_timeout'}, "
                        f"{'running_timeout'}"
                    )
                    return False

        # Get error trials
        error_trial_ids = self.storage.error.get_error_trial_id()
        if len(error_trial_ids) > 0:
            return False

        # Get error messages
        for trial_id in error_trial_ids:
            error_message = self.storage.error.get_any_trial_error(trial_id=trial_id)
            self.logger.error(error_message)

        return True

    def all_done(self) -> bool:
        done_states = [
            "Success",
            "HpCancelFailure",
            "KillFailure",
            "HpExpiredFailure",
            "HpFinishedFailure",
            "JobFailure",
            "HpRunningFailure",
            "RunnerFailure",
        ]
    
        jobstates = [s['jobstate'] for s in self.storage.jobstate.get_all_trial_jobstate()]

        num_trials = 0
        for s in done_states:
            num_trials += jobstates.count(s)

        return num_trials >= self.config.optimize.trial_number

    def resume(self) -> None:
        """When in resume mode, load the previous optimization data in advance.

        Args:
            None

        Returns:
            None
        """
        if self.config.resume is not None and self.config.resume > 0:
            self._deserialize(self.config.resume)
            self.trial_number = self.config.optimize.trial_number

    def __getstate__(self) -> dict[str, Any]:
        obj = super().__getstate__()
        del obj["jobs"]
        del obj["optimizer"]
        return obj

    def create_model(self) -> Any:
        """Creates model object of state machine.

        Override with a Scheduler that uses a Model.
        For example, LocalScheduler, AbciScheduler, etc.
        By the way, PylocalScheduler does not use Model.

        Returns:
            LocalModel: LocalModel object.

            Should return None.
            For that purpose, it is necessary to modify TestAbstractScheduler etc significantly.
            So it returns LocalModel.

            # TODO: Fix TestAbstractScheduler etc to return None.
        """
        return LocalModel()

    def evaluate(self) -> None:
        """Evaluate the result of optimization.

        Returns:
            None
        """

        best_trial_ids, _ = self.storage.get_best_trial(self.goals)
        if best_trial_ids is None:
            self.logger.error(f"Failed to output {self.workspace.final_result_file}.")
            return

        hp_results = []

        for best_trial_id in best_trial_ids:
            hp_results.append(self.storage.get_hp_dict(best_trial_id))

        self.logger.info("Best hyperparameter is followings:")
        self.logger.info(hp_results)

        create_yaml(self.workspace.final_result_file, hp_results, self.workspace.lock)

    def all_parameters_processed(self, num_ready, num_running) -> bool:
        """Checks whether any unprocessed parameters are left.

        This method is beneficial for the case that the maximum number of
        parameter generation is limited by algorithm (e.g. grid search).
        To make this method effective, the algorithm with the parameter
        generation limit should turn `all_parameters_generated` True when all
        of available parameters are generated.

        Returns:
            bool: True if all parameters are generated and are processed.
        """
        return num_ready == 0 and num_running == 0 and self.all_parameters_generated

    def all_parameters_registered(self, num_ready, num_running, num_finished) -> bool:
        """Checks whether all parameters that can be generated with the given
        number of trials are registered.

        This method does not check whether the registered parameters have been
        processed.

        Returns:
            bool: True if all parameters are registerd.
        """
        return self.trial_number - num_finished - num_ready - num_running == 0

    def check_finished(self) -> bool:
        """Checks whether all trials are finished.

        Returns:
            bool: True if all trials are finished.
        """
        return self.num_finished >= self.config.optimize.trial_number
