from __future__ import annotations

import os
import shutil
import time
from argparse import ArgumentParser
from logging import StreamHandler, getLogger
from pathlib import Path
from typing import Any

import yaml
from omegaconf.dictconfig import DictConfig

from aiaccel.cli import CsvWriter
from aiaccel.common import resource_type_mpi
from aiaccel.config import load_config
from aiaccel.manager import create_manager
from aiaccel.optimizer import create_optimizer
from aiaccel.storage import Storage
from aiaccel.tensorboard import TensorBoard
from aiaccel.util.buffer import Buffer
from aiaccel.util.mpi import mpi_enable
from aiaccel.workspace import Workspace

logger = getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
logger.addHandler(StreamHandler())


if mpi_enable:
    from aiaccel.util.mpi import Mpi


def main() -> None:  # pragma: no cover
    """Parses command line options and executes optimization."""
    parser = ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="config.yaml")
    parser.add_argument("--resume", type=int, default=None)
    parser.add_argument("--clean", nargs="?", const=True, default=False)

    parser.add_argument("--from_mpi_bat", action="store_true", help="Only aiaccel is used when mpi bat.")
    parser.add_argument("--make_hostfile", action="store_true", help="Only aiaccel is used when mpi bat.")
    args = parser.parse_args()

    config: DictConfig = load_config(args.config)
    if config.resource.type.value.lower() == resource_type_mpi:  # MPI
        if not mpi_enable:
            raise Exception("MPI is not enabled.")
        if args.make_hostfile:
            Mpi.make_hostfile(config, logger)
            return
        if not args.from_mpi_bat:
            Mpi.run_bat(config, logger)
            return
        logger.info("MPI is enabled.")
        if Mpi.gpu_max == 0:
            Mpi.gpu_max = config.resource.mpi_npernode
        Mpi.run_main()

    config.resume = args.resume
    config.clean = args.clean
    workspace = Workspace(config.generic.workspace)
    if config.resume is None:
        if config.clean is True:
            logger.info("Cleaning workspace")
            workspace.clean()
            logger.info(f"Workspace directory {str(workspace.path)} is cleaned.")
        else:
            if workspace.exists():
                logger.info("workspace exists.")
                return
    workspace.create()
    if workspace.check_consists() is False:
        logger.error("Creating workspace is Failed.")
        return

    logger.info(f"config: {config.config_path}")

    optimizer = create_optimizer(config.optimize.search_algorithm)(config)
    manager = create_manager(config.resource.type.value)(config, optimizer)
    tensorboard = TensorBoard(config)
    storage = Storage(workspace.storage_file_path)

    time_s = time.time()
    max_trial_number = config.optimize.trial_number
    buff = Buffer(["num_finished"])
    buff.d["num_finished"].set_max_len(2)

    manager.pre_process()

    if config.resource.type.value.lower() == resource_type_mpi:  # MPI
        Mpi.prepare(workspace.path)

    while True:
        try:
            if not manager.run_in_main_loop():
                break
            if not manager.is_error_free():
                break
            if int((time.time() - time_s)) % 10 == 0:
                num_ready, num_running, num_finished = storage.get_num_running_ready_finished()
                buff.d["num_finished"].add(num_finished)
                if buff.d["num_finished"].length == 1 or buff.d["num_finished"].has_difference():
                    manager.logger.info(
                        f"{num_finished}/{max_trial_number} finished, "
                        f"max trial number: {max_trial_number}, "
                        f"ready: {num_ready} ,"
                        f"running: {num_running}"
                    )
                    # TensorBoard
                    tensorboard.update()
            time.sleep(config.generic.main_loop_sleep_seconds)
        except Exception as e:
            logger.exception("Unexpected error occurred.")
            logger.exception(e)
            break

    manager.post_process()
    manager.evaluate()

    best_result = manager.get_best_result()
    print("Best hyperparameter is followings:")
    print(best_result)

    csv_writer = CsvWriter(config)
    csv_writer.create()

    print("moving...")
    dst = workspace.move_completed_data()
    if dst is None:
        print("Moving data is failed.")
        return

    config_name = Path(args.config).name
    shutil.copy(Path(args.config), dst / config_name)

    if os.path.exists(workspace.best_result_file):
        with open(workspace.best_result_file, "r") as f:
            final_results: list[dict[str, Any]] = yaml.load(f, Loader=yaml.UnsafeLoader)

        for i, final_result in enumerate(final_results):
            best_id = final_result["trial_id"]
            best_value = final_result["result"][i]
            if best_id is not None and best_value is not None:
                print(f"Best trial [{i}] : {best_id}")
                print(f"\tvalue : {best_value}")
    print(f"result file : {dst}/{'results.csv'}")
    print(f"Total time [s] : {round(time.time() - time_s)}")
    print("Done.")
    return


if __name__ == "__main__":  # pragma: no cover
    main()
