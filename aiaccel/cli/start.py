from __future__ import annotations

import os
import pathlib
import shutil
import time
from argparse import ArgumentParser
from logging import StreamHandler, getLogger
from pathlib import Path
from typing import Any

import yaml

from aiaccel.cli import CsvWriter
from aiaccel.common import dict_result, extension_hp
from aiaccel.config import Config, load_config
from aiaccel.module import AiaccelCore
from aiaccel.optimizer import create_optimizer
from aiaccel.scheduler import create_scheduler
from aiaccel.storage import Storage
from aiaccel.tensorboard import TensorBoard
from aiaccel.util.buffer import Buffer
from aiaccel.util.time_tools import get_time_now_object, get_time_string_from_object
from aiaccel.workspace import Workspace

logger = getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
logger.addHandler(StreamHandler())


def main() -> None:  # pragma: no cover
    """Parses command line options and executes optimization."""
    parser = ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="config.yml")
    parser.add_argument("--resume", type=int, default=None)
    parser.add_argument("--clean", nargs="?", const=True, default=False)
    args = parser.parse_args()

    config: Config = load_config(args.config)
    if config is None:
        logger.error(f"Invalid workspace: {args.workspace} or config: {args.config}")
        return
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

    logger.info(f"config: {str(pathlib.Path(config.config_path).resolve())}")

    # storage
    storage = Storage(workspace.storage_file_path)
    if config.resume is not None:
        storage.rollback_to_ready(config.resume)
        storage.delete_trial_data_after_this(config.resume)

    # optimizer
    Optimizer = create_optimizer(config.optimize.search_algorithm)
    optimizer = Optimizer(config)
    if config.resume is not None:
        optimizer.resume()

    # scheduler
    Scheduler = create_scheduler(config.resource.type.value)
    scheduler = Scheduler(config, optimizer)

    # tensorboard
    tensorboard = TensorBoard(config)

    modules: list[AiaccelCore] = [scheduler]

    time_s = time.time()
    max_trial_number = config.optimize.trial_number
    loop_start_time = get_time_now_object()
    end_estimated_time = "Unknown"
    buff = Buffer(['num_finished', 'available_pool_size'])
    buff.d['num_finished'].set_max_len(2)
    buff.d['available_pool_size'].set_max_len(2)

    # main process
    for module in modules:
        module.pre_process()

    while True:
        try:
            for module in modules:
                if not module.inner_loop_main_process():
                    break
                if not module.check_error():
                    break
            else:
                nun_ready = modules[0].get_num_ready()
                num_running = modules[0].get_num_running()
                num_finished = modules[0].get_num_finished()
                available_pool_size = modules[0].get_available_pool_size()
                now = get_time_now_object()
                looping_time = now - loop_start_time

                if num_finished > 0:
                    one_loop_time = looping_time / num_finished
                    finishing_time = now + (max_trial_number - num_finished) * one_loop_time
                    end_estimated_time = get_time_string_from_object(finishing_time)

                if (
                    int((time.time() - time_s)) % 10 == 0 or
                    num_finished >= max_trial_number
                ):
                    buff.d['num_finished'].Add(num_finished)
                    if (
                        buff.d['num_finished'].Len == 1 or
                        buff.d['num_finished'].has_difference()
                    ):
                        modules[0].logger.info(
                            f"{num_finished}/{max_trial_number} finished, "
                            f"max trial number: {max_trial_number}, "
                            f"ready: {nun_ready} ,"
                            f"running: {num_running}, "
                            f"end estimated time: {end_estimated_time}"
                        )
                        # TensorBoard
                        tensorboard.update()

                    buff.d['available_pool_size'].Add(available_pool_size)
                    if (
                        buff.d['available_pool_size'].Len == 1 or
                        buff.d['available_pool_size'].has_difference()
                    ):
                        modules[0].logger.info(f"pool_size: {available_pool_size}")

                else:
                    buff.d['num_finished'].Clear()
                    buff.d['available_pool_size'].Clear()

                time.sleep(config.generic.sleep_time)
                continue
            break
        except Exception as e:
            logger.exception("Unexpected error occurred.")
            logger.exception(e)
            break

    for module in modules:
        module.post_process()

    scheduler.evaluate()

    csv_writer = CsvWriter(config)
    csv_writer.create()

    logger.info("moving...")
    dst = workspace.move_completed_data()
    if dst is None:
        logger.error("Moving data is failed.")
        return

    config_name = Path(args.config).name
    shutil.copy(Path(args.config), dst / config_name)

    with open(workspace.final_result_file, "r") as f:
        final_results: list[dict[str, Any]] = yaml.load(f, Loader=yaml.UnsafeLoader)

    for i, final_result in enumerate(final_results):
        best_id = final_result["trial_id"]
        best_value = final_result["result"][i]
        if best_id is not None and best_value is not None:
            logger.info(f"Best result [{i}] : {dst}/{dict_result}/{best_id}.{extension_hp}")
            logger.info(f"\tvalue : {best_value}")

    logger.info(f"Total time [s] : {round(time.time() - time_s)}")
    logger.info("Done.")
    return


if __name__ == "__main__":  # pragma: no cover
    main()
