from __future__ import annotations

import copy
from argparse import ArgumentParser
from pathlib import Path

from aiaccel.storage.storage import Storage
from aiaccel.util.data_type import str_or_float_or_int


def write_results_to_database(
    storage_file_path: str | Path,
    trial_id: int,
    objective: float | None,
    start_time: str,
    end_time: str,
    error: str,
    returncode: int | None,
) -> None:

    storage = Storage(storage_file_path)

    if objective is None:
        raise Exception("Could not get objective")
    storage.result.set_any_trial_objective(trial_id, objective)

    storage.timestamp.set_any_trial_start_time_and_end_time(trial_id, start_time, end_time)
    # if start_time is not None:
    #     storage.timestamp.set_any_trial_start_time(trial_id, start_time)
    # if end_time is not None:
    #     storage.timestamp.set_any_trial_end_time(trial_id, end_time)
    if returncode is not None:
        storage.returncode.set_any_trial_returncode(trial_id, returncode)
    if error != "":
        storage.error.set_any_trial_error(trial_id, error)


def main() -> None:
    """Writes the result of a trial to a file."""

    parser = ArgumentParser()
    parser.add_argument("--storage_file_path", type=str, required=True)
    parser.add_argument("--trial_id", type=int, required=True)
    parser.add_argument("--start_time", type=str, default=None)
    parser.add_argument("--end_time", type=str, default=None)
    parser.add_argument("--objective", nargs="+", type=str_or_float_or_int, default=None)
    parser.add_argument("--error", type=str, default="")
    parser.add_argument("--returncode", type=int, default=None)

    args = parser.parse_known_args()[0]

    unknown_args_list = parser.parse_known_args()[1]
    for unknown_arg in unknown_args_list:
        if unknown_arg.startswith("--"):
            parts = unknown_arg.split("=")
            name = parts[0].replace("--", "")
            parser.add_argument(f"--{name}", type=str_or_float_or_int)
    args = parser.parse_known_args()[0]

    xs = vars(copy.deepcopy(args))
    delete_keys = [
        "storage_file_path",
        "trial_id",
        "config",
        "start_time",
        "end_time",
        "objective",
        "error",
        "returncode",
    ]

    for key in delete_keys:
        if key in xs.keys():
            del xs[key]

    contents = {
        "trial_id": args.trial_id,
        "result": args.objective,
        "parameters": xs,
        "start_time": args.start_time,
        "end_time": args.end_time,
        "returncode": args.returncode,
        "error": args.error,
    }

    if args.error == "":
        del contents["error"]

    # print(contents)

    # create_yaml(args.file, contents)
    write_results_to_database(
        storage_file_path=args.storage_file_path,
        trial_id=args.trial_id,
        objective=args.objective,
        start_time=args.start_time,
        end_time=args.end_time,
        error=args.error,
        returncode=args.returncode,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
