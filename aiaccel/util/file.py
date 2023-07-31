from __future__ import annotations

from pathlib import Path
from typing import Any

import fasteners
import yaml


def create_yaml(path: Path, content: Any, dict_lock: Path | None = None) -> None:
    """Create a yaml file.
    Args:
        path (Path): The path of the created yaml file.
        content (dict): The content of the created yaml file.
        dict_lock (Path | None, optional): The path to store lock files.
            Defaults to None.

    Returns:
        None
    """
    if dict_lock is None:
        with open(path, "w") as f:
            f.write(yaml.dump(content, default_flow_style=False))
    else:
        lock_file = dict_lock / path.parent.name
        with fasteners.InterProcessLock(lock_file):
            with open(path, "w") as f:
                f.write(yaml.dump(content, default_flow_style=False))
