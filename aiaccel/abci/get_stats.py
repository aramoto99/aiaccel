from __future__ import annotations

from subprocess import PIPE, Popen, TimeoutExpired
from typing import Any, dict, list

from aiaccel.abci import parse_qstat
from aiaccel.util import retry


@retry(_MAX_NUM=60, _DELAY=1.0)
def get_stats() -> list[dict[str, Any]]:
    """Get a current status and update.

    Returns:
        None
    """
    commands = "qstat -xml"
    p = Popen(commands, stdout=PIPE, shell=True)
    try:
        stdout_data, _ = p.communicate(timeout=1)
    except TimeoutExpired:
        p.kill()
        stdout_data, _ = p.communicate()
    stats = stdout_data.decode("utf-8")
    parsed = parse_qstat(stats)
    return parsed
