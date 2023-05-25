from __future__ import annotations

import copy
import datetime
import subprocess
import threading

from aiaccel.util.time_tools import get_time_now_object, get_time_string_from_object


class OutputHandler(threading.Thread):
    """A class to print subprocess outputs.

    Args:
        proc (Popen): A reference for subprocess.Popen.
            For example, 'Optimizer'.
    Attributes:
        _proc (Popen): A reference for subprocess.Popen.
            For example, 'Optimizer'.
        _sleep_time (int): A sleep time each loop.
    """

    def __init__(self, proc: subprocess.Popen[bytes]) -> None:
        super(OutputHandler, self).__init__()
        self._proc = proc
        self._sleep_time = 1
        self._abort = False

        self._returncode = None
        self._stdouts: list[str] = []
        self._stderrs: list[str] = []
        self._start_time: datetime.datetime | None = None
        self._end_time: datetime.datetime | None = None

    def abort(self) -> None:
        self._abort = True

    def run(self) -> None:
        """Main thread.

        Returns:
            None
        """
        self._start_time = get_time_now_object()
        self._stdouts = []
        self._stderrs = []

        while True:
            if self._proc.stdout is None:
                break

            stdout = self._proc.stdout.readline().decode().strip()
            if stdout:
                self._stdouts.append(stdout)

            if self._proc.stderr is not None:
                stderr = self._proc.stderr.readline().decode().strip()
                if stderr:
                    self._stderrs.append(stderr)
            else:
                stderr = None

            if not (stdout or stderr) and self.get_returncode() is not None:
                break

            if self._abort:
                break

        self._end_time = get_time_now_object()

    def get_stdouts(self) -> list[str]:
        return copy.deepcopy(self._stdouts)

    def get_stderrs(self) -> list[str]:
        return copy.deepcopy(self._stderrs)

    def get_start_time(self) -> str | None:
        if self._start_time is None:
            return ""
        return get_time_string_from_object(self._start_time)

    def get_end_time(self) -> str | None:
        if self._end_time is None:
            return ""
        return get_time_string_from_object(self._end_time)

    def get_returncode(self) -> int | None:
        return self._proc.poll()

    def raise_exception_if_error(self) -> None:
        """Raise an exception if an error is detected.

        Returns:
            None
        """
        if self._proc.returncode != 0:
            raise RuntimeError(
                f'An error occurred in the subprocess.\n'
                f'stdout: {self._stdouts}\n'
                f'stderr: {self._stderrs}'
            )

    def enforce_kill(self) -> None:
        """Enforce to kill the subprocess.

        Returns:
            None
        """
        self._proc.kill()
        raise RuntimeError(
            f'An error occurred in the subprocess.\n'
            f'stdout: {self._stdouts}\n'
            f'stderr: {self._stderrs}'
        )
