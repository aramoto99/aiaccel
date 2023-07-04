from __future__ import annotations

import copy
import datetime
import subprocess
import threading

from aiaccel.common import datetime_format


class OutputHandler(threading.Thread):
    """A class to print subprocess outputs.

    Args:
        proc (subprocess.Popen): A reference to subprocess.Popen.
            For example, 'Optimizer'.
    """

    def __init__(self, proc: subprocess.Popen[bytes]) -> None:
        super().__init__()
        self._proc = proc
        self._sleep_time = 1
        self._abort = False

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
        self._start_time = datetime.datetime.now()
        self._stdouts = []
        self._stderrs = []

        while True:
            if self._proc.stdout is None:
                break

            stdout = self._proc.stdout.readline()
            if stdout:
                stdout_str = stdout.decode().strip()
                print(stdout_str)
                self._stdouts.append(stdout_str)

            if self._proc.stderr is not None:
                stderr = self._proc.stderr.readline()
                if stderr:
                    stderr_str = stderr.decode().strip()
                    print(stderr_str)
                    self._stderrs.append(stderr_str)
            else:
                stderr = None

            if not (stdout or stderr) and self.get_returncode() is not None:
                break

            if self._abort:
                break

        self._end_time = datetime.datetime.now()

    def get_stdouts(self) -> list[str]:
        return copy.deepcopy(self._stdouts)

    def get_stderrs(self) -> list[str]:
        return copy.deepcopy(self._stderrs)

    def get_start_time(self) -> str | None:
        return self._start_time.strftime(datetime_format) if self._start_time else ""

    def get_end_time(self) -> str | None:
        return self._end_time.strftime(datetime_format) if self._end_time else ""

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
        """Enforce killing the subprocess.

        Returns:
            None
        """
        self._proc.kill()
        raise RuntimeError(
            f'An error occurred in the subprocess.\n'
            f'stdout: {self._stdouts}\n'
            f'stderr: {self._stderrs}'
        )
