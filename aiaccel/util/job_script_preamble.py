from __future__ import annotations

# import re
from pathlib import Path

# from aiaccel.config import Config


def create_job_script_preamble(job_script_preamble_path: Path | str | None, job_script_preamble_str: str) -> str:
    if job_script_preamble_path is None or job_script_preamble_path == "":
        return job_script_preamble_str
    else:
        if Path(job_script_preamble_path).exists():
            with open(job_script_preamble_path, "r") as f:
                job_script_preamble = f.read()
            return job_script_preamble
        else:
            raise FileNotFoundError(f"File not found: {job_script_preamble_path}")


# WIP
# class JobScriptPreambleCreator:
#     def __init__(self, config: Config) -> None:
#         self.job_script_preamble_path = config.ABCI.job_script_preamble_path
#         self.job_script_preamble_str = config.ABCI.job_script_preamble
#         self.rt_type = config.resource.rt_type
#         self.rt_num = config.resource.rt_num
#         self.h_rt = config.resource.h_rt
#         self.num_workers = config.resource.num_workers
#         self.variables = {
#             "rt_type": self.rt_type,
#             "rt_num": self.rt_num,
#             "h_rt": self.h_rt,
#             "num_workers": self.num_workers,
#         }

#     def get(self) -> str:
#         if self.job_script_preamble_path is None or self.job_script_preamble_path == "":
#             return self.job_script_preamble_str
#         else:
#             with open(self.job_script_preamble_path, "r") as f:
#                 job_script_preamble = f.read()
#             return job_script_preamble

#     def replace_variables(self, input_text: str, variables: dict[str, str]) -> str:
#         pattern = r"\{([^}]+)\}"
#         output_text = re.sub(pattern, lambda match: variables.get(match.group(1), match.group(0)), input_text)
#         return output_text
