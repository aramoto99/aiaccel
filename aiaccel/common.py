"""Common variables and methods.

Example: ::

    from aiaccel.common import alive_optimizer

"""

dict_hp = "hp"
dict_lock = "lock"
dict_log = "log"
dict_error = "error"
dict_output = "abci_output"
dict_jobstate = "jobstate"
dict_result = "result"
dict_runner = "runner"
dict_timestamp = "timestamp"
dict_storage = "storage"
dict_tensorboard = "tensorboard"
dict_mpi = "mpi"
dict_rank_log = "rank_log"

extension_hp = "hp"

file_hp_count = "count.txt"
file_hp_count_lock = "count.lock"
file_hp_count_lock_timeout = 10

file_mpi_lock = "mpi.lock"
file_mpi_lock_timeout = 10

goal_maximize = "maximize"
goal_minimize = "minimize"

resource_type_local = "local"
resource_type_abci = "abci"
resource_type_memory = "python_local"
resource_type_mpi = "mpi"

datetime_format = "%m/%d/%Y %H:%M:%S"
