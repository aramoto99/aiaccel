# from time import sleep
import numpy as np
<<<<<<< HEAD:examples/mpi/sphere_cpu_1node_1try/user.py
from aiaccel.util import aiaccel
=======

from aiaccel.experimental.mpi.util import aiaccel
>>>>>>> d4c097c (Merge changes from the original fork):examples/experimental/mpi/sphere_cpu_1node_1try/user.py


def main(p):
    x = np.array([p["x1"], p["x2"], p["x3"], p["x4"], p["x5"]])
    y = np.sum(x ** 2)
    # sleep(20)
    return float(y)


if __name__ == "__main__":
    run = aiaccel.Run()
    run.execute_and_report(main)
