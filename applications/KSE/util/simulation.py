import time

import numpy as np
from prettyPlot.progressBar import print_progress_bar


def run(Sim):
    # Timing
    t_start = time.time()

    Result = {}

    Ndof = Sim["Ndof"]
    h = Sim["Timestep"]
    tmax = Sim["Tf"]
    nmax = Sim["nmax"]
    nplt = Sim["nplt"]
    stepFunc = Sim["stepFunc"]
    srcFunc = Sim["srcFunc"]
    epsilon_init = Sim["epsilon_init"]
    nMonitor = int(nmax / 100)

    uu = np.zeros((nmax + 1, Ndof, 1))

    tt = np.arange(0, (nmax + 1) * h, h)

    u = np.transpose(
        Sim["u0"] * np.ones((1, Ndof))
    ) + epsilon_init * np.random.normal(loc=0.0, scale=1.0, size=(Ndof, 1))

    # Init
    uu[0, :, 0] = u[:, 0]
    t_per_step = 0

    print_progress_bar(
        0,
        nmax,
        prefix="Iter " + str(0) + " / " + str(nmax),
        suffix="Complete",
        length=50,
    )
    # main loop
    for n in range(1, nmax + 1):
        t = n * h
        t_start_step = time.time()
        u = stepFunc(u, Sim)
        u = srcFunc(u, Sim)
        t_end_step = time.time()
        t_per_step = (t_per_step * (n - 1) + (t_end_step - t_start_step)) / n
        if n % nplt == 0:
            uu[n, :, 0] = u[:, 0]
        print_progress_bar(
            n,
            nmax,
            prefix="Iter %d/%d , t/step = %.3f mu s"
            % (n, nmax, 1e6 * t_per_step),
            suffix="Complete",
            length=20,
        )
    # Timing
    t_end = time.time()

    Result["tt"] = tt
    Result["uu"] = uu
    Result["timeExec"] = t_end - t_start

    return Result


def simRun(Sim):

    if Sim["Simulation name"] == "KS":
        return run(Sim)
