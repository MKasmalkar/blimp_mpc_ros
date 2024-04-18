from . NonlinearBlimpSim import NonlinearBlimpSim

from . BlimpPlotter import BlimpPlotter
from . BlimpLogger import BlimpLogger

from . OriginLQRController import OriginLQRController

import numpy as np
import time
import sys


def main():
    ## Parameters

    TITLE = "Plots"

    # Neither of these selections matter - these objects
    # just need to be created in order to load and plot
    # the simulation data from the file.

    Simulator = NonlinearBlimpSim
    Controller = OriginLQRController

    ## Plotting

    if len(sys.argv) < 2:
        print("Please run with data file name as first argument.")
        sys.exit(0)

    dT = 0.001  # will be overridden by data load anyways
    sim = Simulator(dT)
    ctrl = Controller(dT)
    plotter = BlimpPlotter()

    sim.load_data(sys.argv[1])
    ctrl.load_data(sys.argv[1])
    plotter.init_plot(TITLE, True)

    plotter.update_plot(sim, ctrl)
    plotter.block()
