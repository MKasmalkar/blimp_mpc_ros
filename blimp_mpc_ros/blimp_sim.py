from . NonlinearBlimpSim import NonlinearBlimpSim
from . OriginLQRController import OriginLQRController
from . FeedbackLinLineFollower import FeedbackLinLineFollower
from . TrackingNoDamping import TrackingNoDamping
from . SingleActionDrive import SingleActionDrive
from . CBFHelix import CBFHelix
from . BlimpPlotter import BlimpPlotter
from . BlimpLogger import BlimpLogger

import numpy as np
import sys
import time

import rclpy
from rclpy.node import Node


def main(args=None):
    try:
        dT = 0.001
        ctrl_dT = 0.05
        ctrl_period = int(ctrl_dT / dT)
        
        ctrl_ctr = 0
        
        STOP_TIME = 120
        PLOT_ANYTHING = False
        PLOT_WAVEFORMS = False

        WINDOW_TITLE = 'Nonlinear'

        Simulator = NonlinearBlimpSim
        Controller = CBFHelix
        
        print("Running blimp simulator")
        
        ## SIMULATION

        sim = Simulator(dT)

        plotter = BlimpPlotter()
        plotter.init_plot(WINDOW_TITLE,
                waveforms=PLOT_WAVEFORMS,
                disable_plotting=(not PLOT_ANYTHING))

        ctrl = Controller(dT)
        ctrl.init_sim(sim)
        
        u = ctrl.get_ctrl_action(sim)        
        for n in range(int(STOP_TIME / dT)):
            print("Time: " + str(round(n*dT, 2))) 
            sim.update_model(u)
            
            ctrl_ctr += 1
            if ctrl_ctr > ctrl_period:
                u = ctrl.get_ctrl_action(sim)        
                ctrl_ctr = 0
            
            print(f"Current state: {round(sim.get_var('x'), 6)}, {round(sim.get_var('y'), 6)}, {round(sim.get_var('z'), 6)}, {round(sim.get_var('psi'), 6)}")

            plotter.update_plot(sim, ctrl)
            
            if plotter.window_was_closed():
                raise KeyboardInterrupt()
            
    finally:
        print("Logging to logfile.csv")
        logger = BlimpLogger("logfile.csv")
        logger.log(sim, ctrl)

if __name__ == '__main__':
    main()
