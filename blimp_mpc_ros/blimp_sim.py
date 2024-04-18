from . NonlinearBlimpSim import NonlinearBlimpSim
from . OriginLQRController import OriginLQRController
from . FeedbackLinLineFollower import FeedbackLinLineFollower
from . TrackingNoDamping import TrackingNoDamping
from . SingleActionDrive import SingleActionDrive
from . BlimpPlotter import BlimpPlotter
from . BlimpLogger import BlimpLogger

import numpy as np
import sys
import time

import rclpy
from rclpy.node import Node


def main(args=None):
    try:
        dT = 0.005
        
        STOP_TIME = 22
        PLOT_ANYTHING = False
        PLOT_WAVEFORMS = False

        WINDOW_TITLE = 'Nonlinear'

        Simulator = NonlinearBlimpSim
        Controller = SingleActionDrive
        
        print("Running blimp simulator")
        
        ## SIMULATION

        sim = Simulator(dT)

        plotter = BlimpPlotter()
        plotter.init_plot(WINDOW_TITLE,
                waveforms=PLOT_WAVEFORMS,
                disable_plotting=(not PLOT_ANYTHING))

        ctrl = Controller(dT)
        ctrl.init_sim(sim)
        
        sim.set_var('vx', -0.00825)
        sim.set_var('vy', -0.01607)
        sim.set_var('vz', -0.02236)
        sim.set_var('wx', 0.03879)
        sim.set_var('wy', 0.10188)
        sim.set_var('wz', 0.009198)
        sim.set_var('x', -2.9138)
        sim.set_var('y', -0.5215)
        sim.set_var('z', -1.197)
        sim.set_var('phi', -0.1354)
        sim.set_var('theta', -0.457)
        sim.set_var('psi', -0.01149)
              
        for n in range(int(STOP_TIME / dT)):
            print("Time: " + str(round(n*dT, 2)))
            u = ctrl.get_ctrl_action(sim)         
            sim.update_model(u)
            
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
