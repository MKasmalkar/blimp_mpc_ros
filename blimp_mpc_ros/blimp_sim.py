from . NonlinearBlimpSim import NonlinearBlimpSim
from . OriginLQRController import OriginLQRController
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
        dT = 0.05 
        STOP_TIME = 7
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
        
        sim.set_var('vx', -0.00131)
        sim.set_var('vy', 0.0485)
        sim.set_var('vz', -0.000369)
        sim.set_var('wx', -0.101)
        sim.set_var('wy', -0.07715)
        sim.set_var('wz', 0.15069)
        sim.set_var('x', -2.59867)
        sim.set_var('y', -0.090999)
        sim.set_var('z', -0.985318)
        sim.set_var('phi', 0.039786)
        sim.set_var('theta', 0.15306)
        sim.set_var('psi', -0.078)
        
        sim.set_var_dot('vx', 17.3285)
        sim.set_var_dot('vy', -10.097)
        sim.set_var_dot('vz', -1.57)
        sim.set_var_dot('wx', -0.101)
        sim.set_var_dot('wy', -0.077)
        sim.set_var_dot('wz', 0.1506)
        sim.set_var_dot('x', 0.00272)
        sim.set_var_dot('y', 0.048)
        sim.set_var_dot('z', 0.0017)
        sim.set_var_dot('phi', -0.13127)
        sim.set_var_dot('theta', -0.394)
        sim.set_var_dot('psi', 0.31634)

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
