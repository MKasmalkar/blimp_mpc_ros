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
        dT = 0.001 
        dT_controller = 0.05
        
        ctrl_ctr = 0
        ctrl_period = int(dT_controller / dT)
        
        STOP_TIME = 4
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
        
        sim.set_var('vx', -0.00467)
        sim.set_var('vy', -0.08222)
        sim.set_var('vz', -0.050567)
        sim.set_var('wx', 0.2167)
        sim.set_var('wy', 0.0543472)
        sim.set_var('wz', -0.048587)
        sim.set_var('x', -3.1111)
        sim.set_var('y', 0.2364)
        sim.set_var('z', -1.0988)
        sim.set_var('phi', -0.0056608)
        sim.set_var('theta', 0.175726)
        sim.set_var('psi', 0.30198)
        
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
