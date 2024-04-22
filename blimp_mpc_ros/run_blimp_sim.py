from . BlimpSim import BlimpSim
from . BlimpPlotter import BlimpPlotter
from . BlimpLogger import BlimpLogger

from . CtrlCBFLine import CtrlCBFLine
from . CtrlCBFTriangle import CtrlCBFTriangle
from . CtrlCBFHelix import CtrlCBFHelix
from . CtrlFblLine import CtrlFblLine
from . CtrlFblTriangle import CtrlFblTriangle
from . CtrlFblHelix import CtrlFblHelix
from . CtrlPIDLine import CtrlPIDLine
from . CtrlPIDTriangle import CtrlPIDTriangle
from . CtrlPIDHelix import CtrlPIDHelix

import numpy as np
import sys
import time

import rclpy
from rclpy.node import Node


def main(args=sys.argv):
    if len(args) < 3:
        print("Please run with controller and log file name as arguments.")
        sys.exit(0)
        
    mapping = {
        'cbf_line': CtrlCBFLine,
        'cbf_triangle': CtrlCBFTriangle,
        'cbf_helix': CtrlCBFHelix,
        'fbl_line': CtrlFblLine,
        'fbl_triangle': CtrlFblTriangle,
        'fbl_helix': CtrlFblHelix,
        'pid_line': CtrlPIDLine,
        'pid_triangle': CtrlPIDTriangle,
        'pid_helix': CtrlPIDHelix
    }
        
    try:
        dT = 0.001
        ctrl_dT = 0.05
        ctrl_period = int(ctrl_dT / dT)
        
        ctrl_ctr = 0
        
        STOP_TIME = 20
        PLOT_ANYTHING = False
        PLOT_WAVEFORMS = False

        WINDOW_TITLE = 'Nonlinear'

        Simulator = BlimpSim
        Controller = mapping[args[1]]
        
        print("Running blimp simulator: " + args[1])
        
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
            print(f"Control: {round(u[0].item(), 6)}, {round(u[1].item(), 6)}, {round(u[2].item(), 6)}, {round(u[3].item(), 6)}")

            plotter.update_plot(sim)
            
            if plotter.window_was_closed():
                raise KeyboardInterrupt()
            
    finally:
        print("Logging to logs/" + args[2])
        logger = BlimpLogger(args[2])
        logger.log(sim, ctrl)

if __name__ == '__main__':
    main()