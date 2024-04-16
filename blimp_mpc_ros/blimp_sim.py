from . NonlinearBlimpSim import NonlinearBlimpSim
from . CBF import CBF
from . BlimpPlotter import BlimpPlotter
from . BlimpLogger import BlimpLogger

import numpy as np
import sys
import time

import rclpy
from rclpy.node import Node

class BlimpSimNode(Node):

    def __init__(self):
        super().__init__("blimp_sim")
        
        ## PARAMETERS

        dT = 0.05 
        STOP_TIME = 120
        PLOT_ANYTHING = True
        PLOT_WAVEFORMS = False

        WINDOW_TITLE = 'Nonlinear'

        Simulator = NonlinearBlimpSim
        Controller = CBF
        
        ## SIMULATION

        sim = Simulator(dT)

        plotter = BlimpPlotter()
        plotter.init_plot(WINDOW_TITLE,
                waveforms=PLOT_WAVEFORMS,
                disable_plotting=(not PLOT_ANYTHING))

        ctrl = Controller(dT)
        ctrl.init_sim(sim)

        try:
            for n in range(int(STOP_TIME / dT)):
                #print("Time: " + str(round(n*dT, 2)))
                u = ctrl.get_ctrl_action(sim)
                sim.update_model(u)
                
                print(f"Current state: {round(sim.get_var('x'), 6)}, {round(sim.get_var('y'), 6)}, {round(sim.get_var('z'), 6)}, {round(sim.get_var('psi'), 6)}")

                plotter.update_plot(sim, ctrl)
                
                if plotter.window_was_closed():
                    break

        except KeyboardInterrupt:
            print("Done!")
            sys.exit(0)

        finally:
            logger = BlimpLogger("logfile.csv")
            logger.log(sim, ctrl)

            if not plotter.window_was_closed():
                plotter.block()


def main(args=None):
    print('Running helix MPC simulated')

    try:
        rclpy.init(args=args)

        node = BlimpSimNode()
        
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.destroy_node()

if __name__ == '__main__':
    main()
