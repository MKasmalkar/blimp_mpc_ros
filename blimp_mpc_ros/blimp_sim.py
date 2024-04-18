from . NonlinearBlimpSim import NonlinearBlimpSim
from . OriginLQRController import OriginLQRController
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
        Controller = OriginLQRController
        
        ## SIMULATION

        sim = Simulator(dT)

        plotter = BlimpPlotter()
        plotter.init_plot(WINDOW_TITLE,
                waveforms=PLOT_WAVEFORMS,
                disable_plotting=(not PLOT_ANYTHING))

        ctrl = Controller(dT)
        ctrl.init_sim(sim)
        
        sim.set_var('vx', 0.0005157)
        sim.set_var('vy', 0.00055818)
        sim.set_var('vz', -0.001037)
        sim.set_var('wx', -0.00445598)
        sim.set_var('wy', -0.0016)
        sim.set_var('wz', 0.0054)
        sim.set_var('x', -2.981)
        sim.set_var('y', 0.07118)
        sim.set_var('z', -1.098)
        sim.set_var('phi', 0.0408)
        sim.set_var('theta', 0.1657)
        sim.set_var('psi', 0.03557)

        try:
            for n in range(int(STOP_TIME / dT)):
                print("Time: " + str(round(n*dT, 2)))
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
