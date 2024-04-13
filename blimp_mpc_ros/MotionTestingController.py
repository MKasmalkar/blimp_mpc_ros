from . BlimpController import BlimpController
import control
import numpy as np

from . operators import *

class MotionTestingController(BlimpController):

    def __init__(self, dT):
        super().__init__(dT)
        
    def get_ctrl_action(self, sim):

        return np.array([0.1, 0.0, 0.0, 0.0]).reshape((4,1))

    def get_error(self, sim):
        return np.array([
            sim.get_var_history('x'),
            sim.get_var_history('y'),
            sim.get_var_history('z'),
            sim.get_var_history('psi')
        ]).T