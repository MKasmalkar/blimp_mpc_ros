from . BlimpController import BlimpController
import control
import numpy as np

from . operators import *

class OriginLQRController(BlimpController):

    def __init__(self, dT):
        super().__init__(dT)

        self.Q = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1000, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1000, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1000, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1000, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1000, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        
        self.R = np.array([
            [1000, 0, 0, 0],
            [0, 1000, 0, 0],
            [0, 0, 1000, 0],
            [0, 0, 0, 1e6]
        ])
        
    def get_ctrl_action(self, sim):
        A = sim.get_A_lin()
        B = sim.get_B_lin()

        K = control.lqr(A, B, self.Q, self.R)[0]

        reference = np.array([0, 0, 0, 0, 0, 0, 0, 0, -1.5, 0, 0, 0])

        ctrl = (-K @ (sim.get_state().reshape(12) - reference)).reshape((4,1))

        return np.array([ctrl[0].item(), ctrl[1].item(), ctrl[2].item(), ctrl[3].item()]).reshape((4,1))

    def get_error(self, sim):
        return np.array([
            sim.get_var_history('x'),
            sim.get_var_history('y'),
            sim.get_var_history('z'),
            sim.get_var_history('psi')
        ]).T