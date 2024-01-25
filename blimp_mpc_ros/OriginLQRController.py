from . BlimpController import BlimpController
import control
import numpy as np

from . operators import *

class OriginLQRController(BlimpController):

    def __init__(self, dT):
        super().__init__(dT)

        self.Q = np.array([
            [0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0.001, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.001, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.001, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1000, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1000, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1000, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.001, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1000]
        ])
        self.R = np.eye(4) * 10000
        
    def get_ctrl_action(self, sim):
        A = sim.get_A_lin()
        B = sim.get_B_lin()

        K = control.lqr(A, B, self.Q, self.R)[0]

        reference = np.array([0, 0, 0, 0, 0, 0, 0, 0, -1.5, 0, 0, 0])

        return (-K @ (sim.get_state().reshape(12) - reference)).reshape((4,1))
    
    def get_error(self, sim):
        return np.array([
            sim.get_var_history('x'),
            sim.get_var_history('y'),
            sim.get_var_history('z'),
            sim.get_var_history('psi')
        ]).T