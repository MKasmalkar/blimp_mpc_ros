from . BlimpController import BlimpController
import control
import numpy as np

class OriginLQRController(BlimpController):

    def __init__(self, dT):
        super().__init__(dT)

        self.Q = np.eye(12)
        self.R = np.eye(4)

    def get_ctrl_action(self, sim):
        A = sim.get_A_lin()
        B = sim.get_B_lin()

        K = control.lqr(A, B, self.Q, self.R)[0]
        return (-K @ sim.get_state().reshape(12)).reshape((4,1))
    
    def get_error(self, sim):
        return np.array([
            sim.get_var_history('x'),
            sim.get_var_history('y'),
            sim.get_var_history('z'),
            sim.get_var_history('psi')
        ]).T