from . BlimpController import BlimpController
import numpy as np
import control
from . parameters import *
from . PID import PID
import math
import sys

class SingleActionDrive(BlimpController):

    def __init__(self, dT, skip_derivatives=True):
        super().__init__(dT)

        self.z_pid = PID(0.1, 0, 0, dT)
        self.yaw_pid = PID(0.001/(2*np.pi), 0, 0, dT)

        self.z_initial = np.NaN
        self.yaw_initial = np.NaN

    def init_sim(self, sim):
        pass

    def get_ctrl_action(self, sim):

        if np.isnan(self.z_initial):
            self.z_initial = sim.get_var('z')

        if np.isnan(self.yaw_initial):
            self.yaw_initial = sim.get_var('psi')

        yaw = sim.get_var('psi')
        yaw_action = self.yaw_pid.get_ctrl(self.yaw_initial - yaw)
        
        z = sim.get_var('z')
        z_action = self.z_pid.get_ctrl(self.z_initial - z)

        return np.array([0.0, 0.0, -0.025, 0.0])
    
    def get_error(self, sim):
        n = sim.get_current_timestep() + 1
        return np.array([
            sim.get_var_history('x'),
            sim.get_var_history('y'),
            sim.get_var_history('z'),
            sim.get_var_history('psi')
        ]).T
