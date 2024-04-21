import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
from . BlimpController import BlimpController
from . Trajectories import Trajectories
from . parameters import *
from . PID import PID
import sys

class CtrlPID(BlimpController):

    def __init__(self, dT):
        super().__init__(dT)
        
        self.kp_x = 1
        self.ki_x = 0.1
        self.kd_x = 0
        
        self.kp_y = 1
        self.ki_y = 0.1
        self.kd_y = 0
        
        self.kp_z = 1
        self.ki_z = 0.1
        self.kd_z = 0
        
        self.kp_psi = 1
        self.ki_psi = 0.1
        self.kd_psi = 0
        
        self.x_pid = PID(self.kp_x, self.ki_x, self.kd_x, dT)
        self.y_pid = PID(self.kp_y, self.ki_y, self.kd_y, dT)
        self.z_pid = PID(self.kp_z, self.ki_z, self.kd_z, dT)
        self.psi_pid = PID(self.kp_psi, self.ki_psi, self.kd_psi, dT)

    def get_ctrl_action(self, sim):
        
        sim.start_timer()
        
        n = sim.get_current_timestep()
        
        if n > len(self.traj_x):
            return None

        # Extract state variables
        x = sim.get_var('x')
        y = sim.get_var('y')
        z = sim.get_var('z')

        psi = sim.get_var('psi')
        
        u = np.array([
            self.x_pid.get_ctrl(self.traj_x[n] - x),
            self.y_pid.get_ctrl(self.traj_y[n] - y),
            self.z_pid.get_ctrl(self.traj_z[n] - z),
            self.psi_pid.get_ctrl(self.traj_psi[n] - psi)
        ])

        sim.end_timer()

        return u.reshape((4,1))
