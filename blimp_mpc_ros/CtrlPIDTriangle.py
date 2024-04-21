from . BlimpController import BlimpController
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import control
from . parameters import *
from . PID import PID
from . CtrlPID import CtrlPID
from . Trajectories import Trajectories
import math
import sys

class CtrlPIDTriangle(BlimpController):

    def __init__(self, dT, skip_derivatives=True):
        super().__init__(dT)

        self.metadata = np.array([
            f"kp_x = {self.kp_x}",
            f"ki_x = {self.ki_x}",
            f"kd_x = {self.kd_x}",
            f"kp_y = {self.kp_y}",
            f"ki_y = {self.ki_y}",
            f"kd_y = {self.kd_y}",
            f"kp_z = {self.kp_z}",
            f"ki_z = {self.ki_z}",
            f"kd_z = {self.kd_z}",
            f"kp_psi = {self.kp_psi}",
            f"ki_psi = {self.ki_psi}",
            f"kd_psi = {self.kd_psi}",
            f"dT = {dT}"
        ])
        

    def init_sim(self, sim):
        
        x0 = sim.get_var('x')
        y0 = sim.get_var('y')
        z0 = sim.get_var('z')
        psi0 = sim.get_var('psi')
        
        trajectory = Trajectories.get_triangle(x0, y0, z0, psi0, self.dT)
        self.init_trajectory(trajectory)
        
        self.is_initialized = True
