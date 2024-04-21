from . BlimpController import BlimpController
from . CtrlFbl import CtrlFbl
from . Trajectories import Trajectories
import numpy as np
import control
from . parameters import *

class CtrlFblLine(CtrlFbl):

    def __init__(self, dT, skip_derivatives=True):
        super().__init__(dT)
        
        self.metadata = np.array([
            f"k1 = {self.k1.T}",
            f"k2 = {self.k2.T}",
            f"dT = {dT}"
        ])
        
    def init_sim(self, sim):
        x0 = sim.get_var('x')
        y0 = sim.get_var('y')
        z0 = sim.get_var('z')
        psi0 = sim.get_var('psi')
        
        trajectory = Trajectories.get_line(x0, y0, z0, psi0, self.dT)
        self.init_trajectory(trajectory)
        
        self.is_initialized = True

