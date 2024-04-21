from . BlimpController import BlimpController
import numpy as np
import control
from . parameters import *
from . PID import PID
import math
import sys

class StopBlimp(BlimpController):

    def __init__(self, dT, skip_derivatives=True):
        super().__init__(dT)

    def init_sim(self, sim):
        pass

    def get_ctrl_action(self, sim):
        return np.array([0, 0, 0, 0])