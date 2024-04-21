
from . NonlinearBlimpSim import NonlinearBlimpSim

class RealBlimp(NonlinearBlimpSim):
    
    def __init__(self, dT):
        super().__init__(dT)