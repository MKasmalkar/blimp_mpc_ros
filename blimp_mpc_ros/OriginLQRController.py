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
        	[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        	[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        	[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        	[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
        self.R = np.array([
        	[100, 0, 0, 0],
        	[0, 100, 0, 0],
        	[0, 0, 1, 0],
        	[0, 0, 0, 1000]
        ])

        self.reference = np.array([0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0])

    def init_sim(self, sim):
        A = sim.get_A_lin()
        B = sim.get_B_lin()
        
        print(A)
       
        self.K = control.lqr(A, B, self.Q, self.R)[0]

    def get_ctrl_action(self, sim):
        error = sim.get_state().reshape(12) - self.reference

        print(f"Error: {error[6].item()}, {error[7].item()}, {error[8].item()}, {error[11].item()}")

        u = (-self.K @ error.reshape((12,1))).reshape((4,1))
        # u_modified = np.array([u[0].item(), u[1].item(), u[2].item(), 0]).reshape((4,1))

        #for i in range(len(u)):
        #    if abs(u[i].item()) > 0.06:
        #        u[i] = u[i].item() / abs(u[i].item()) * 0.06

        print(f"Control: {u[0].item()}, {u[1].item()}, {u[2].item()}, {u[3].item()}")

        return u
    
    def get_error(self, sim):
        return np.array([
            sim.get_var_history('x'),
            sim.get_var_history('y'),
            sim.get_var_history('z') - (-1),
            sim.get_var_history('psi')
        ]).T
    
# from . BlimpController import BlimpController
# import control
# import numpy as np

# from . operators import *

# class OriginLQRController(BlimpController):

#     def __init__(self, dT):
#         super().__init__(dT)

#         self.Q = np.eye(12) * 1/10

#         max_force = 1e-7
#         max_torque = 1e-7
#         self.R = np.array([
#             [1/max_force**2, 0, 0, 0],
#             [0, 1/max_force**2, 0, 0],
#             [0, 0, 1/max_force**2, 0],
#             [0, 0, 0, 1/max_torque**2]
#         ])

#         self.A_lin = np.array([
#             [-0.024918743228681705659255385398865, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, -0.024918743228681705659255385398865, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, -0.064008534471213351935148239135742, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, -0.016812636348731757607311010360718, 0, 0, 0, 0, 0, -0.15352534538760664872825145721436, 0, 0],
#             [0, 0, 0, 0, -0.016812636348731757607311010360718, 0, 0, 0, 0, 0, -0.15352534538760664872825145721436, 0],
#             [0, 0, 0, 0, 0, -0.016835595258726243628188967704773, 0, 0, 0, 0, 0, 0],
#             [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0]
#         ])

#     def get_ctrl_action(self, sim):
#         # A = sim.get_A_lin()
#         B = sim.get_B_lin()
#         K = control.lqr(self.A_lin, B, self.Q, self.R)[0]
#         u_orig = (-K @ sim.get_state().reshape(12)).reshape((4,1))

#         # A matrix is generated assuming psi = 0
#         # need to perform some rotations to account for this
#         sim_state = sim.get_state()
#         psi = sim_state[11].item()
        
#         u_rot = R_b__n_inv(0, 0, psi) @ np.array([u_orig[0], u_orig[1], u_orig[2]]).reshape((3,1))

#         u = np.array([
#             [u_rot[0].item()],
#             [u_rot[1].item()],
#             [u_rot[2].item()],
#             [u_orig[3].item()]
#         ])


#         return u
    
#     def get_error(self, sim):
#         return np.array([
#             sim.get_var_history('x'),
#             sim.get_var_history('y'),
#             sim.get_var_history('z'),
#             sim.get_var_history('psi')
#         ]).T
