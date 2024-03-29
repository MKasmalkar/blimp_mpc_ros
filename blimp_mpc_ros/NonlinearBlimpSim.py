from . BlimpSim import BlimpSim
import numpy as np
from . parameters import *

class NonlinearBlimpSim(BlimpSim):

    def __init__(self, dT):
        super().__init__(dT)

    def update_model(self, u):
        self.u = u

        tau_B = np.array([u[0],
                            u[1],
                            u[2],
                            -r_z_tg__b * u[1],
                            r_z_tg__b * u[0],
                            u[3]]).reshape((6,1))
        
        x = self.get_var('x')
        y = self.get_var('y')
        z = self.get_var('z')
        phi = self.get_var('phi')
        theta = self.get_var('theta')
        psi = self.get_var('psi')

        eta_bn_n = np.array([x, y, z, phi, theta, psi]).reshape((6,1))
        
        vx = self.get_var('vx')
        vy = self.get_var('vy')
        vz = self.get_var('vz')
        wx = self.get_var('wx')
        wy = self.get_var('wy')
        wz = self.get_var('wz')

        nu_bn_b = np.array([vx, vy, vz, wx, wy, wz]).reshape((6,1))

        # Restoration torque
        fg_B = R_b__n_inv(phi, theta, psi) @ fg_n
        g_CB = -np.block([[np.zeros((3, 1))],
                        [np.reshape(np.cross(r_gb__b, fg_B), (3, 1))]])

        # Update state
        eta_bn_n_dot = np.block([[R_b__n(phi, theta, psi),    np.zeros((3, 3))],
                                [np.zeros((3, 3)),            T(phi, theta)]]) @ nu_bn_b
        
        nu_bn_b_dot = np.reshape(-M_CB_inv @ (C(M_CB, nu_bn_b) @ nu_bn_b + \
                            D_CB @ nu_bn_b + g_CB - tau_B), (6, 1))
        
        eta_bn_n = eta_bn_n + eta_bn_n_dot * self.dT
        nu_bn_b = nu_bn_b + nu_bn_b_dot * self.dT

        self.state_dot = np.vstack((nu_bn_b_dot, eta_bn_n_dot))
        self.state = np.vstack((nu_bn_b, eta_bn_n))

        self.update_history()