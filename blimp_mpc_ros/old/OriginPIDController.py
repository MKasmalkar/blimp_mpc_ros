from . BlimpController import BlimpController
import numpy as np
import time

from . operators import *

class OriginPIDController(BlimpController):

    def __init__(self, dT):
        super().__init__(dT)

        self.Kp_pos = 0.0125 * 1e-4
        self.Ki_pos = 0.0
        self.Kd_pos = 0.0

        self.Kp_x = self.Kp_pos
        self.Ki_x = self.Ki_pos
        self.Kd_x = self.Kd_pos

        self.Kp_y = self.Kp_pos
        self.Ki_y = self.Ki_pos
        self.Kd_y = self.Kd_pos

        self.Kp_z = 0.0625 * 1e-4
        self.Ki_z = 0.005 * 1e-4
        self.Kd_z = 0.2 * 1e-4

        self.Kp_psi = 0.0001 * 1e-4
        self.Ki_psi = 0.0
        self.Kd_psi = 0.0

        self.x_error_prev = None
        self.y_error_prev = None
        self.z_error_prev = None
        self.psi_error_prev = None

        self.x_error_int = 0
        self.y_error_int = 0
        self.z_error_int = 0
        self.psi_error_int = 0
        
        self.x_ref = 0
        self.y_ref = 0
        self.z_ref = -1.5
        self.psi_ref = 0

        self.prev_time = None

    def get_ctrl_action(self, sim):
        current_time = time.time()
        
        x   = sim.get_var('x')
        y   = sim.get_var('y')
        z   = sim.get_var('z')
        psi = sim.get_var('psi')

        phi = sim.get_var('phi')
        theta = sim.get_var('theta')

        if self.prev_time is None:
            self.prev_time = current_time
            
            self.x_error_prev   = self.x_ref   - x
            self.y_error_prev   = self.y_ref   - y
            self.z_error_prev   = self.z_ref   - z
            self.psi_error_prev = self.psi_ref - psi

            return np.array([0.0, 0.0, 0.0, 0.0])
        
        deltaT = current_time - self.prev_time
        self.prev_time = current_time
        
        x_error = self.x_ref - x
        y_error = self.y_ref - y
        z_error = self.z_ref - z
        psi_error = self.psi_ref - psi

        self.x_error_int   += x_error   * deltaT
        self.y_error_int   += y_error   * deltaT
        self.z_error_int   += z_error   * deltaT
        self.psi_error_int += psi_error * deltaT
        
        dXdt   = (x_error   - self.x_error_prev)   / deltaT
        dYdt   = (y_error   - self.y_error_prev)   / deltaT
        dZdt   = (z_error   - self.z_error_prev)   / deltaT
        dPsidt = (psi_error - self.psi_error_prev) / deltaT

        self.x_error_prev   = x_error
        self.y_error_prev   = y_error
        self.z_error_prev   = z_error
        self.psi_error_prev = psi_error

        fx_world   = self.Kp_x   * x_error   + self.Ki_x   * self.x_error_int   + self.Kd_x   * dXdt
        fy_world   = self.Kp_y   * y_error   + self.Ki_y   * self.y_error_int   + self.Kd_y   * dYdt
        fz_world   = self.Kp_z   * z_error   + self.Ki_z   * self.z_error_int   + self.Kd_z   * dZdt
        tauz_world = self.Kp_psi * psi_error + self.Ki_psi * self.psi_error_int + self.Kd_psi * dPsidt 

        f_world = np.array([fx_world, fy_world, fz_world])
        f_body = R_b__n_inv(phi, theta, psi) @ f_world.reshape((3,1))

        return np.array([f_body[0].item(), f_body[1].item(), f_body[2].item(), tauz_world])
        # return np.array([f_body[0].item(), f_body[1].item(), f_body[2].item(), 0.0])
        # return np.array([0.0, 0.0, 0.0, tauz_world])

    def get_error(self, sim):
        return np.array([
            sim.get_var_history('x') - self.x_ref,
            sim.get_var_history('y') - self.y_ref,
            sim.get_var_history('z') - self.z_ref,
            sim.get_var_history('psi') - self.psi_ref
        ]).T