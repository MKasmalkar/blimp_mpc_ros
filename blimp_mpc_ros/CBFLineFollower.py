import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
from . BlimpController import BlimpController
from . parameters import *
import sys

class CBFLineFollower(BlimpController):

    def __init__(self, dT):
        super().__init__(dT)

        self.order = 12
        self.num_inputs = 4
        self.num_outputs = 6

        # Trajectory definition
        TRACKING_TIME = 20
        SETTLE_TIME = 100

        self.theta_limit = 10 * np.pi/180
        self.phi_limit = 10 * np.pi/180
        self.psi_limit = 5 * np.pi/180
        
        tracking_time = np.arange(0, TRACKING_TIME, self.dT)
        settle_time = np.arange(TRACKING_TIME, TRACKING_TIME + SETTLE_TIME + 1, self.dT)

        time_vec = np.concatenate((tracking_time, settle_time))
        
        self.k_x_list = []
        self.u_delta_list = []
        
        m = 0.05
        
        x0 = 0
        y0 = 0
        z0 = 0
        psi0 = 0
        
        self.traj_x = np.concatenate((x0 + m*tracking_time, (x0 + m*TRACKING_TIME) * np.ones(len(settle_time))))
        self.traj_y = np.concatenate((y0 * np.ones(len(tracking_time)), y0 * np.ones(len(settle_time))))
        self.traj_z = np.concatenate((z0 * np.ones(len(tracking_time)), z0 * np.ones(len(settle_time))))
        self.traj_psi = np.concatenate((psi0 * np.ones(len(tracking_time)), psi0 * np.ones(len(settle_time))))
        
        self.traj_x_dot = np.concatenate((m * np.ones(len(tracking_time)), np.zeros(len(settle_time))))
        self.traj_y_dot = np.concatenate((np.zeros(len(tracking_time)), np.zeros(len(settle_time))))
        self.traj_z_dot = np.concatenate((np.zeros(len(tracking_time)), np.zeros(len(settle_time))))
        self.traj_psi_dot = np.concatenate((np.zeros(len(tracking_time)), np.zeros(len(settle_time))))
        
        self.traj_x_ddot = np.concatenate((np.zeros(len(tracking_time)), np.zeros(len(settle_time))))
        self.traj_y_ddot = np.concatenate((np.zeros(len(tracking_time)), np.zeros(len(settle_time))))
        self.traj_z_ddot = np.concatenate((np.zeros(len(tracking_time)), np.zeros(len(settle_time))))
        self.traj_psi_ddot = np.concatenate((np.zeros(len(tracking_time)), np.zeros(len(settle_time))))
        
        self.ran_before = False
        
    def get_ctrl_action(self, sim):
        
        if not self.ran_before:
             # Trajectory definition
            TRACKING_TIME = 20
            SETTLE_TIME = 100

            tracking_time = np.arange(0, TRACKING_TIME, self.dT)
            settle_time = np.arange(TRACKING_TIME, TRACKING_TIME + SETTLE_TIME + 1, self.dT)

            time_vec = np.concatenate((tracking_time, settle_time))
            
            m = 0.05
            
            x0 = sim.get_var('x')
            y0 = sim.get_var('y')
            z0 = sim.get_var('z')
            psi0 = sim.get_var('psi')
            
            self.traj_x = np.concatenate((x0 + m*tracking_time, (x0 + m*TRACKING_TIME) * np.ones(len(settle_time))))
            self.traj_y = np.concatenate((y0 * np.ones(len(tracking_time)), y0 * np.ones(len(settle_time))))
            self.traj_z = np.concatenate((z0 * np.ones(len(tracking_time)), z0 * np.ones(len(settle_time))))
            self.traj_psi = np.concatenate((psi0 * np.ones(len(tracking_time)), psi0 * np.ones(len(settle_time))))
            
            self.traj_x_dot = np.concatenate((m * np.ones(len(tracking_time)), np.zeros(len(settle_time))))
            self.traj_y_dot = np.concatenate((np.zeros(len(tracking_time)), np.zeros(len(settle_time))))
            self.traj_z_dot = np.concatenate((np.zeros(len(tracking_time)), np.zeros(len(settle_time))))
            self.traj_psi_dot = np.concatenate((np.zeros(len(tracking_time)), np.zeros(len(settle_time))))
            
            self.traj_x_ddot = np.concatenate((np.zeros(len(tracking_time)), np.zeros(len(settle_time))))
            self.traj_y_ddot = np.concatenate((np.zeros(len(tracking_time)), np.zeros(len(settle_time))))
            self.traj_z_ddot = np.concatenate((np.zeros(len(tracking_time)), np.zeros(len(settle_time))))
            self.traj_psi_ddot = np.concatenate((np.zeros(len(tracking_time)), np.zeros(len(settle_time))))
           
            self.env = gp.Env(empty=True)
            self.env.setParam('OutputFlag', 0)
            self.env.setParam('LogToConsole', 0)
            self.env.start()

            self.m = gp.Model(env=self.env)

            self.mu = self.m.addMVar(shape=(self.num_inputs, 1),
                                     lb=-GRB.INFINITY, ub=GRB.INFINITY)

            self.th_cbf_constraint = self.m.addConstr(0 == 0)
            self.ph_cbf_constraint = self.m.addConstr(0 == 0)
            self.ps_cbf_constraint = self.m.addConstr(0 == 0)

            self.gamma_th = 1
            self.gamma_ph = 1
            self.gamma_ps = 1
            
            self.ran_before = True

        sim.start_timer()
        
        n = sim.get_current_timestep()

        # Extract state variables
        x = sim.get_var('x')
        y = sim.get_var('y')
        z = sim.get_var('z')

        phi = sim.get_var('phi')
        theta = sim.get_var('theta')
        psi = sim.get_var('psi')
        
        v_x__b = sim.get_var('vx')
        v_y__b = sim.get_var('vy')
        v_z__b = sim.get_var('vz')

        w_x__b = sim.get_var('wx')
        w_y__b = sim.get_var('wy')
        w_z__b = sim.get_var('wz')

        x_dot = sim.get_var_dot('x')
        y_dot = sim.get_var_dot('y')
        z_dot = sim.get_var_dot('z')

        phi_dot = sim.get_var_dot('psi')
        theta_dot = sim.get_var_dot('theta')
        psi_dot = sim.get_var_dot('psi')

        ## Feedback-linearized tracking controller

        A = np.array([[v_y__b*(- np.cos(phi)*np.cos(psi)*psi_dot + np.sin(phi)*np.sin(psi)*phi_dot + ((np.cos(phi)*np.sin(psi) - np.cos(psi)*np.sin(phi)*np.sin(theta))*(D_vxy__CB*I_x + m_RB*m_z*r_z_gb__b*v_z__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2) + np.cos(phi)*np.cos(psi)*np.sin(theta)*phi_dot + np.cos(psi)*np.cos(theta)*np.sin(phi)*theta_dot - np.sin(phi)*np.sin(psi)*np.sin(theta)*psi_dot) + w_z__b*(((I_x*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2) - (m_RB*r_z_gb__b*(I_y*w_y__b + m_RB*r_z_gb__b*v_x__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2))*(np.cos(phi)*np.sin(psi) - np.cos(psi)*np.sin(phi)*np.sin(theta)) + np.cos(psi)*np.cos(theta)*((I_y*(m_y*v_y__b - m_RB*r_z_gb__b*w_x__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2) + (m_RB*r_z_gb__b*(I_x*w_x__b - m_RB*r_z_gb__b*v_y__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2))) - v_z__b*((D_vz__CB*(np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.sin(theta)))/m_z - np.cos(phi)*np.sin(psi)*phi_dot - np.cos(psi)*np.sin(phi)*psi_dot + np.cos(psi)*np.sin(phi)*np.sin(theta)*phi_dot + np.cos(phi)*np.sin(psi)*np.sin(theta)*psi_dot - np.cos(phi)*np.cos(psi)*np.cos(theta)*theta_dot + (m_RB*r_z_gb__b*(m_y*v_y__b - m_RB*r_z_gb__b*w_x__b)*(np.cos(phi)*np.sin(psi) - np.cos(psi)*np.sin(phi)*np.sin(theta)))/(I_x*m_y - m_RB**2*r_z_gb__b**2) - (m_RB*r_z_gb__b*np.cos(psi)*np.cos(theta)*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2)) - w_x__b*(((m_y*v_y__b - m_RB*r_z_gb__b*w_x__b)*(np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.sin(theta)))/m_z + ((np.cos(phi)*np.sin(psi) - np.cos(psi)*np.sin(phi)*np.sin(theta))*(I_x*m_z*v_z__b - D_omega_xy__CB*m_RB*r_z_gb__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2) + (I_z*m_RB*r_z_gb__b*np.cos(psi)*np.cos(theta)*w_z__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2)) + w_y__b*(((m_x*v_x__b + m_RB*r_z_gb__b*w_y__b)*(np.sin(phi)*np.sin(psi) + np.cos(phi)*np.cos(psi)*np.sin(theta)))/m_z - (np.cos(psi)*np.cos(theta)*(I_y*m_z*v_z__b - D_omega_xy__CB*m_RB*r_z_gb__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2) + (I_z*m_RB*r_z_gb__b*w_z__b*(np.cos(phi)*np.sin(psi) - np.cos(psi)*np.sin(phi)*np.sin(theta)))/(I_x*m_y - m_RB**2*r_z_gb__b**2)) - v_x__b*(np.cos(theta)*np.sin(psi)*psi_dot + np.cos(psi)*np.sin(theta)*theta_dot + (np.cos(psi)*np.cos(theta)*(D_vxy__CB*I_y + m_RB*m_z*r_z_gb__b*v_z__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2)) + (f_g*m_RB*r_z_gb__b**2*np.cos(theta)*np.sin(phi)*(np.cos(phi)*np.sin(psi) - np.cos(psi)*np.sin(phi)*np.sin(theta)))/(I_x*m_y - m_RB**2*r_z_gb__b**2) + (f_g*m_RB*r_z_gb__b**2*np.cos(psi)*np.cos(theta)*np.sin(theta))/(I_y*m_x - m_RB**2*r_z_gb__b**2)],
                    [v_z__b*((D_vz__CB*(np.cos(psi)*np.sin(phi) - np.cos(phi)*np.sin(psi)*np.sin(theta)))/m_z - np.cos(phi)*np.cos(psi)*phi_dot + np.sin(phi)*np.sin(psi)*psi_dot + np.cos(phi)*np.cos(psi)*np.sin(theta)*psi_dot + np.cos(phi)*np.cos(theta)*np.sin(psi)*theta_dot - np.sin(phi)*np.sin(psi)*np.sin(theta)*phi_dot + (m_RB*r_z_gb__b*(m_y*v_y__b - m_RB*r_z_gb__b*w_x__b)*(np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi)*np.sin(theta)))/(I_x*m_y - m_RB**2*r_z_gb__b**2) + (m_RB*r_z_gb__b*np.cos(theta)*np.sin(psi)*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2)) - w_z__b*(((I_x*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2) - (m_RB*r_z_gb__b*(I_y*w_y__b + m_RB*r_z_gb__b*v_x__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2))*(np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi)*np.sin(theta)) - np.cos(theta)*np.sin(psi)*((I_y*(m_y*v_y__b - m_RB*r_z_gb__b*w_x__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2) + (m_RB*r_z_gb__b*(I_x*w_x__b - m_RB*r_z_gb__b*v_y__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2))) - v_y__b*(np.cos(psi)*np.sin(phi)*phi_dot + np.cos(phi)*np.sin(psi)*psi_dot + ((np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi)*np.sin(theta))*(D_vxy__CB*I_x + m_RB*m_z*r_z_gb__b*v_z__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2) - np.cos(phi)*np.sin(psi)*np.sin(theta)*phi_dot - np.cos(psi)*np.sin(phi)*np.sin(theta)*psi_dot - np.cos(theta)*np.sin(phi)*np.sin(psi)*theta_dot) + w_x__b*(((m_y*v_y__b - m_RB*r_z_gb__b*w_x__b)*(np.cos(psi)*np.sin(phi) - np.cos(phi)*np.sin(psi)*np.sin(theta)))/m_z + ((np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi)*np.sin(theta))*(I_x*m_z*v_z__b - D_omega_xy__CB*m_RB*r_z_gb__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2) - (I_z*m_RB*r_z_gb__b*np.cos(theta)*np.sin(psi)*w_z__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2)) - w_y__b*(((m_x*v_x__b + m_RB*r_z_gb__b*w_y__b)*(np.cos(psi)*np.sin(phi) - np.cos(phi)*np.sin(psi)*np.sin(theta)))/m_z + (np.cos(theta)*np.sin(psi)*(I_y*m_z*v_z__b - D_omega_xy__CB*m_RB*r_z_gb__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2) + (I_z*m_RB*r_z_gb__b*w_z__b*(np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi)*np.sin(theta)))/(I_x*m_y - m_RB**2*r_z_gb__b**2)) - v_x__b*(- np.cos(psi)*np.cos(theta)*psi_dot + np.sin(psi)*np.sin(theta)*theta_dot + (np.cos(theta)*np.sin(psi)*(D_vxy__CB*I_y + m_RB*m_z*r_z_gb__b*v_z__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2)) - (f_g*m_RB*r_z_gb__b**2*np.cos(theta)*np.sin(phi)*(np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi)*np.sin(theta)))/(I_x*m_y - m_RB**2*r_z_gb__b**2) + (f_g*m_RB*r_z_gb__b**2*np.cos(theta)*np.sin(psi)*np.sin(theta))/(I_y*m_x - m_RB**2*r_z_gb__b**2)],
                    [w_x__b*((np.cos(theta)*np.sin(phi)*(I_x*m_z*v_z__b - D_omega_xy__CB*m_RB*r_z_gb__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2) - (np.cos(phi)*np.cos(theta)*(m_y*v_y__b - m_RB*r_z_gb__b*w_x__b))/m_z + (I_z*m_RB*r_z_gb__b*np.sin(theta)*w_z__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2)) + w_y__b*((np.sin(theta)*(I_y*m_z*v_z__b - D_omega_xy__CB*m_RB*r_z_gb__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2) + (np.cos(phi)*np.cos(theta)*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/m_z - (I_z*m_RB*r_z_gb__b*np.cos(theta)*np.sin(phi)*w_z__b)/(I_x*m_y - m_RB**2*r_z_gb__b**2)) - v_x__b*(np.cos(theta)*theta_dot - (np.sin(theta)*(D_vxy__CB*I_y + m_RB*m_z*r_z_gb__b*v_z__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2)) - w_z__b*(np.cos(theta)*np.sin(phi)*((I_x*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2) - (m_RB*r_z_gb__b*(I_y*w_y__b + m_RB*r_z_gb__b*v_x__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2)) + (np.sin(theta)*(I_y*m_y*v_y__b - m_RB**2*r_z_gb__b**2*v_y__b + I_x*m_RB*r_z_gb__b*w_x__b - I_y*m_RB*r_z_gb__b*w_x__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2)) - v_z__b*(np.cos(theta)*np.sin(phi)*phi_dot + np.cos(phi)*np.sin(theta)*theta_dot + (D_vz__CB*np.cos(phi)*np.cos(theta))/m_z + (m_RB*r_z_gb__b*np.sin(theta)*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2) - (m_RB*r_z_gb__b*np.cos(theta)*np.sin(phi)*(m_y*v_y__b - m_RB*r_z_gb__b*w_x__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2)) - v_y__b*(- np.cos(phi)*np.cos(theta)*phi_dot + np.sin(phi)*np.sin(theta)*theta_dot + (np.cos(theta)*np.sin(phi)*(D_vxy__CB*I_x + m_RB*m_z*r_z_gb__b*v_z__b))/(I_x*m_y - m_RB**2*r_z_gb__b**2)) - (f_g*m_RB*r_z_gb__b**2*np.sin(theta)**2)/(I_y*m_x - m_RB**2*r_z_gb__b**2) + (f_g*m_RB*r_z_gb__b**2*np.sin(phi)**2*(np.sin(theta)**2 - 1))/(I_x*m_y - m_RB**2*r_z_gb__b**2)],
                    [w_y__b*((np.cos(phi)*phi_dot)/np.cos(theta) + (np.cos(phi)*(I_x*w_x__b - m_RB*r_z_gb__b*v_y__b))/(I_z*np.cos(theta)) - (np.sin(phi)*(D_omega_xy__CB*m_x - m_RB*m_z*r_z_gb__b*v_z__b))/(np.cos(theta)*(I_y*m_x - m_RB**2*r_z_gb__b**2)) - (np.sin(phi)*np.sin(theta)*theta_dot)/(np.sin(theta)**2 - 1)) - w_z__b*((np.sin(phi)*phi_dot)/np.cos(theta) + (D_omega_z__CB*np.cos(phi))/(I_z*np.cos(theta)) - (np.cos(phi)*np.sin(theta)*theta_dot)/np.cos(theta)**2 + (np.sin(phi)*(I_x*m_x*w_x__b - m_RB**2*r_z_gb__b**2*w_x__b - m_RB*m_x*r_z_gb__b*v_y__b + m_RB*m_y*r_z_gb__b*v_y__b))/(np.cos(theta)*(I_y*m_x - m_RB**2*r_z_gb__b**2))) - v_x__b*((np.cos(phi)*(m_y*v_y__b - m_RB*r_z_gb__b*w_x__b))/(I_z*np.cos(theta)) - (np.sin(phi)*(m_x*m_z*v_z__b + D_vxy__CB*m_RB*r_z_gb__b))/(np.cos(theta)*(I_y*m_x - m_RB**2*r_z_gb__b**2))) - w_x__b*((np.cos(phi)*(I_y*w_y__b + m_RB*r_z_gb__b*v_x__b))/(I_z*np.cos(theta)) - (I_z*m_x*np.sin(phi)*w_z__b)/(np.cos(theta)*(I_y*m_x - m_RB**2*r_z_gb__b**2))) + (np.cos(phi)*v_y__b*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/(I_z*np.cos(theta)) - (m_x*np.sin(phi)*v_z__b*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/(np.cos(theta)*(I_y*m_x - m_RB**2*r_z_gb__b**2)) - (f_g*m_x*r_z_gb__b*np.sin(phi)*np.sin(theta))/(np.cos(theta)*(I_y*m_x - m_RB**2*r_z_gb__b**2))]
                    ])
        
        Binv = np.array([[(np.cos(psi)*np.cos(theta)*(I_y*m_x - m_RB**2*r_z_gb__b**2))/(I_y - m_RB*r_z_gb__b*r_z_tg__b), (np.cos(theta)*np.sin(psi)*(I_y*m_x - m_RB**2*r_z_gb__b**2))/(I_y - m_RB*r_z_gb__b*r_z_tg__b), -(np.sin(theta)*(I_y*m_x - m_RB**2*r_z_gb__b**2))/(I_y - m_RB*r_z_gb__b*r_z_tg__b), 0],
                        [-((I_x*m_y - m_RB**2*r_z_gb__b**2)*(np.cos(phi)*np.sin(psi) - np.cos(psi)*np.sin(phi)*np.sin(theta)))/(I_x - m_RB*r_z_gb__b*r_z_tg__b), ((I_x*m_y - m_RB**2*r_z_gb__b**2)*(np.cos(phi)*np.cos(psi) + np.sin(phi)*np.sin(psi)*np.sin(theta)))/(I_x - m_RB*r_z_gb__b*r_z_tg__b), (np.cos(theta)*np.sin(phi)*(I_x*m_y - m_RB**2*r_z_gb__b**2))/(I_x - m_RB*r_z_gb__b*r_z_tg__b), 0],
                        [m_z*np.sin(phi)*np.sin(psi) + m_z*np.cos(phi)*np.cos(psi)*np.sin(theta), m_z*np.cos(phi)*np.sin(psi)*np.sin(theta) - m_z*np.cos(psi)*np.sin(phi), m_z*np.cos(phi)*np.cos(theta), 0],
                        [(I_z*np.cos(psi)*np.cos(theta)*np.sin(phi)*(m_RB*r_z_gb__b - m_x*r_z_tg__b))/(np.cos(phi)*(I_y - m_RB*r_z_gb__b*r_z_tg__b)), (I_z*np.cos(theta)*np.sin(phi)*np.sin(psi)*(m_RB*r_z_gb__b - m_x*r_z_tg__b))/(np.cos(phi)*(I_y - m_RB*r_z_gb__b*r_z_tg__b)), -(I_z*np.sin(phi)*np.sin(theta)*(m_RB*r_z_gb__b - m_x*r_z_tg__b))/(np.cos(phi)*(I_y - m_RB*r_z_gb__b*r_z_tg__b)), (I_z*np.cos(theta))/np.cos(phi)]
                        ])
        
        zeta1 = np.array([[x],
                          [y],
                          [z],
                          [psi]])
        zeta2 = np.array([[x_dot],
                          [y_dot],
                          [z_dot],
                          [psi_dot]])

        yd = np.array([[self.traj_x[n]],
                       [self.traj_y[n]],
                       [self.traj_z[n]],
                       [self.traj_psi[n]]])
        yd_dot = np.array([[self.traj_x_dot[n]],
                           [self.traj_y_dot[n]],
                           [self.traj_z_dot[n]],
                           [self.traj_psi_dot[n]]])
        yd_ddot = np.array([[self.traj_x_ddot[n]],
                            [self.traj_y_ddot[n]],
                            [self.traj_z_ddot[n]],
                            [self.traj_psi_ddot[n]]])
        
        e1 = zeta1 - yd
        e2 = zeta2 - yd_dot
        
        k1 = np.array([1, 1, 10, 1]).reshape((4,1))
        k2 = np.array([1, 1, 10, 10]).reshape((4,1))

        q = -k1 * e1.reshape((4,1)) - k2 * e2.reshape((4,1)) + yd_ddot
        
        k_x = Binv @ (q - A)

        ## Control barrier function
        
        # theta
        h_th = 1/2 * (-theta**2 + self.theta_limit**2)
        psi1_th = - theta*(np.cos(phi)*w_y__b - np.sin(phi)*w_z__b) + self.gamma_th*h_th
        
        lfpsi1_th = theta*(np.cos(phi)*w_z__b + np.sin(phi)*w_y__b)*(w_x__b + np.cos(phi)*np.tan(theta)*w_z__b + np.sin(phi)*np.tan(theta)*w_y__b) - (np.cos(phi)*w_y__b - np.sin(phi)*w_z__b)*(np.cos(phi)*w_y__b - np.sin(phi)*w_z__b + self.gamma_th*theta) + np.cos(phi)*theta*(w_z__b*((m_x*(I_x*w_x__b - m_RB*r_z_gb__b*v_y__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2) + (m_RB*r_z_gb__b*(m_y*v_y__b - m_RB*r_z_gb__b*w_x__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2)) - v_x__b*((D_vxy__CB*m_RB*r_z_gb__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2) + (m_x*m_z*v_z__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2)) + w_y__b*((D_omega_xy__CB*m_x)/(I_y*m_x - m_RB**2*r_z_gb__b**2) - (m_RB*m_z*r_z_gb__b*v_z__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2)) + (m_x*v_z__b*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2) - (I_z*m_x*w_x__b*w_z__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2) + (f_g*m_x*r_z_gb__b*np.sin(theta))/((I_y*m_x - m_RB**2*r_z_gb__b**2)*(np.cos(theta)**2 + np.sin(theta)**2))) - np.sin(phi)*theta*((w_x__b*(I_y*w_y__b + m_RB*r_z_gb__b*v_x__b))/I_z - (w_y__b*(I_x*w_x__b - m_RB*r_z_gb__b*v_y__b))/I_z - (v_y__b*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/I_z + (v_x__b*(m_y*v_y__b - m_RB*r_z_gb__b*w_x__b))/I_z + (D_omega_z__CB*w_z__b)/I_z)
        lgpsi1_th = np.array(
            [np.cos(phi)*theta*((m_RB*r_z_gb__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2) - (m_x*r_z_tg__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2)), 0, 0, (np.sin(phi)*theta)/I_z]
        ).reshape((1,4))

        self.m.remove(self.th_cbf_constraint)
        self.th_cbf_constraint = self.m.addConstr(lfpsi1_th + lgpsi1_th @ self.mu >= -self.gamma_th*psi1_th, "th_cbf")

        # phi
        h_ph = 1/2 * (-phi**2 + self.phi_limit**2)
        psi1_ph = - phi*(w_x__b + np.cos(phi)*np.tan(theta)*w_z__b + np.sin(phi)*np.tan(theta)*w_y__b) + self.gamma_ph * h_ph

        lfpsi1_ph = - phi*(w_x__b + np.cos(phi)*np.tan(theta)*w_z__b + np.sin(phi)*np.tan(theta)*w_y__b) - self.gamma_ph*(phi**2/2 - self.phi_limit**2/2)
        lgpsi1_ph = np.array(
            [np.sin(phi)*np.tan(theta)*phi*((m_RB*r_z_gb__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2) - (m_x*r_z_tg__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2)), -phi*((m_RB*r_z_gb__b)/(I_x*m_y - m_RB**2*r_z_gb__b**2) - (m_y*r_z_tg__b)/(I_x*m_y - m_RB**2*r_z_gb__b**2)), 0, -(np.cos(phi)*np.tan(theta)*phi)/I_z]
        ).reshape((1,4))

        self.m.remove(self.ph_cbf_constraint)
        self.ph_cbf_constraint = self.m.addConstr(lfpsi1_ph + lgpsi1_ph @ self.mu >= -self.gamma_ph*psi1_ph, "ph_cbf")

        # psi
        h_ps = 1/2 * (-psi**2 + self.psi_limit**2)
        psi1_ps = - psi*((np.cos(phi)*w_z__b)/np.cos(theta) + (np.sin(phi)*w_y__b)/np.cos(theta)) + self.gamma_ph * h_ps

        lfpsi1_ps = (np.cos(phi)*psi*((w_x__b*(I_y*w_y__b + m_RB*r_z_gb__b*v_x__b))/I_z - (w_y__b*(I_x*w_x__b - m_RB*r_z_gb__b*v_y__b))/I_z - (v_y__b*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/I_z + (v_x__b*(m_y*v_y__b - m_RB*r_z_gb__b*w_x__b))/I_z + (D_omega_z__CB*w_z__b)/I_z))/np.cos(theta) - psi*((np.cos(phi)*w_y__b)/np.cos(theta) - (np.sin(phi)*w_z__b)/np.cos(theta))*(w_x__b + np.cos(phi)*np.tan(theta)*w_z__b + np.sin(phi)*np.tan(theta)*w_y__b) - psi*(np.cos(phi)*w_y__b - np.sin(phi)*w_z__b)*((np.cos(phi)*np.sin(theta)*w_z__b)/np.cos(theta)**2 + (np.sin(phi)*np.sin(theta)*w_y__b)/np.cos(theta)**2) - ((np.cos(phi)*w_z__b)/np.cos(theta) + (np.sin(phi)*w_y__b)/np.cos(theta))*(self.gamma_ps*psi + (np.cos(phi)*w_z__b)/np.cos(theta) + (np.sin(phi)*w_y__b)/np.cos(theta)) + (np.sin(phi)*psi*(w_z__b*((m_x*(I_x*w_x__b - m_RB*r_z_gb__b*v_y__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2) + (m_RB*r_z_gb__b*(m_y*v_y__b - m_RB*r_z_gb__b*w_x__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2)) - v_x__b*((D_vxy__CB*m_RB*r_z_gb__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2) + (m_x*m_z*v_z__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2)) + w_y__b*((D_omega_xy__CB*m_x)/(I_y*m_x - m_RB**2*r_z_gb__b**2) - (m_RB*m_z*r_z_gb__b*v_z__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2)) + (m_x*v_z__b*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2) - (I_z*m_x*w_x__b*w_z__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2) + (f_g*m_x*r_z_gb__b*np.sin(theta))/((I_y*m_x - m_RB**2*r_z_gb__b**2)*(np.cos(theta)**2 + np.sin(theta)**2))))/np.cos(theta)
        lgpsi1_ps = np.array(
            [(np.sin(phi)*psi*((m_RB*r_z_gb__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2) - (m_x*r_z_tg__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2)))/np.cos(theta), 0, 0, -(np.cos(phi)*psi)/(I_z*np.cos(theta))]
        ).reshape((1,4))

        #self.m.remove(self.ps_cbf_constraint)
        #self.ps_cbf_constraint = self.m.addConstr(lfpsi1_ps + lgpsi1_ps @ self.mu >= -self.gamma_ps*psi1_ps, "ps_cbf")

        # objective
        obj = (self.mu.T - k_x.T) @ (self.mu - k_x)
        self.m.setObjective(obj)

        self.m.optimize()

        if self.m.Status == 4:
            self.m.computeIIS()

            print("\nModel is infeasible")

            # Print out the IIS constraints and variables
            print('The following constraints and variables are in the IIS:')

            print("Constraints:")
            for c in self.m.getConstrs():
                if c.IISConstr: print(f'\t{c.constrname}: {self.m.getRow(c)} {c.Sense} {c.RHS}')

            print("Variables:")
            for v in self.m.getVars():
                if v.IISLB: print(f'\t{v.varname} >= {v.LB}')
                if v.IISUB: print(f'\t{v.varname} <= {v.UB}')
                sys.exit(1)
            print()

        u = self.mu.X
        
        u_delta = u - k_x
        
        self.k_x_list.append(k_x)
        self.u_delta_list.append(u_delta)
        
        psi1_dot = lfpsi1_th + lgpsi1_th @ u.reshape((4,1))

        sim.end_timer()

        return u.reshape((4,1))
        
    def get_trajectory(self):
        return (self.traj_x,
                self.traj_y,
                self.traj_z)
    
    def get_error(self, sim):
        n = sim.get_current_timestep() + 1
        return np.array([
            sim.get_var_history('x') - self.traj_x[0:n],
            sim.get_var_history('y') - self.traj_y[0:n],
            sim.get_var_history('z') - self.traj_z[0:n],
            sim.get_var_history('psi') - self.traj_psi[0:n]
        ]).T
