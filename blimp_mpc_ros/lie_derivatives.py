import numpy as np
import csv
from utilities import *
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from Trajectories import Trajectories

with open('logs/cbf_triangle.csv') as file:
    csvfile = csv.reader(file)

    lines = []
    first = True
    for line in csvfile:
        if first:
            first = False
            continue
        lines.append(list([float(i) for i in line[:-1]]))
        
    data = np.array(lines)

    time = data[:, 0]
    state = data[:, 1:13]
    state_dot = data[:, 13:25]
    u = data[:, 25:29]
    xref = data[:, 29:33]

    v_x__b_v = state[:, 0]
    v_y__b_v = state[:, 1]
    v_z__b_v = state[:, 2]
    w_x__b_v = state[:, 3]
    w_y__b_v = state[:, 4]
    w_z__b_v = state[:, 5]
    x_v      = state[:, 6]
    y_v      = state[:, 7]
    z_v      = state[:, 8]
    phi_v    = state[:, 9]
    theta_v  = state[:, 10]
    psi_v    = state[:, 11]

    v_x__b_dot_v = state_dot[:, 0]
    v_y__b_dot_v = state_dot[:, 1]
    v_z__b_dot_v = state_dot[:, 2]
    w_x__b_dot_v = state_dot[:, 3]
    w_y__b_dot_v = state_dot[:, 4]
    w_z__b_dot_v = state_dot[:, 5]
    x_dot_v      = state_dot[:, 6]
    y_dot_v      = state_dot[:, 7]
    z_dot_v      = state_dot[:, 8]
    phi_dot_v    = state_dot[:, 9]
    theta_dot_v  = state_dot[:, 10]
    psi_dot_v    = state_dot[:, 11]

    psi1_th_dot_v = np.empty(len(time))
    psi1_th_v = np.empty(len(time))

    trajectory = Trajectories.get_triangle(x_v[0], y_v[0], z_v[0], psi_v[0], 0.05)

    traj_x         = trajectory[0]
    traj_y         = trajectory[1]
    traj_z         = trajectory[2]
    traj_psi       = trajectory[3]

    traj_x_dot     = trajectory[4]
    traj_y_dot     = trajectory[5]
    traj_z_dot     = trajectory[6]
    traj_psi_dot   = trajectory[7]
    
    traj_x_ddot    = trajectory[8]
    traj_y_ddot    = trajectory[9]
    traj_z_ddot    = trajectory[10]
    traj_psi_ddot  = trajectory[11]

    theta_limit = 5 * np.pi/180
    phi_limit = 5 * np.pi/180
    
    k1 = np.array([1.1, 1.1, 5, 0.4]).reshape((4,1))
    k2 = np.array([0.5, 0.5, 5, 0.4]).reshape((4,1))

    gamma_th = 1
    gamma_ph = 1
    
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.setParam('LogToConsole', 0)
    env.start()

    m = gp.Model(env=env)

    mu = m.addMVar(shape=(4, 1),
                                lb=-GRB.INFINITY, ub=GRB.INFINITY)
    
    mu_v = np.empty((4, len(time)))

    th_cbf_constraint = m.addConstr(0 == 0)
    ph_cbf_constraint = m.addConstr(0 == 0)
    ps_cbf_constraint = m.addConstr(0 == 0)

    for n in range(2, len(time)):
        v_x__b = v_x__b_v[n]
        v_y__b = v_y__b_v[n]
        v_z__b = v_z__b_v[n]
        w_x__b = w_x__b_v[n]
        w_y__b = w_y__b_v[n]
        w_z__b = w_z__b_v[n]
        x      = x_v[n]
        y      = y_v[n]
        z      = z_v[n]
        phi    = phi_v[n]
        theta  = theta_v[n]
        psi    = psi_v[n]
        x_dot      = x_dot_v[n]
        y_dot      = y_dot_v[n]
        z_dot      = z_dot_v[n]
        phi_dot    = phi_dot_v[n]
        theta_dot  = theta_dot_v[n]
        psi_dot    = psi_dot_v[n]
        
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

        yd = np.array([[traj_x[n]],
                       [traj_y[n]],
                       [traj_z[n]],
                       [traj_psi[n]]])
        yd_dot = np.array([[traj_x_dot[n]],
                           [traj_y_dot[n]],
                           [traj_z_dot[n]],
                           [traj_psi_dot[n]]])
        yd_ddot = np.array([[traj_x_ddot[n]],
                            [traj_y_ddot[n]],
                            [traj_z_ddot[n]],
                            [traj_psi_ddot[n]]])
        
        e1 = zeta1 - yd
        e2 = zeta2 - yd_dot
        
        q = -k1 * e1.reshape((4,1)) - k2 * e2.reshape((4,1)) + yd_ddot
        
        k_x = Binv @ (q - A)

        ## Control barrier function
        
        h_th = 1/2 * (-theta**2 + theta_limit**2)
        psi1_th = - theta*(np.cos(phi)*w_y__b - np.sin(phi)*w_z__b) + gamma_th*h_th
        
        lfpsi1_th = theta*(np.cos(phi)*w_z__b + np.sin(phi)*w_y__b)*(w_x__b + np.cos(phi)*np.tan(theta)*w_z__b + np.sin(phi)*np.tan(theta)*w_y__b) - (np.cos(phi)*w_y__b - np.sin(phi)*w_z__b)*(np.cos(phi)*w_y__b - np.sin(phi)*w_z__b + gamma_th*theta) + np.cos(phi)*theta*(w_z__b*((m_x*(I_x*w_x__b - m_RB*r_z_gb__b*v_y__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2) + (m_RB*r_z_gb__b*(m_y*v_y__b - m_RB*r_z_gb__b*w_x__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2)) - v_x__b*((D_vxy__CB*m_RB*r_z_gb__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2) + (m_x*m_z*v_z__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2)) + w_y__b*((D_omega_xy__CB*m_x)/(I_y*m_x - m_RB**2*r_z_gb__b**2) - (m_RB*m_z*r_z_gb__b*v_z__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2)) + (m_x*v_z__b*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/(I_y*m_x - m_RB**2*r_z_gb__b**2) - (I_z*m_x*w_x__b*w_z__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2) + (f_g*m_x*r_z_gb__b*np.sin(theta))/((I_y*m_x - m_RB**2*r_z_gb__b**2)*(np.cos(theta)**2 + np.sin(theta)**2))) - np.sin(phi)*theta*((w_x__b*(I_y*w_y__b + m_RB*r_z_gb__b*v_x__b))/I_z - (w_y__b*(I_x*w_x__b - m_RB*r_z_gb__b*v_y__b))/I_z - (v_y__b*(m_x*v_x__b + m_RB*r_z_gb__b*w_y__b))/I_z + (v_x__b*(m_y*v_y__b - m_RB*r_z_gb__b*w_x__b))/I_z + (D_omega_z__CB*w_z__b)/I_z)
        lgpsi1_th = np.array(
            [np.cos(phi)*theta*((m_RB*r_z_gb__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2) - (m_x*r_z_tg__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2)), 0, 0, (np.sin(phi)*theta)/I_z]
        ).reshape((1,4))

        m.remove(th_cbf_constraint)
        th_cbf_constraint = m.addConstr(lfpsi1_th + lgpsi1_th @ mu >= -gamma_th*psi1_th, "th_cbf")

        h_ph = 1/2 * (-phi**2 + phi_limit**2)
        psi1_ph = - phi*(w_x__b + np.cos(phi)*np.tan(theta)*w_z__b + np.sin(phi)*np.tan(theta)*w_y__b) + gamma_ph * h_ph

        lfpsi1_ph = - phi*(w_x__b + np.cos(phi)*np.tan(theta)*w_z__b + np.sin(phi)*np.tan(theta)*w_y__b) - gamma_ph*(phi**2/2 - phi_limit**2/2)
        lgpsi1_ph = np.array(
            [np.sin(phi)*np.tan(theta)*phi*((m_RB*r_z_gb__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2) - (m_x*r_z_tg__b)/(I_y*m_x - m_RB**2*r_z_gb__b**2)), -phi*((m_RB*r_z_gb__b)/(I_x*m_y - m_RB**2*r_z_gb__b**2) - (m_y*r_z_tg__b)/(I_x*m_y - m_RB**2*r_z_gb__b**2)), 0, -(np.cos(phi)*np.tan(theta)*phi)/I_z]
        ).reshape((1,4))

        m.remove(ph_cbf_constraint)
        ph_cbf_constraint = m.addConstr(lfpsi1_ph + lgpsi1_ph @ mu >= -gamma_ph*psi1_ph, "ph_cbf")
        
        # objective
        obj = (mu.T - k_x.T) @ (mu - k_x)
        m.setObjective(obj)

        m.optimize()

        mu_v[:, n] = mu.X.reshape(4)
        psi1_th_v[n] = psi1_th
        psi1_th_dot_v[n] = (lfpsi1_th + lgpsi1_th @ u[n, :]).item()

        print(mu_v[:, n])
        print(u[n, :])
        break

    # plt.plot(time, traj_x)
    # plt.plot(time, traj_y)
    # plt.plot(time, traj_z)
    # plt.plot(time, traj_psi)
    # plt.show()