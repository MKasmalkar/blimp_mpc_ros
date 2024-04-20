from . operators import *
import numpy as np


## Blimp model parameters

# Height of envelope Henv = 51 cm
Henv = 0.51

# Length of envelope Denv = 77 cm
Denv = 0.77

# Height of the gondola Hgon = 4 cm
Hgon = 0.04

# Distance from CB to motor thrust application point dvt
dvt = Henv/2 + Hgon

# Measured buoyant force = 0.725 N
FB_meas = 0.725

# Mass of envelope = 38.14 g
m_env = 0.03814

# Gravitational acceleration
g = 9.8

# Actual buoyant force
FB = FB_meas + m_env * g

# Density of air = 1.161 kg/m^3
rho_air = 1.161

# Volume of blimp envelope
Venv = FB / (rho_air * g)

# Mass of markers = 13.65 g
m_mkr = 0.01365

# Mass of gondola = 58.45 g
m_gon = 0.05845

# Mass of large battery = 14.65 g
m_largebat = 0.01465

# Mass of small battery = 8.29 g
m_smallbat = 0.00829

# Mass of helium
rho_he = 0.164
m_He = rho_he * Venv

# Mass of ballast = 1.81 g
m_blst = 0.00181

# Mass of blimp
# m1 = m_env + m_He + m_gon + m_mkr + m_blst + m_largebat
m = rho_air * Venv

# Venv2 = (m_gon + m_mkr + m_blst + m_env + m_largebat) / (rho_air - rho_he)

# Center of gravity to center of buoyancy
# r_z_g__b = 0.05 is excellent! was originally 0.09705 I think
# = 0.02 is also good
r_z_gb__b = 0.05
r_gb__b = np.array([0, 0, r_z_gb__b]).T
r_z_tg__b = dvt - r_z_gb__b

## Inertia matrix

m_Ax = 0.0466
m_Ay = m_Ax
m_Az = 0.0545
m_Axy = m_Ax

I_Ax = 0.0
I_Ay = I_Ax
I_Az = 0.0
I_Axy = I_Ax

M_A_CB = np.diag([m_Ax, m_Ay, m_Az, I_Ax, I_Ay, I_Az])

m_RB = 0.1249
I_RBx = 0.005821
I_RBy = I_RBx
I_RBz = I_RBx
I_RBxy = I_RBx

M_RB_CG = np.diag([m_RB, m_RB, m_RB, I_RBx, I_RBy, I_RBz])

M_RB_CB = H(r_gb__b).T @ M_RB_CG @ H(r_gb__b)

M_CB = M_RB_CB + M_A_CB

M_CB_inv = np.linalg.inv(M_CB)

m_x = m_RB + m_Ax
m_y = m_RB + m_Ay
m_z = m_RB + m_Az

I_x = I_RBx + m_RB * r_z_gb__b**2 + I_Ax
I_y = I_RBx + m_RB * r_z_gb__b**2 + I_Ay
I_z = I_RBz + I_Az

g_acc = 9.8
fg_n = m_RB * np.array([0, 0, g_acc]).T
f_g = fg_n[2]

## Aerodynamic damping
D_vx__CB = 0.0125
D_vy__CB = D_vx__CB
D_vz__CB = 0.0480
D_vxy__CB = D_vx__CB

D_wx__CB = 0.000862
D_wy__CB = D_wx__CB
D_wz__CB = D_wx__CB
D_omega_xy__CB = D_wx__CB
D_omega_z__CB = D_wz__CB

D_CB = np.diag([D_vx__CB, D_vy__CB, D_vz__CB, D_wx__CB, D_wy__CB, D_wz__CB])
