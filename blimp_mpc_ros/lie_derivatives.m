clear
close all
clc

Henv = 0.51;
Denv = 0.77;
Hgon = 0.04;
dvt = Henv/2 + Hgon;
FB_meas = 0.725;
m_env = 0.03814;
g = 9.8;
FB = FB_meas + m_env * g;
rho_air = 1.161;
Venv = FB / (rho_air * g);
m_mkr = 0.01365;
m_gon = 0.05845;
m_largebat = 0.01465;
m_smallbat = 0.00829;
rho_he = 0.164;
m_He = rho_he * Venv;
m_blst = 0.00181;
m = rho_air * Venv;
r_z_gb__b = 0.02;
r_gb__b = [0, 0, r_z_gb__b].';
r_z_tg__b = dvt - r_z_gb__b;
m_Ax = 0.0466;
m_Ay = m_Ax;
m_Az = 0.0545;
m_Axy = m_Ax;
I_Ax = 0.0;
I_Ay = I_Ax;
I_Az = 0.0;
I_Axy = I_Ax;
m_RB = 0.1249;
I_RBx = 0.005821;
I_RBy = I_RBx;
I_RBz = I_RBx;
I_RBxy = I_RBx;
m_x = m_RB + m_Ax;
m_y = m_RB + m_Ay;
m_z = m_RB + m_Az;
I_x = I_RBx + m_RB * r_z_gb__b^2 + I_Ax;
I_y = I_RBx + m_RB * r_z_gb__b^2 + I_Ay;
I_z = I_RBz + I_Az;
g_acc = 9.8;
fg_n = m_RB * [0, 0, g_acc];
f_g = fg_n(3);
D_vx__CB = 0.0125;
D_vy__CB = D_vx__CB;
D_vz__CB = 0.0480;
D_vxy__CB = D_vx__CB;
D_wx__CB = 0.000862;
D_wy__CB = D_wx__CB;
D_wz__CB = D_wx__CB;
D_omega_xy__CB = D_wx__CB;
D_omega_z__CB = D_wz__CB;


file = 'logs./cbf_triangle.csv';

dataset = readmatrix(file);
time = dataset(:, 1);
state = dataset(:, 2:13);
state_dot = dataset(:, 14:25);
u = dataset(:, 26:29);
ref = dataset(:, 30:33);
error = ref - state(:, [7 8 9 12]);
solve_time = dataset(:, 38);

v_x__b = state(:, 1);
v_y__b = state(:, 2);
v_z__b = state(:, 3);
w_x__b = state(:, 4);
w_y__b = state(:, 5);
w_z__b = state(:, 6);
x = state(:, 7);
y = state(:, 8);
z = state(:, 9);
phi = state(:, 10);
theta = state(:, 11);
psi = state(:, 12);

theta_limit = 5.*pi./180;
gamma = 1;

h = 1./2 .* (-theta.^2 + theta_limit.^2);
psi1 = - theta.*(cos(phi).*w_y__b - sin(phi).*w_z__b) + gamma.*h;

lfpsi1_th = theta.*(cos(phi).*w_z__b + sin(phi).*w_y__b).*(w_x__b + cos(phi).*tan(theta).*w_z__b + sin(phi).*tan(theta).*w_y__b) - (cos(phi).*w_y__b - sin(phi).*w_z__b).*(cos(phi).*w_y__b - sin(phi).*w_z__b + gamma.*theta) + cos(phi).*theta.*(w_z__b.*((m_x.*(I_x.*w_x__b - m_RB.*r_z_gb__b.*v_y__b))./(I_y.*m_x - m_RB.^2.*r_z_gb__b.^2) + (m_RB.*r_z_gb__b.*(m_y.*v_y__b - m_RB.*r_z_gb__b.*w_x__b))./(I_y.*m_x - m_RB.^2.*r_z_gb__b.^2)) - v_x__b.*((D_vxy__CB.*m_RB.*r_z_gb__b)./(I_y.*m_x - m_RB.^2.*r_z_gb__b.^2) + (m_x.*m_z.*v_z__b)./(I_y.*m_x - m_RB.^2.*r_z_gb__b.^2)) + w_y__b.*((D_omega_xy__CB.*m_x)./(I_y.*m_x - m_RB.^2.*r_z_gb__b.^2) - (m_RB.*m_z.*r_z_gb__b.*v_z__b)./(I_y.*m_x - m_RB.^2.*r_z_gb__b.^2)) + (m_x.*v_z__b.*(m_x.*v_x__b + m_RB.*r_z_gb__b.*w_y__b))./(I_y.*m_x - m_RB.^2.*r_z_gb__b.^2) - (I_z.*m_x.*w_x__b.*w_z__b)./(I_y.*m_x - m_RB.^2.*r_z_gb__b.^2) + (f_g.*m_x.*r_z_gb__b.*sin(theta))./((I_y.*m_x - m_RB.^2.*r_z_gb__b.^2).*(cos(theta).^2 + sin(theta).^2))) - sin(phi).*theta.*((w_x__b.*(I_y.*w_y__b + m_RB.*r_z_gb__b.*v_x__b))./I_z - (w_y__b.*(I_x.*w_x__b - m_RB.*r_z_gb__b.*v_y__b))./I_z - (v_y__b.*(m_x.*v_x__b + m_RB.*r_z_gb__b.*w_y__b))./I_z + (v_x__b.*(m_y.*v_y__b - m_RB.*r_z_gb__b.*w_x__b))./I_z + (D_omega_z__CB.*w_z__b)./I_z);
lgpsi1_th = ...
    [cos(phi).*theta.*((m_RB.*r_z_gb__b)./(I_y.*m_x - m_RB.^2.*r_z_gb__b.^2) - (m_x.*r_z_tg__b)./(I_y.*m_x - m_RB.^2.*r_z_gb__b.^2)), ...
     zeros(length(time), 1), ...
     zeros(length(time), 1), ...
     (sin(phi).*theta)./I_z];

plot(time, lfpsi1_th + diag(lgpsi1_th * u.' + gamma*psi1))
hold on
plot(time, [0 diff(psi1).'/0.05])
legend('code', 'correct')