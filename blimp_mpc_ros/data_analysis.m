clear
close all
clc

%% Line follower

line_cbf_1 = 'logs/cbf_line.csv';
line_cbf_2 = 'logs/cbf_line2.csv';
line_cbf_3 = 'logs/cbf_line3.csv';

line_lqr_1 = 'logs/final_testing_4_22/lqr_line.csv';
line_lqr_2 = 'logs/final_testing_4_22/lqr_line2.csv';
line_lqr_3 = 'logs/final_testing_4_22/lqr_line3.csv';

line_cbf_1_metrics = compute_metrics(line_cbf_1);
line_cbf_2_metrics = compute_metrics(line_cbf_2);
line_cbf_3_metrics = compute_metrics(line_cbf_3);
line_cbf_metrics = [line_cbf_1_metrics
                    line_cbf_2_metrics
                    line_cbf_3_metrics];

line_cbf_metrics_means = mean(line_cbf_metrics, 1)

line_lqr_1_metrics = compute_metrics(line_lqr_1);
line_lqr_2_metrics = compute_metrics(line_lqr_2);
line_lqr_3_metrics = compute_metrics(line_lqr_3);
line_lqr_metrics = [line_lqr_1_metrics
                    line_lqr_2_metrics
                    line_lqr_3_metrics];

line_lqr_metrics_means = mean(line_lqr_metrics, 1)

% plot_data(line_cbf_1, 60, 'Line Trajectory')
plot_data(line_cbf_2, 60, 'Line Trajectory')
% plot_data(line_cbf_3, 60, 'Line Trajectory')

%% Helix follower

helix_cbf_1 = 'logs/cbf_helix.csv';
helix_cbf_2 = 'logs/cbf_helix2.csv';
helix_cbf_3 = 'logs/cbf_helix3.csv';

helix_lqr_1 = 'logs/final_testing_4_22/lqr_helix.csv';
helix_lqr_2 = 'logs/final_testing_4_22/lqr_helix2.csv';
helix_lqr_3 = 'logs/final_testing_4_22/lqr_helix3.csv';

helix_cbf_1_metrics = compute_metrics(helix_cbf_1);
helix_cbf_2_metrics = compute_metrics(helix_cbf_2);
helix_cbf_3_metrics = compute_metrics(helix_cbf_3);
helix_cbf_metrics = [helix_cbf_1_metrics
                    helix_cbf_2_metrics
                    helix_cbf_3_metrics];

helix_cbf_metrics_means = mean(helix_cbf_metrics, 1)

helix_lqr_1_metrics = compute_metrics(helix_lqr_1);
helix_lqr_2_metrics = compute_metrics(helix_lqr_2);
helix_lqr_3_metrics = compute_metrics(helix_lqr_3);
helix_lqr_metrics = [helix_lqr_1_metrics
                    helix_lqr_2_metrics
                    helix_lqr_3_metrics];

helix_lqr_metrics_means = mean(helix_lqr_metrics, 1)

plot_data(helix_cbf_1, 60, 'Helix Trajectory')
% plot_data(helix_cbf_2, 60, 'Helix Trajectory')
% plot_data(helix_cbf_3, 60, 'Helix Trajectory')

%% Triangle follower

triangle_cbf_1 = 'logs/cbf_triangle.csv';
triangle_cbf_2 = 'logs/cbf_triangle2.csv';
triangle_cbf_3 = 'logs/cbf_triangle3.csv';

triangle_lqr_1 = 'logs/final_testing_4_22/lqr_triangle.csv';
triangle_lqr_2 = 'logs/final_testing_4_22/lqr_triangle2.csv';
triangle_lqr_3 = 'logs/final_testing_4_22/lqr_triangle3.csv';

triangle_cbf_1_metrics = compute_metrics(triangle_cbf_1);
triangle_cbf_2_metrics = compute_metrics(triangle_cbf_2);
triangle_cbf_3_metrics = compute_metrics(triangle_cbf_3);
triangle_cbf_metrics = [triangle_cbf_1_metrics
                    triangle_cbf_2_metrics
                    triangle_cbf_3_metrics];

triangle_cbf_metrics_means = mean(triangle_cbf_metrics, 1)

triangle_lqr_1_metrics = compute_metrics(triangle_lqr_1);
triangle_lqr_2_metrics = compute_metrics(triangle_lqr_2);
triangle_lqr_3_metrics = compute_metrics(triangle_lqr_3);
triangle_lqr_metrics = [triangle_lqr_1_metrics
                    triangle_lqr_2_metrics
                    triangle_lqr_3_metrics];

triangle_lqr_metrics_means = mean(triangle_lqr_metrics, 1)

plot_data(triangle_cbf_1, 180, 'Triangle Trajectory')
% plot_data(triangle_cbf_2, 180, 'Triangle Trajectory')
% plot_data(triangle_cbf_3, 180, 'Triangle Trajectory')

function [time, ...
          state, ...
          state_dot, ...
          u, ...
          ref, ...
          error, ...
          solve_time] ...
          = load_data(file)
    dataset = readmatrix(file);
    time = dataset(:, 1);
    state = dataset(:, 2:13);
    state_dot = dataset(:, 14:25);
    u = dataset(:, 26:29);
    ref = dataset(:, 30:33);
    error = ref - state(:, [7 8 9 12]);
    solve_time = dataset(:, 38);
end

function metrics = compute_metrics(file)
    
    [t, x, xd, u, ref, err, st] = load_data(file);

    control_effort = 0;
    for i = 1:length(u)
        control_effort = control_effort + norm(u(i, :));
    end
    control_effort = control_effort / length(u);

    pos_err = 0;
    yaw_err = 0;
    for i = 1:length(err)
        pos_err = pos_err + norm(err(i, 1:3));
        yaw_err = yaw_err + norm(err(i, 4));
    end
    pos_err = pos_err / length(err);
    yaw_err = yaw_err / length(err) * 180/pi;

    roll_osc_rms = rms(x(:, 10))*180/pi;
    pitch_osc_rms = rms(x(:, 11))*180/pi;

    solve_time = mean(st)/1e9;
    
    metrics = [control_effort pos_err yaw_err roll_osc_rms pitch_osc_rms solve_time];
end

function display_data(filename, name)
    dataset = readmatrix(filename);
    [ce, err, roll, pitch, st] = compute_metrics(dataset);
    
    disp("Dataset " + name + ":" + filename)
    disp("Control effort: " + ce * 1e3 + " mN")
    disp("Tracking error: " + err + " m")
    disp("Roll: " + roll*180/pi + " degrees")
    disp("Pitch: " + pitch*180/pi + " degrees")
    disp("Solve time: " + st/1e6 + " ms")
    disp(" ")
end

function plot_data(cbf_file, time_end, plot_title)
    [t, cbf_x, ~, ~, cbf_ref, cbf_err, ~] = load_data(cbf_file);

    % Trajectory tracking
    figure
    
    subplot(121)

    colororder(["b", "g", "m"])
    plot(t, cbf_x(:, 7:9), 'LineWidth', 2, 'LineStyle', '-')
    hold on
    plot(t, cbf_ref(:, 1:3), 'LineWidth', 2, 'LineStyle', ':')

    xlim([0 time_end])
    xlabel('Time (sec)')
    ylabel('Position (m)')
    legend('x', 'y', 'z', ...
           'x_{ref}', 'y_{ref}','z_{ref}')
    title(plot_title + ": Position tracking")
    pbaspect([1 1 1])

    % Attitude oscillations
    subplot(122)
    plot(t, cbf_x(:, 10) * 180/pi, 'LineWidth', 0.75, 'LineStyle', '-', 'Color', '#D95319')
    hold on
    plot(t, cbf_x(:, 11) * 180/pi, 'LineWidth', 0.75, 'LineStyle', '-', 'Color', '#7E2F8E')
    plot(t, cbf_x(:, 12) * 180/pi, 'LineWidth', 1, 'LineStyle', '-', 'Color', '#77AC30')
    plot(t, cbf_ref(:, 4) * 180/pi, 'LineWidth', 1.25, 'LineStyle', ':', 'Color', '#77AC30')
    
    ylim([-10 10])
    xlabel('Time (sec)')
    ylabel('Angle (deg)')
    title(plot_title + ": Roll, Pitch, and Yaw")
    xlim([0 time_end])
    legend('phi', 'theta', 'psi', 'psi_{ref}')
    
    set(gcf, 'color', 'white')
    pbaspect([1 1 1])
end

function plot_3d(cbf_file, lqr_file, plot_title)
    [~, cbf_x, ~, ~, ref, ~, ~] = load_data(cbf_file);
    [~, lqr_x, ~, ~, ~, ~, ~] = load_data(lqr_file);

    figure
    plot3(cbf_x(:, 7), cbf_x(:, 8), cbf_x(:, 9), 'LineWidth', 2, 'Color', 'b');
    hold on
    plot3(lqr_x(:, 7), cbf_x(:, 8), cbf_x(:, 9), 'LineWidth', 1, 'Color', 'm');
    plot3(ref(:, 1), ref(:, 2), ref(:, 3), 'LineWidth', 1, 'Color', 'g', 'LineStyle', '--');
    
    title(plot_title)
    xlabel('x')
    ylabel('y')
    zlabel('z')
    pbaspect([1 1 1])
    set(gca, 'ydir', 'reverse')
    set(gca, 'zdir', 'reverse')
    set(gcf, 'color', 'white')

    legend('FBL+CBF', 'LQR', 'Reference')
end