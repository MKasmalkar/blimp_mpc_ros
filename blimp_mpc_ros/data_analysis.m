clear
close all
clc

disp("Control effort, error, roll, pitch, solve time")

%% Line follower

line_cbf_1 = 'logs/cbf_line.csv';
line_cbf_2 = 'logs/cbf_line2.csv';
line_cbf_3 = 'logs/cbf_line3.csv';

line_pid_1 = 'logs/pid_line.csv';
line_pid_2 = 'logs/pid_line2.csv';
line_pid_3 = 'logs/pid_line3.csv';

line_cbf_1_metrics = compute_metrics(line_cbf_1);
line_cbf_2_metrics = compute_metrics(line_cbf_2);
line_cbf_3_metrics = compute_metrics(line_cbf_3);
line_cbf_metrics = [line_cbf_1_metrics
                    line_cbf_2_metrics
                    line_cbf_3_metrics];

line_cbf_metrics_means = mean(line_cbf_metrics, 1)

line_pid_1_metrics = compute_metrics(line_pid_1);
line_pid_2_metrics = compute_metrics(line_pid_2);
line_pid_3_metrics = compute_metrics(line_pid_3);
line_pid_metrics = [line_pid_1_metrics
                    line_pid_2_metrics
                    line_pid_3_metrics];

line_pid_metrics_means = mean(line_pid_metrics, 1)

%% Helix follower

helix_cbf_1 = 'logs/cbf_helix.csv';
helix_cbf_2 = 'logs/cbf_helix2.csv';
helix_cbf_3 = 'logs/cbf_helix3.csv';

helix_pid_1 = 'logs/pid_helix.csv';
helix_pid_2 = 'logs/pid_helix2.csv';
helix_pid_3 = 'logs/pid_helix3.csv';

helix_cbf_1_metrics = compute_metrics(helix_cbf_1);
helix_cbf_2_metrics = compute_metrics(helix_cbf_2);
helix_cbf_3_metrics = compute_metrics(helix_cbf_3);
helix_cbf_metrics = [helix_cbf_1_metrics
                    helix_cbf_2_metrics
                    helix_cbf_3_metrics];

helix_cbf_metrics_means = mean(helix_cbf_metrics, 1)

helix_pid_1_metrics = compute_metrics(helix_pid_1);
helix_pid_2_metrics = compute_metrics(helix_pid_2);
helix_pid_3_metrics = compute_metrics(helix_pid_3);
helix_pid_metrics = [helix_pid_1_metrics
                    helix_pid_2_metrics
                    helix_pid_3_metrics];

helix_pid_metrics_means = mean(helix_pid_metrics, 1)

%% Triangle follower

triangle_cbf = 'logs/cbf_triangle.csv';
triangle_pid = 'logs/pid_triangle.csv';

triangle_cbf_metrics = compute_metrics(triangle_cbf)
triangle_pid_metrics = compute_metrics(triangle_pid)

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

    tracking_error = 0;
    for i = 1:length(err)
        tracking_error = tracking_error + norm(err(i, :));
    end
    tracking_error = tracking_error / length(err);

    roll_osc_rms = rms(x(:, 7))*180/pi;
    pitch_osc_rms = rms(x(:, 8))*180/pi;

    solve_time = mean(st)/1e9;
    
    metrics = [control_effort tracking_error roll_osc_rms pitch_osc_rms solve_time];
end

function display_data(filename, name)
    dataset = readmatrix(filename);
    [ce, err, roll, pitch, st] = compute_metrics(dataset);
    
    disp("Dataset " + name + ":" + filename)
    disp("Control effort: " + ce * 1e3 + " mN")
    disp("Tracking error: " + err + " m")
    disp("Pk-pk roll: " + roll*180/pi + " degrees")
    disp("Pk-pk pitch: " + pitch*180/pi + " degrees")
    disp("Solve time: " + st/1e6 + " ms")
    disp(" ")
end