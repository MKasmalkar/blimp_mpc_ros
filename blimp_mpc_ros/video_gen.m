clear
close all
clc

play_video('logs/videos_5_1/cbf_line_video.csv', 1, "Line Trajectory")

function play_video(filename, speed, plot_title)
    [tt, x, xd, u, ref, err, st] = load_data(filename);

    plot3(ref(:, 1), ref(:, 2), ref(:, 3), 'LineWidth', 2, 'Color', [0    0.4470    0.7410])
    
    set(gcf, 'color', 'white')
    set(gca, 'ydir', 'reverse')
    set(gca, 'zdir', 'reverse')
    pbaspect([1 1 1])
    title(plot_title)

    for t = 1:length(t)
        
    end
end

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