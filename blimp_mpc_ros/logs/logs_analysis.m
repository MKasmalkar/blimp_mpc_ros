
data = csvread('log.csv');

time = data(:, 1);

x = data(:, 8);
y = data(:, 9);
z = data(:, 10);

h = scatter3(x, y, z);
xlabel('x')
ylabel('y')
zlabel('z')
waitfor(h)