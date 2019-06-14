%% Testing data----------------------------------------------------
clear all

% GCRR
% r = [4.8, 1, 3.7, 3.2];
% r6 = 2.5;
% theta6 = pi/4;
% N = 100;
% x = 10;
% y = 23;
% theta1 = 0.5;
% target_x = [4.8, 3.7, 3.2, 2.5, pi/4]; % [l,p,q,r6,theta6]

%% Q.J.Ge Closed path parameters------------------------------------
r = [11, 6, 8, 10];
r6 = 7;
theta6 = 0.6981;
N = 360;
x = 10;
y = 14;
theta1 = 0.1745;
target_x = [11/6, 8/6, 10/6, 7/6, pi/4]; % [l,p,q,r6,theta6]

%% Q.J.Ge Open path parameters---------------------------------------
% r = [3, 1, 2, 1.6];
% r6 = 0.5;
% theta6 = 0.3;
% N = 360;
% x = -2;
% y = -3;
% theta1 = 0.2;

%% Generate a set of data points and Fourier descriptors-------------
sol = 2;
n = 360; % number of points for optimization
[~,sl] = min(r); % shortest link
[target, ~] = path_gen_open_v2(r, r6, theta6, N, x, y, theta1,sol);
% plot(real(data), imag(data), 'bo')
% axis equal

%% Find the parameters of mechansim by using fmincon
fun = @(x)objfun(x,target,sol,sl,n);
scale = 1;

err_rate = 0.01;
noise = -1 + 2.*rand(1,size(target_x,2));
x0 = target_x.*(1 + err_rate.*noise);
init_err = (target_x-x0)./x0 * 100;

x0 = [x0 scale];

A = [];
b = [];
Aeq = [];
beq = [];
lb = [];
ub = [];
nonlcon = @(x)fourbarcon(x,sl);
options = optimoptions(@fmincon,'MaxIterations',1500,'MaxFunctionEvaluations',1e4);

[x, fval] = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon,options);

result_err = (x-x0)./x0 * 100;
pred_r = [x(1:sl-1), 1, x(sl:3)];
[pred, ~] = path_gen_open_v2(pred_r, x(4), x(5), 360, 0, 0, 0, sol);
pred_x = real(pred)+mean(real(target))-mean(real(pred));
pred_y = imag(pred)+mean(imag(target))-mean(imag(pred));
plot(real(target), imag(target), 'bo', pred_x, pred_y, 'r*')
axis equal