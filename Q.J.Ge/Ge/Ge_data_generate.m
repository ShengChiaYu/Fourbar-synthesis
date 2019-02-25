%% Testing data----------------------------------------------------
% r = [30, 40, 31, 38]; % Upper limit exists. c1>0  c2>=0
% r = [20, 29, 30, 40]; % Lower limit exists. c1<=0 c2<0
% r = [31, 40, 20, 40]; % Both limit exist.   c1>0  c2<0
% r = [20, 40, 30, 40]; % No limit exists.    c1<=0 c2>=0
% r6 = r(3)/2*sqrt(2);
% theta6 = pi/4;
% N = 360;
% x = 0;
% y = 0;
% theta1 = 0;
% pp = 2;

%% Q.J.Ge Closed path parameters------------------------------------
% r = [11, 6, 8, 10];
% r6 = 7;
% theta6 = 0.6981;
% N = 360;
% x = 10;
% y = 14;
% theta1 = 0.1745;
% pp = 2;
% Tk_Ge = [-0.7574+0.0774i -2.3581+0.9571i 8.5305+19.644i 4.6530-1.3332i 0.5513-0.4648i].';

%% Q.J.Ge Open path parameters---------------------------------------
% r = [3, 1, 2, 1.6];
% r6 = 0.5;
% theta6 = 0.3;
% N = 360;
% x = -2;
% y = -3;
% theta1 = 0.2;
% pp = 2;
% Tk_Ge = [0.0061+0.0137i -0.0590+0.1365i -1.6911-2.6475i 0.8290+0.2092i 0.0202-0.0364i].';

%% Generate a set of data points and Fourier descriptors-------------
% [data_v1, theta2] = path_gen_open(r, r6, theta6, N, x, y, theta1);
% data = data_v1(1,:);
% [data_v2, theta2] = path_gen_open_v2(r, r6, theta6, N, x, y, theta1,2);
% data = data_v2;

% Tk = Fourier_descriptors(pp, theta2, data);                   % Generate task curve Fourier descriptors.
% 
% z = zeros(1,N);                                               % Calculate the complex z(i) by FD.                       
% for i = 1:1:N
%     for k = -pp:1:pp
%         z(i) = z(i) + Tk(k+pp+1)*exp(1i*k*theta2(i));
%     end
% end

% z_Ge = zeros(1,N);                                            % Calculate the complex z(i) by FD, Ge's version.
% for i = 1:1:N
%     for k = -pp:1:pp
%         z_Ge(i) = z_Ge(i) + Tk_Ge(k+pp+1)*exp(1i*k*theta2(i));
%     end
% end

% plot(real(data), imag(data), 'bo', real(z), imag(z), 'r*')
% plot(real(data), imag(data), 'bo', real(z), imag(z), 'r*', real(z_Ge), imag(z_Ge), 'g.')
% axis equal

%% Generate numerous sets of data points and Fourier descriptors--------------
% Generate random parameters according to ANN and FD paper.
n = 500; % Number of data sets.
r = zeros(n,4);
r(:,2) = 1;                                                                 % r2 is the shortest link and set as unit.
r(:,1) = 1 + (5-1)*rand(n,1);                                               % r1 is p-link and chosen 1 <= p <= 5 at random.
r(:,4) = r(:,1) + (5-r(:,1)).*rand(n,1);                                    % r4 is the longest link and chosen p <= l <= 5 at random.
r(:,3) = r(:,2)+r(:,4)-r(:,1) + (r(:,4)-(r(:,2)+r(:,4)-r(:,1))).*rand(n,1); % r3 is q-link and chosen s+l-p <= q <= l at random.
r6 = 1 + (5-1)*rand(n,1);                                                   % r6 is chosen 1 <= r6 <= 5 at random.
theta6 = 2*pi*rand(n,1);                                                    % theta6 is chosen 0 <= theta6 <= 2*pi at random.
N = 360;
x = 0;
y = 0;
theta1 = 0;
pp = 2;

% Generate task curve Fourier descriptors.
Tk = zeros(n,pp*2+1);
data_v2 = zeros(n,N);
z = zeros(n,N);
for i = 1:1:n
[data_v2(i,:), theta2] = path_gen_open_v2(r(i,:), r6(i,:), theta6(i,:), N, x, y, theta1,1);
data = data_v2(i,:);
Tk(i,:) = Fourier_descriptors(pp, theta2, data);
                                               % Calculate the complex z(i) by FD.                       
for j = 1:1:N
    for k = -pp:1:pp
        z(i,j) = z(i,j) + Tk(i,k+pp+1)*exp(1i*k*theta2(j));
    end
end

end

%% Plot the data sets
fig = 1 + 9 * 5;            % Index of first figure.
si = 1;                     % Index of subplot.
for i = fig:1:fig+8
    subplot(3,3,si)
    plot(real(data_v2(i,:)), imag(data_v2(i,:)), 'bo', real(z(i,:)), imag(z(i,:)), 'r*')
    axis equal
    si = si + 1;
end
