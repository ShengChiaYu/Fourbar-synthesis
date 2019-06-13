function [data, t] =  ...
    path_gen_open_v2(r, r6, theta6, N, x, y, theta1, path)

% % % % % % % % % % % % CHECK TH2 % % % % % % % % % % % % % % 

% r=[1, 1/3, 2/3, 1.6/3];
% r6=0.5/3;
% theta6=0.3;                  %% RAD
% theta2=pi/3;
% theta1=0.2;
% N=256;
% x=-2;
% y=-3;

% % Limit Postion
if (r(1)+r(2))<=(r(3)+r(4)) && abs(r(1)-r(2))>=abs(r(3)-r(4))
    t=linspace(0, 2*pi, N); 
    
elseif (r(1)+r(2))>=(r(3)+r(4)) && abs(r(1)-r(2))>=abs(r(3)-r(4))
    alfa1=acos((r(1)^2+r(2)^2-(r(3)+r(4))^2)/(2*r(1)*r(2)));
    t=linspace(-alfa1, +alfa1, N);

elseif (r(1)+r(2))>=(r(3)+r(4)) && abs(r(1)-r(2))<=abs(r(3)-r(4))
    alfaM=acos((r(1)^2+r(2)^2-(r(3)+r(4))^2)/(2*r(1)*r(2)));
    alfam=acos((r(1)^2+r(2)^2-(r(3)-r(4))^2)/(2*r(1)*r(2)));
    t=linspace(alfam, alfaM, N);
%     t=linspace(2*pi-alfaM, 2*pi-alfam, N);
    
elseif (r(1)+r(2))<=(r(3)+r(4)) && abs(r(1)-r(2))<=abs(r(3)-r(4))
    alfaM=2*pi-acos((r(1)^2+r(2)^2-(r(3)-r(4))^2)/(2*r(1)*r(2)));
    alfam=acos((r(1)^2+r(2)^2-(r(3)-r(4))^2)/(2*r(1)*r(2)));
    t=linspace(alfam, alfaM, N);
end

% %   create theta3
for i = 1:1:N
    k1 = r(1)^2 + r(2)^2 + r(3)^2 - r(4)^2 - 2*r(1)*r(2)*cos(t(i)-theta1);
    k2 = 2*r(1)*r(3)*cos(theta1) - 2*r(2)*r(3)*cos(t(i));
    k3 = 2*r(1)*r(3)*sin(theta1) - 2*r(2)*r(3)*sin(t(i));
    a = k1 + k2;
    b = -2 * k3;
    c = k1 -k2;

    x_1 = (-b + real(sqrt(b^2 - 4*a*c))) / (2 * a); % x_1 and x_2 = tan((1/2)*th3)
    x_2 = (-b - real(sqrt(b^2 - 4*a*c))) / (2 * a);

    th3_1 = 2*atan(x_1);
    th3_2 = 2*atan(x_2);

    p1x(i) = r(2)*cos(t(i)) + r6*cos(theta6+th3_1) + x;
    p1y(i) = r(2)*sin(t(i)) + r6*sin(theta6+th3_1) + y;
    
    p2x(i) = r(2)*cos(t(i)) + r6*cos(theta6+th3_2) + x;
    p2y(i) = r(2)*sin(t(i)) + r6*sin(theta6+th3_2) + y;
end

% plot(p1x, p1y, 'o', p2x, p2y, '*',x, y, '+')
% plot(p1x, p1y, '+')
% grid on
% axis equal
% axis square

data_all=[(p1x+p1y*1i); (p2x+p2y*1i)];          %% data(2,360)
if path == 1
    data = data_all(1,:);
elseif path == 2
    data = data_all(2,:);    
end

end
