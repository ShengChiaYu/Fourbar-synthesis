function [data, t] =  ...
    path_gen_open(r, r6, theta6, N, x, y, theta1)

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
    
elseif (r(1)+r(2))<=(r(3)+r(4)) && abs(r(1)-r(2))<=abs(r(3)-r(4))
    alfaM=2*pi-acos((r(1)^2+r(2)^2-(r(3)-r(4))^2)/(2*r(1)*r(2)));
    alfam=acos((r(1)^2+r(2)^2-(r(3)-r(4))^2)/(2*r(1)*r(2)));
    t=linspace(alfam, alfaM, N);
end

% %   create theta3
h1=r(1)/r(2);
h2=r(1)/r(3);
h4=(-r(1)^2-r(2)^2-r(3)^2+r(4)^2)/(2*r(2)*r(3));

for i=1:1:N      %% i = theta2
    a(i)=-h1+(1+h2)*cos(t(i))+h4;
    b(i)=-2*sin(t(i));
    c(i)=h1-(1-h2)*cos(t(i))+h4;
    theta3(i,:)=[(2*atan((-b(i)+sqrt(b(i)^2-4*a(i)*c(i)))/(2*a(i)))),...
        (2*atan((-b(i)-sqrt(b(i)^2-4*a(i)*c(i)))/(2*a(i))))];          %% theta3 with + and -

    % % creat coupler curve
    xcorp(i)=x+r(2)*cos(t(1, i)+theta1)+r6*cos(theta6+theta3(i,1)+theta1);
    ycorp(i)=y+r(2)*sin(t(1, i)+theta1)+r6*sin(theta6+theta3(i,1)+theta1);

    xcorm(i)=x+r(2)*cos(t(1, i)+theta1)+r6*cos(theta6+theta3(i,2)+theta1);
    ycorm(i)=y+r(2)*sin(t(1, i)+theta1)+r6*sin(theta6+theta3(i,2)+theta1);

end


% plot(xcorp, ycorp, '+', xcorm, ycorm, '*')
% plot(xcorm, ycorm, '+')
% grid on
% axis equal
% % axis square

data=[(xcorp+ycorp*1i); (xcorm+ycorm*1i)];          %% data(2,360)


end
