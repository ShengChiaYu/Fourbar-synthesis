function [c,ceq] = fourbarcon(x,sl)
r = [x(1:sl-1), x(end), x(sl:3)];
ratio = 5;

% Grashof
r = sort(r(1:4),2,'ascend');
c(1) = r(1)+r(4)-r(2)-r(3);

% Ratio constraints
c(2) = x(end)-x(1);
c(3) = x(end)-x(2);
c(4) = x(end)-x(3);
c(5) = x(end)-x(4);
c(6) = x(1)/x(end)-ratio;
c(7) = x(2)/x(end)-ratio;
c(8) = x(3)/x(end)-ratio;
c(8) = x(4)/x(end)-ratio;

ceq = [];
