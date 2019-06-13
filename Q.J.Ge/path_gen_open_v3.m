function data_all = path_gen_open_v3(n, L, N)

cond_1_mask_b = (L(:,1) + L(:,2)) - (L(:,3) + L(:,4)) > 0;
cond_1_mask_s = (L(:,1) + L(:,2)) - (L(:,3) + L(:,4)) <= 0;
cond_2_mask_b = abs(L(:,1) - L(:,2)) - abs(L(:,3) - L(:,4)) >= 0;
cond_2_mask_s = abs(L(:,1) - L(:,2)) - abs(L(:,3) - L(:,4)) < 0;

% % Limit Postion
no_mask = logical(cond_1_mask_s .* cond_2_mask_b);
no_L = L(no_mask,:);
no_th2_min = zeros(size(no_L,1),1);
no_th2_max = 2*pi*ones(size(no_L,1),1);

up_mask = logical(cond_1_mask_b .* cond_2_mask_b);
up_L = L(up_mask,:);
up_th2_max = acos((up_L(:,1).^2+up_L(:,2).^2-(up_L(:,3)+up_L(:,4)).^2)./(2.*up_L(:,1).*up_L(:,2)));

bo_mask = logical(cond_1_mask_b .* cond_2_mask_s);
bo_L = L(bo_mask,:);
bo_th2_min = acos((bo_L(:,1).^2+bo_L(:,2).^2-(bo_L(:,3)-bo_L(:,4)).^2)./(2.*bo_L(:,1).*bo_L(:,2)));
bo_th2_max = acos((bo_L(:,1).^2+bo_L(:,2).^2-(bo_L(:,3)+bo_L(:,4)).^2)./(2.*bo_L(:,1).*bo_L(:,2)));

low_mask = logical(cond_1_mask_s .* cond_2_mask_s);
low_L = L(low_mask,:);
low_th2_min = acos((low_L(:,1).^2+low_L(:,2).^2-(low_L(:,3)-low_L(:,4)).^2)./(2.*low_L(:,1).*low_L(:,2)));

all_L = [no_L; up_L; bo_L; low_L];
all_th2_min = [no_th2_min; -up_th2_max; bo_th2_min; low_th2_min];
all_th2_max = [no_th2_max; up_th2_max; bo_th2_max; 2*pi-low_th2_min];

% %   create theta3
p1x = zeros(n,N);
p1y = zeros(n,N);
p2x = zeros(n,N);
p2y = zeros(n,N);
step = (all_th2_max-all_th2_min)/(N-1);
for i = 1:1:N
    th2 = all_th2_min + step*(i-1);
    
    k1 = all_L(:,1).^2 + all_L(:,2).^2 + all_L(:,3).^2 - all_L(:,4).^2 - 2*all_L(:,1).*all_L(:,2).*cos(th2-all_L(:,5));
    k2 = 2*all_L(:,1).*all_L(:,3).*cos(all_L(:,5)) - 2.*all_L(:,2).*all_L(:,3).*cos(th2);
    k3 = 2*all_L(:,1).*all_L(:,3).*sin(all_L(:,5)) - 2*all_L(:,2).*all_L(:,3).*sin(th2);
    a = k1 + k2;
    b = -2 * k3;
    c = k1 - k2;

    x_1 = (-b + real(sqrt(b.^2 - 4*a.*c))) ./ (2 * a); % x_1 and x_2 = tan((1/2)*th3)
    x_2 = (-b - real(sqrt(b.^2 - 4*a.*c))) ./ (2 * a);

    th3_1 = 2*atan(x_1);
    th3_2 = 2*atan(x_2);

    p1x(:,i) = all_L(:,2).*cos(th2) + all_L(:,6).*cos(all_L(:,7)+th3_1) + all_L(:,8);
    p1y(:,i) = all_L(:,2).*sin(th2) + all_L(:,6).*sin(all_L(:,7)+th3_1) + all_L(:,9);
    
    p2x(:,i) = all_L(:,2).*cos(th2) + all_L(:,6).*cos(all_L(:,7)+th3_2) + all_L(:,8);
    p2y(:,i) = all_L(:,2).*sin(th2) + all_L(:,6).*sin(all_L(:,7)+th3_2) + all_L(:,9);
end

% % plot(p1x, p1y, 'bo', p2x, p2y, 'r*',x, y, '+')
% % plot(p1x, p1y, '+')
% % grid on
% % axis equal
% % axis square

data_all=[(p1x+p1y*1i); (p2x+p2y*1i)];          %% data(2,360)

end