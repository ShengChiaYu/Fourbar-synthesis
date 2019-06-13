function Grashof(dataset, n, Inversion, set)
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

%% Generate a image
% [data_v2, theta2] = path_gen_open_v2(r, r6, theta6, N, x, y, theta1,1);
% data = data_v2;
% 
% figure('Name','1','NumberTitle','off');
% plot(real(data), imag(data), 'bo', 'markerfacecolor', 'b')
% set(gca, 'Visible', 'off')
% axis equal
% 
% imagedir = '/home/jjl/Yu/Fourbar/Pytorch/image_data/Grashof_crank_rocker/';
% set(gcf,'PaperUnits','inches','PaperPosition',[0 0 4 3])
% print(strcat(imagedir, '1'),'-dpng','-r100')

%% Permutation of Grashof four-bar mechanism
link_permutations = perms([1 2 3 4]);
keySet = {'GRRC','GRCR','GCRR','GCCC'};
valueSet = {link_permutations(1:6,:), link_permutations(7:12,:), link_permutations(13:18,:), link_permutations(19:24,:)};
link_set = containers.Map(keySet, valueSet);

%% Generate numerous sets of data points and images--------------
% Generate random parameters according to ANN and FD paper.
% dataset = 'train';
% n = 1000;  % Number of data sets.
% Inversion = 'GRRC';
% set = 1;
start_ind = 1;
N = 512;

fprintf(strcat(Inversion,' set %d\n'),set)

% Generate random length of links.
links = link_set(Inversion);
link = links(2*set-1,:);

image_generator_v2(dataset, n, Inversion, link, start_ind, N);
end

