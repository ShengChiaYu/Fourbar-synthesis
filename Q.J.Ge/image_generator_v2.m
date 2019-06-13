function image_generator_v2(dataset, n, Inversion, link, start_ind, N)
s = ones(n,1); 
p = 1 + (5-1).*rand(n,1);                                                    % p-link and chosen 1 <= p <= 5 at random.
l = p + (5-p).*rand(n,1);                                                    % l-link and chosen p <= l <= 5 at random.
q = s+l-p + (l-(s+l-p)).*rand(n,1);                                          % q-link and chosen s+l-p <= q <= l at random.
r6 = 1 + (5-1).*rand(n,1);                                                   % r6 is chosen 1 <= r6 <= 5 at random.
theta6 = 2*pi.*rand(n,1);                                                    % theta6 is chosen 0 <= theta6 <= 2*pi at random.

x = zeros(n,1);
y = zeros(n,1);
theta1 = zeros(n,1);

L(:,link(1)) = s;                                                              % r1 is ground link
L(:,link(2)) = l;                                                              % r2 is input link
L(:,link(3)) = p;                                                              % r3 is coupler link 
L(:,link(4)) = q;                                                              % r4 is output link
L(:,5) = theta1;
L(:,6) = r6;
L(:,7) = theta6;
L(:,8) = x;
L(:,9) = y;

% Output labels
labeldir = strcat('/home/jjl/Yu/Fourbar/Pytorch/image_data/',dataset,'/Grashof_', Inversion, '/', num2str(link(2)), '_label.csv');
% labeldir = strcat('C:\Users\dolla\Documents\Graduate_1st semester\SPforMDS\Fourbar\Pytorch\image_data\',dataset,...
% '\Grashof_'..., Inversion, '\', num2str(link(2)), '_label.csv');
csvwrite(labeldir, [L;L])

% Generate points and output images
msg = strcat('Generating', num2str(n*2), Inversion, num2str(link(2)), 'images...');
pbar = waitbar(0, msg);
data_all = path_gen_open_v3(n, L, N);
for i = 1: size(data_all,1)
    % print image
    plot(real(data_all(i,:)), imag(data_all(i,:)), 'ko', 'markerfacecolor', 'k');
    set(gca, 'Visible', 'off')
    axis equal
    
    imagedir = strcat('/home/jjl/Yu/Fourbar/Pytorch/image_data/',dataset,'/Grashof_', Inversion,'/',num2str(link(2)),'/');
%     imagedir = strcat('C:\Users\dolla\Documents\Graduate_1st semester\SPforMDS\Fourbar\Pytorch\image_data\',dataset,...
%     '\Grashof_', Inversion,'\',num2str(link(2)),'\');
    
    set(gcf,'PaperUnits','inches','PaperPosition',[0 0 4 4])
    print(strcat(imagedir, num2str(i+start_ind-1)),'-dpng','-r50')
    
    if rem(i,2) == 0
        waitbar(i/size(data_all,1), pbar);
    end

end
close(gcf)
close(pbar)

end

