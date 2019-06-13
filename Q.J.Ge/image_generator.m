function [count, data_param] = image_generator(Inversion, link, s, l, p, q, r6, theta6, n, count, train, sol, pbar, start_ind)
N = 128;   % Number of points
x = 0;
y = 0;
theta1 = 0;

r(:,link(1)) = s;                                                              % r1 is ground link
r(:,link(2)) = l;                                                              % r2 is input link
r(:,link(3)) = p;                                                              % r3 is coupler link 
r(:,link(4)) = q;                                                              % r4 is output link

data_param(:,1:4) = r;
data_param(:,5) = r6;
data_param(:,6) = theta6;

% Generate points and output images
data_v2 = zeros(n,N);
% figure('visible', 'off');
% ax1 = axes('Position',[0 0 1 1]);
% axes(ax1);
for i = 1: n
    [data_v2(i,:), ~] = path_gen_open_v2(r(i,:), r6(i,1), theta6(i,1), N, x, y, theta1, sol);
    
    % print image
    % figure('Name',num2str(i),'NumberTitle','off');
    plot(real(data_v2(i,:)), imag(data_v2(i,:)), 'ko', 'markerfacecolor', 'k');
%     ax1.Visible = 'off';
    set(gca, 'Visible', 'off')
    axis equal
    
    if train
%         imagedir = strcat('/home/jjl/Yu/Fourbar/Pytorch/image_data/train/Grashof_', Inversion,'/',num2str(link(2)),'/');
        imagedir = strcat('C:\Users\dolla\Documents\Graduate_1st semester\SPforMDS\Fourbar\Pytorch\image_data\train\Grashof_'...
            , Inversion,'\',num2str(link(2)),'\');
    else
%         imagedir = strcat('/home/jjl/Yu/Fourbar/Pytorch/image_data/valid/Grashof_', Inversion,'/',num2str(link(2)),'/');
        imagedir = strcat('C:\Users\dolla\Documents\Graduate_1st semester\SPforMDS\Fourbar\Pytorch\image_data\valid\Grashof_'...
            , Inversion,'\',num2str(link(2)),'\');
    end
    
    set(gcf,'PaperUnits','inches','PaperPosition',[0 0 4 4])
    print(strcat(imagedir, num2str(count+start_ind-1)),'-dpng','-r50')
    
    count = count + 1;
    if rem(count,2) == 0
        waitbar(count/(n*2), pbar);
    end
end