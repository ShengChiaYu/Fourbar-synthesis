function f = objfun(x, target, sol, sl, n)
scale = x(end);
X = x(1:end-1)*scale;
r = [X(1:sl-1), scale, X(sl:3)];

[pred, ~] = path_gen_open_v2(r, X(4), X(5), n, 0, 0, 0, sol);

target_x = real(target); 
target_y = imag(target);
target_centroid_x = mean(target_x);
target_centroid_y = mean(target_y);

pred_x = real(pred);
pred_y = imag(pred);
pred_centroid_x = mean(pred_x);
pred_centroid_y = mean(pred_y);

pred_x = pred_x+(target_centroid_x-pred_centroid_x);
pred_y = pred_y+(target_centroid_y-pred_centroid_y);

[pred_X,target_X] = meshgrid(pred_x,target_x);
[pred_Y,target_Y] = meshgrid(pred_y,target_y);

dist = sqrt((pred_X-target_X).^2 + (pred_Y-target_Y).^2);
dist = reshape(dist,[],1);
dist = sort(dist,1, 'ascend');

f = sum(dist(1:size(target,2)));