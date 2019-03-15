function plot_check(original, predict, fig)
% fig, Index of first figure.

i_splt = 1;                     % Index of subplot.

for i = fig:1:fig+8
subplot(3,3,i_splt)
r = zeros(2,4);
r6 = zeros(2,1);
r(1,2) = 1;                                    % r2 is the shortest link and set as unit.
r(1,1) = original(i,1);                        % r1 is p-link and chosen 1 <= p <= 5 at random.
r(1,4) = original(i,3);                        % r4 is the longest link and chosen p <= l <= 5 at random.
r(1,3) = original(i,2);                        % r3 is q-link and chosen s+l-p <= q <= l at random.
r6(1,1) = original(i,4);                       % r6 is chosen 1 <= r6 <= 5 at random.
theta6(1,:) = original(i,5);                   % theta6 is chosen 0 <= theta6 <= 2*pi at random.
% original(n,:)

r(2,2) = 1;                                    % r2 is the shortest link and set as unit.
r(2,1) = predict(i,1);                        % r1 is p-link and chosen 1 <= p <= 5 at random.
r(2,4) = predict(i,3);                        % r4 is the longest link and chosen p <= l <= 5 at random.
r(2,3) = predict(i,2);                        % r3 is q-link and chosen s+l-p <= q <= l at random.
r6(2,1) = predict(i,4);                       % r6 is chosen 1 <= r6 <= 5 at random.
theta6(2,:) = predict(i,5);                   % theta6 is chosen 0 <= theta6 <= 2*pi at random.
% predict(n,:)

N = 100;   % Number of points
x = 0;
y = 0;
theta1 = 0;

[data_original, ~] = path_gen_open_v2(r(1,:), r6(1,:), theta6(1,:), N, x, y, theta1,1);
[data_predict, ~] = path_gen_open_v2(r(2,:), r6(2,:), theta6(2,:), N, x, y, theta1,1);

plot(real(data_original(1,:)), imag(data_original(1,:)), 'bo', real(data_predict(1,:)), imag(data_predict(1,:)), 'r*')
axis equal
legend('original','predict')
i_splt = i_splt + 1;

end

end