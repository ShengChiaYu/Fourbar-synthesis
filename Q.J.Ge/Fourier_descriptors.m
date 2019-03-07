function [Tkall] = Fourier_descriptors(pp, lin, data)

for m=-pp:1:pp
    for k=-pp:1:pp
        omega(m+pp+1, k+pp+1) = sum(exp(1i*(k-m)*lin));
    end
    capY(m+pp+1, 1) = sum(data.*exp(-1i*m*lin));
end

% % #1 FINDING FFT COEFFICIENTS TASK CURVE
[luresL, luresU, luresP] = lu(omega);
d = luresP*capY;
Y = luresL\d;
Tkall = ((luresU\Y));      %% FFT COEFFICIENTS[-pp~+pp]
% Tk = [Tkall(1:pp);Tkall(pp+3:2*pp+1)];

end