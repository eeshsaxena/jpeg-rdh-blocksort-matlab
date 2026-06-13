function S = smoothness_scores(coeffs, T_zz)
% SMOOTHNESS_SCORES  Block sorting optimization score (Eqs. 11-12).
%
%   S = smoothness_scores(coeffs, T_zz)
%
%   coeffs : nblocks x 64 quantized DCT coefficients (zigzag; col 1 = DC)
%   T_zz   : 1 x 64 quantization-table values (zigzag)
%
%   S      : nblocks x 1 smoothness score, higher = smoother (preferred
%            for embedding). Per block k:
%              tau_k = number of zero AC coefficients
%              E_k   = sum over AC of [Q!=0] * T^2            (Eq. 12)
%              S_k   = tau_k + tau_k / E_k                    (Eq. 11)
%
%   S_k is INVARIANT under the histogram-shifting embedding (zeros stay
%   zero, non-zeros stay non-zero), so the receiver re-derives the same
%   block order — the basis of reversibility for the sorting step.

ac    = coeffs(:, 2:64);
T_ac  = T_zz(2:64);
tau   = sum(ac == 0, 2);                          % zero AC count per block
Ek    = sum((ac ~= 0) .* (T_ac.^2), 2);           % Eq. 12

S = tau + tau ./ Ek;
S(Ek == 0) = 2 * tau(Ek == 0);   % all-zero AC block: define S = 2*tau (finite)
end
