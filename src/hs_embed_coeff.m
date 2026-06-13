function Eh = hs_embed_coeff(E, t)
% HS_EMBED_COEFF  1-D histogram-shifting embedding (Eqs. 1-2), vectorized.
%
%   Eh = hs_embed_coeff(E, t)
%
%   E : array of quantized coefficients
%   t : array of secret bits (0/1), same size as E; only consumed where
%       |E| == 1 (elsewhere the value of t is ignored)
%
%   Rule:
%     |E| == 1 : Eh = E + sign(E)*t   (+-1 stays, or expands to +-2)
%     |E|  > 1 : Eh = E + sign(E)      (shift outward to vacate +-2)
%     E   == 0 : unchanged

s  = sign(E);
Eh = E;
one  = abs(E) == 1;
big  = abs(E) > 1;
Eh(one) = E(one) + s(one) .* t(one);
Eh(big) = E(big) + s(big);
end
