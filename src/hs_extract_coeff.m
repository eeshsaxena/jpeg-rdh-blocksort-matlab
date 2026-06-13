function [Er, t, hasbit] = hs_extract_coeff(Eh)
% HS_EXTRACT_COEFF  1-D histogram-shifting extraction/recovery (Eqs. 3-4),
%                   vectorized.
%
%   [Er, t, hasbit] = hs_extract_coeff(Eh)
%
%   Eh     : array of marked coefficients
%   Er     : recovered original coefficients
%   t      : extracted bits (0 where |Eh|==1, 1 where |Eh|==2; 0 elsewhere)
%   hasbit : logical mask, true where a secret bit was carried (|Eh| in {1,2})
%
%   Rule:
%     1 <= |Eh| <= 2 : Er = sign(Eh);  bit = 0 if |Eh|==1 else 1
%     |Eh| >= 3      : Er = Eh - sign(Eh)   (shift back)
%     Eh == 0        : unchanged

s   = sign(Eh);
a   = abs(Eh);
Er  = Eh;
t   = zeros(size(Eh));
hasbit = (a >= 1) & (a <= 2);

Er(hasbit) = s(hasbit);
t(a == 2)  = 1;

shift = a >= 3;
Er(shift) = Eh(shift) - s(shift);
end
