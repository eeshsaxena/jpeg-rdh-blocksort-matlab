function order = block_order(coeffs, T_zz)
% BLOCK_ORDER  Order blocks by DESCENDING smoothness score (Eqs. 11-12).
%              Smoother (more zero-coefficient) blocks are embedded first
%              to minimise distortion. Stable sort so the order is
%              reproducible by the receiver.
%
%   order = block_order(coeffs, T_zz)   % nblocks x 1 permutation

S = smoothness_scores(coeffs, T_zz);
[~, order] = sort(S, 'descend');
end
