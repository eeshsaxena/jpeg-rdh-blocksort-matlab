function [secret_bits, rec] = extract_data(marked, T_zz, aux)
% EXTRACT_DATA  JPEG RDH data extraction and image (coefficient) recovery.
%               Inverse of embed_data (Section III-E).
%
%   [secret_bits, rec] = extract_data(marked, T_zz, aux)
%
%   marked : nblocks x 64 marked coefficients
%   T_zz   : 1 x 64 quantization-table values
%   aux    : struct from embed_data (.payload, .bands)
%
%   secret_bits : 1 x payload recovered secret bits
%   rec         : nblocks x 64 recovered original coefficients
%
%   Because the smoothness score is invariant under histogram shifting,
%   re-sorting the marked coefficients reproduces the embedding block
%   order exactly, so extraction is deterministic and lossless.

bands   = aux.bands;
payload = aux.payload;

order = block_order(marked, T_zz);    % identical to embedding order
rec   = marked;
bits  = zeros(1, 0);

for kk = 1:numel(order)
    k = order(kk);
    for i = bands
        Eh = marked(k, i);
        a = abs(Eh);
        if a >= 1 && a <= 2
            [r, t] = hs_extract_coeff(Eh);
            rec(k, i) = r;
            bits(end+1) = t;            %#ok<AGROW>
        elseif a >= 3
            [r, ~] = hs_extract_coeff(Eh);
            rec(k, i) = r;
        end
    end
end

secret_bits = bits(1:payload);
end
