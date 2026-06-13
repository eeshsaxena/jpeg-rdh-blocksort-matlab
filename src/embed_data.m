function [marked, aux] = embed_data(coeffs, T_zz, secret_bits)
% EMBED_DATA  JPEG RDH data embedding (Algorithm 1).
%
%   [marked, aux] = embed_data(coeffs, T_zz, secret_bits)
%
%   coeffs      : nblocks x 64 quantized DCT coefficients (zigzag)
%   T_zz        : 1 x 64 quantization-table values (zigzag)
%   secret_bits : row/col vector of 0/1 bits to embed (length P)
%
%   marked : nblocks x 64 marked coefficients
%   aux    : auxiliary data the receiver needs (Section III-E):
%              .payload  number of secret bits (P)
%              .bands    selected optimal frequency band set F*
%              .order_check  smoothness scores (for an internal assert)
%
%   Steps (Algorithm 1):
%     1-3. Compute smoothness scores S_k and sort blocks descending.
%     4.   Select optimal frequency band set F* meeting capacity P
%          (minimum additive unit-band distortion, Eqs. 13-15).
%     7.   Histogram-shift embed (Eqs. 1-2) into the +-1 coefficients of
%          F*, processing blocks in smoothness order; |coef|>1 shift out.
%
%   Auxiliary data (P, F*) does not consume embedding capacity here; in a
%   full system it is stored at fixed positions / a reserved header.

secret_bits = secret_bits(:)';
payload = numel(secret_bits);

order = block_order(coeffs, T_zz);                 % Eqs. 11-12
[bands, capacity] = select_bands(coeffs, T_zz, payload);   % Eqs. 13-15
if capacity < payload
    error('embed_data: insufficient capacity (%d) for payload (%d).', ...
          capacity, payload);
end

marked = coeffs;
bi = 1;                          % next secret bit index
for kk = 1:numel(order)
    k = order(kk);
    for i = bands
        E = marked(k, i);
        if abs(E) == 1
            if bi <= payload
                t = secret_bits(bi);
            else
                t = 0;           % pad remaining capacity with zeros
            end
            bi = bi + 1;
            marked(k, i) = hs_embed_coeff(E, t);
        elseif abs(E) > 1
            marked(k, i) = hs_embed_coeff(E, 0);   % shift (t ignored)
        end
    end
end

aux = struct('payload', payload, 'bands', bands);
end
