function [coeffs, T_zz, dims] = jpeg_forward(img, qf)
% JPEG_FORWARD  Simulate JPEG encode up to quantized DCT coefficients.
%
%   [coeffs, T_zz, dims] = jpeg_forward(img, qf)
%
%   img    : grayscale image (uint8 or double), dimensions multiple of 8
%   qf     : JPEG quality factor (1..100)
%
%   coeffs : nblocks x 64 integer matrix of quantized DCT coefficients in
%            zigzag order (column 1 = DC, columns 2..64 = AC bands F_1..F_63)
%   T_zz   : 1 x 64 quantization-table values in the same zigzag order
%   dims   : struct with .M .N .nbr .nbc (block grid) for reconstruction
%
%   These quantized coefficients are the carrier the RDH scheme operates on.

img = double(img);
[M, N] = size(img);
if mod(M, 8) || mod(N, 8)
    error('Image dimensions must be multiples of 8.');
end

T = jpeg_qtable(qf);
C = dct_matrix();
Z = zigzag_index();
lin = sub2ind([8 8], Z(:,1), Z(:,2));   % zigzag -> linear 8x8 index
T_zz = T(lin)';

nbr = M / 8;  nbc = N / 8;
coeffs = zeros(nbr * nbc, 64);
idx = 1;
for br = 1:nbr
    for bc = 1:nbc
        blk = img((br-1)*8+1:br*8, (bc-1)*8+1:bc*8) - 128;
        D = C * blk * C';
        Q = round(D ./ T);
        coeffs(idx, :) = Q(lin)';
        idx = idx + 1;
    end
end

dims = struct('M', M, 'N', N, 'nbr', nbr, 'nbc', nbc);
end
