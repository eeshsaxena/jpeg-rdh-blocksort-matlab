function img = jpeg_inverse(coeffs, T_zz, dims)
% JPEG_INVERSE  Reconstruct a grayscale image from quantized DCT
%               coefficients (dequantize -> inverse DCT -> +128 -> clip).
%               Used for PSNR/SSIM evaluation of the marked image.
%
%   img = jpeg_inverse(coeffs, T_zz, dims)

C = dct_matrix();
Z = zigzag_index();
lin = sub2ind([8 8], Z(:,1), Z(:,2));
T8 = zeros(8, 8);
T8(lin) = T_zz;

img = zeros(dims.M, dims.N);
idx = 1;
for br = 1:dims.nbr
    for bc = 1:dims.nbc
        Q = zeros(8, 8);
        Q(lin) = coeffs(idx, :);
        D = Q .* T8;
        blk = C' * D * C + 128;
        img((br-1)*8+1:br*8, (bc-1)*8+1:bc*8) = blk;
        idx = idx + 1;
    end
end
img = uint8(min(255, max(0, round(img))));
end
