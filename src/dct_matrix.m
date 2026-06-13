function C = dct_matrix()
% DCT_MATRIX  8x8 orthonormal type-II DCT matrix (no toolbox required).
%             Block DCT:  D = C * (X-128) * C';  inverse: X = C' * D * C + 128.

C = zeros(8, 8);
for k = 0:7
    if k == 0
        a = sqrt(1/8);
    else
        a = sqrt(2/8);
    end
    for nn = 0:7
        C(k+1, nn+1) = a * cos((2*nn + 1) * k * pi / 16);
    end
end
end
