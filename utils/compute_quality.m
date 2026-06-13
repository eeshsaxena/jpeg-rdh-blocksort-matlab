function out = compute_quality(mode, A, B)
% COMPUTE_QUALITY  Visual-quality metrics used in Section IV of the paper.
%
%   p = compute_quality('psnr', A, B)   % PSNR in dB  (Eq. 19), Inf if equal
%   s = compute_quality('ssim', A, B)   % global SSIM (Eq. 20)
%
%   A, B : grayscale images (uint8 or double), same size.

A = double(A);  B = double(B);
switch lower(mode)
    case 'psnr'
        mse = mean((A(:) - B(:)).^2);
        if mse == 0
            out = Inf;
        else
            out = 10 * log10(255^2 / mse);
        end
    case 'ssim'
        % Global single-window SSIM (Eq. 20).
        c1 = (0.01 * 255)^2;
        c2 = (0.03 * 255)^2;
        muA = mean(A(:));  muB = mean(B(:));
        vA  = var(A(:), 1); vB = var(B(:), 1);
        cov = mean((A(:) - muA) .* (B(:) - muB));
        out = ((2*muA*muB + c1) * (2*cov + c2)) / ...
              ((muA^2 + muB^2 + c1) * (vA + vB + c2));
    otherwise
        error('Unknown metric: %s', mode);
end
end
