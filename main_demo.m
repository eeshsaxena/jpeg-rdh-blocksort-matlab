%% MAIN_DEMO  JPEG RDH via block sorting optimization and dynamic iterative
%             histogram modification (Li et al., IEEE TMM 2025).
%
%  Pipeline (Fig. 2):
%    JPEG decode -> quantized DCT coefficients
%    -> block smoothness sorting (Eqs. 11-12)
%    -> optimal frequency-band selection (Eqs. 13-15)
%    -> dynamic iterative histogram-shift embedding (Algorithm 1, Eqs. 1-2)
%    -> marked image
%  Receiver:
%    re-sort (smoothness invariant) -> extract secret (Eqs. 3-4)
%    -> recover coefficients -> perfect original JPEG image.

clear; clc; close all;
addpath(genpath('src'));
addpath(genpath('utils'));
rng(2025);

%% ── Parameters ──────────────────────────────────────────────────────────────
QF      = 70;
IMTYPE  = 'smooth';
IMSIZE  = [256, 256];

img = make_test_image(IMTYPE, IMSIZE(1), IMSIZE(2));

%% ── JPEG forward: quantized DCT coefficients ────────────────────────────────
fprintf('=== JPEG forward transform (QF=%d) ===\n', QF);
[coeffs, T_zz, dims] = jpeg_forward(img, QF);
fprintf('    %d blocks, %d coefficients each\n', size(coeffs,1), size(coeffs,2));

total_pm1 = sum(abs(coeffs(:,2:64)) == 1, 'all');
fprintf('    Total +-1 AC coefficients (max capacity): %d bits\n', total_pm1);

%% ── Choose payload and embed ────────────────────────────────────────────────
payload = round(total_pm1 * 0.4);
secret  = randi([0,1], 1, payload);
fprintf('\n=== Embedding %d secret bits ===\n', payload);

[marked, aux, info] = embed_data_optimized(coeffs, T_zz, secret, dims);
fprintf('    Selected %d frequency bands: %s\n', numel(aux.bands), mat2str(aux.bands));
fprintf('    Optimal band budget r=%d  (min-MSE over feasible candidates)\n', info.best_nbands);
fprintf('    Image MSE: %.4f | File size increment: %d bits\n', info.mse, info.fsi_bits);

marked_img = jpeg_inverse(marked, T_zz, dims);
psnr_val = compute_quality('psnr', img, marked_img);
ssim_val = compute_quality('ssim', img, marked_img);
fprintf('    Marked image PSNR: %.2f dB | SSIM: %.5f\n', psnr_val, ssim_val);

%% ── Extract + recover ───────────────────────────────────────────────────────
fprintf('\n=== Extraction and recovery ===\n');
[secret_out, rec_coeffs] = extract_data(marked, T_zz, aux);

data_ok  = isequal(secret_out, secret);
coeff_ok = isequal(rec_coeffs, coeffs);
rec_img  = jpeg_inverse(rec_coeffs, T_zz, dims);
img_ok   = isequal(rec_img, jpeg_inverse(coeffs, T_zz, dims));

fprintf('    Secret data bit-exact : %d\n', data_ok);
fprintf('    Coefficients restored : %d\n', coeff_ok);
fprintf('    JPEG image restored   : %d\n', img_ok);
if data_ok && coeff_ok && img_ok
    fprintf('    >>> Perfect reversibility confirmed <<<\n');
end

%% ── Visualisation ───────────────────────────────────────────────────────────
recompressed = jpeg_inverse(coeffs, T_zz, dims);   % original JPEG (no data)
figure('Name','JPEG RDH Pipeline','NumberTitle','off','Position',[60 60 1200 640]);
subplot(2,3,1); imshow(img);          title('Original');
subplot(2,3,2); imshow(recompressed); title(sprintf('JPEG QF=%d', QF));
subplot(2,3,3); imshow(marked_img);
    title(sprintf('Marked  (PSNR %.2f dB)', psnr_val));
subplot(2,3,4); imshow(rec_img);      title('Recovered (lossless)');
subplot(2,3,5);
    diffimg = abs(double(recompressed) - double(rec_img));
    imshow(diffimg, []); title('Recovery diff (all black)');
subplot(2,3,6);
    if ~isempty(info.trace)
        yyaxis left;  plot(info.trace(:,1), info.trace(:,2), '-o'); ylabel('MSE');
        yyaxis right; plot(info.trace(:,1), info.trace(:,3), '-s'); ylabel('FSI (bits)');
        xlabel('Band budget r'); title('Dynamic selection trace'); grid on;
    end
sgtitle(sprintf('JPEG RDH — block sorting + dynamic histogram (QF=%d, %d bits)', QF, payload));
