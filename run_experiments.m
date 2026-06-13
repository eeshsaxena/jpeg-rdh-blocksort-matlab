%% RUN_EXPERIMENTS  Reproduce the paper's analyses (Section IV):
%   E1: PSNR / SSIM / FSI vs. embedding capacity (Figs. 10-12 equivalent)
%   E2: effect of quality factor on capacity and quality (Table I/II spirit)
%   E3: block-sorting benefit — smoothness-ordered vs. natural-order embedding
%   E4: dynamic band-budget trace (MSE & FSI vs. r), the (alpha,beta) analog
%
%  Synthetic images are used; swap in imread('baboon.png') etc. for the
%  paper's USC-SIPI / Kodak / UCID test images.

clear; clc; close all;
addpath(genpath('src'));
addpath(genpath('utils'));
rng(7);

img = make_test_image('smooth', 256, 256);

%% ── E1: PSNR / SSIM / FSI vs. capacity ──────────────────────────────────────
fprintf('=== E1: PSNR / SSIM / FSI vs. embedding capacity (QF=70) ===\n');
[coeffs, T_zz, dims] = jpeg_forward(img, 70);
img0 = jpeg_inverse(coeffs, T_zz, dims);
capmax = sum(abs(coeffs(:,2:64)) == 1, 'all');

loads = round(capmax * (0.1:0.1:0.6));
fprintf('%10s %10s %10s %12s\n', 'Payload', 'PSNR(dB)', 'SSIM', 'FSI(bits)');
fprintf('%s\n', repmat('-',1,46));
E1 = zeros(numel(loads), 4);
for j = 1:numel(loads)
    P = loads(j);
    sec = randi([0,1], 1, P);
    [marked, aux] = embed_data(coeffs, T_zz, sec);
    [so, rc] = extract_data(marked, T_zz, aux);
    assert(isequal(rc, coeffs) && isequal(so, sec), 'reversibility broken');
    mimg = jpeg_inverse(marked, T_zz, dims);
    ps = compute_quality('psnr', img0, mimg);
    ss = compute_quality('ssim', img0, mimg);
    fsi = estimate_jpeg_bits(marked) - estimate_jpeg_bits(coeffs);
    E1(j,:) = [P, ps, ss, fsi];
    fprintf('%10d %10.2f %10.5f %12d\n', P, ps, ss, fsi);
end

%% ── E2: quality factor sweep ────────────────────────────────────────────────
fprintf('\n=== E2: capacity & quality vs. quality factor (40%% load) ===\n');
fprintf('%6s %12s %10s %12s\n', 'QF', 'MaxCap', 'PSNR(dB)', 'FSI(bits)');
fprintf('%s\n', repmat('-',1,44));
for qf = [50, 60, 70, 80, 90]
    [cf, Tz, dm] = jpeg_forward(img, qf);
    cmax = sum(abs(cf(:,2:64)) == 1, 'all');
    P = round(cmax * 0.4);
    sec = randi([0,1], 1, P);
    [mk, ax] = embed_data(cf, Tz, sec);
    [so, rc] = extract_data(mk, Tz, ax);
    assert(isequal(rc, cf) && isequal(so, sec));
    ps = compute_quality('psnr', jpeg_inverse(cf,Tz,dm), jpeg_inverse(mk,Tz,dm));
    fsi = estimate_jpeg_bits(mk) - estimate_jpeg_bits(cf);
    fprintf('%6d %12d %10.2f %12d\n', qf, cmax, ps, fsi);
end

%% ── E3: block-sorting benefit ───────────────────────────────────────────────
fprintf('\n=== E3: block sorting vs. natural order (QF=70) ===\n');
P = round(capmax * 0.3);
sec = randi([0,1], 1, P);
[bands, ~] = select_bands(coeffs, T_zz, P);

% Sorted order (proposed)
ord_sorted = block_order(coeffs, T_zz);
m_sorted = local_embed_order(coeffs, ord_sorted, bands, sec);
ps_sorted = compute_quality('psnr', img0, jpeg_inverse(m_sorted, T_zz, dims));

% Natural raster order (baseline)
ord_nat = (1:size(coeffs,1))';
m_nat = local_embed_order(coeffs, ord_nat, bands, sec);
ps_nat = compute_quality('psnr', img0, jpeg_inverse(m_nat, T_zz, dims));

fprintf('    Smoothness-sorted order PSNR : %.3f dB\n', ps_sorted);
fprintf('    Natural raster order   PSNR : %.3f dB\n', ps_nat);
fprintf('    Sorting gain                : %+.3f dB\n', ps_sorted - ps_nat);

%% ── E4: dynamic band-budget trace ───────────────────────────────────────────
fprintf('\n=== E4: dynamic band-budget trace (the (alpha,beta) analog) ===\n');
[~, ~, info] = embed_data_optimized(coeffs, T_zz, randi([0,1],1,P), dims);
fprintf('%8s %12s %12s\n', 'r bands', 'MSE', 'FSI(bits)');
fprintf('%s\n', repmat('-',1,34));
for j = 1:size(info.trace,1)
    fprintf('%8d %12.4f %12d\n', info.trace(j,1), info.trace(j,2), info.trace(j,3));
end
fprintf('    Chosen budget r=%d minimises MSE (%.4f).\n', info.best_nbands, info.mse);

%% ── Plot E1 ─────────────────────────────────────────────────────────────────
figure('Name','Capacity-Distortion','NumberTitle','off','Position',[80 80 1100 340]);
subplot(1,3,1); plot(E1(:,1), E1(:,2), '-o','LineWidth',1.2);
    xlabel('Payload (bits)'); ylabel('PSNR (dB)'); title('PSNR vs capacity'); grid on;
subplot(1,3,2); plot(E1(:,1), E1(:,3), '-s','LineWidth',1.2);
    xlabel('Payload (bits)'); ylabel('SSIM'); title('SSIM vs capacity'); grid on;
subplot(1,3,3); plot(E1(:,1), E1(:,4), '-^','LineWidth',1.2);
    xlabel('Payload (bits)'); ylabel('FSI (bits)'); title('File size increment'); grid on;
sgtitle('JPEG RDH capacity-distortion behaviour (QF=70, smooth image)');

% ── local helper ─────────────────────────────────────────────────────────────
function marked = local_embed_order(coeffs, order, bands, secret)
marked = coeffs; bi = 1; P = numel(secret);
for kk = 1:numel(order)
    k = order(kk);
    for i = bands
        E = marked(k,i);
        if abs(E) == 1
            if bi <= P, t = secret(bi); else, t = 0; end
            bi = bi + 1; marked(k,i) = hs_embed_coeff(E, t);
        elseif abs(E) > 1
            marked(k,i) = hs_embed_coeff(E, 0);
        end
    end
end
end
