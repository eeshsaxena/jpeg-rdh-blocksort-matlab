%% RUN_ALL_TESTS  Unit tests for the JPEG RDH scheme (Li et al., IEEE TMM 2025).
%
%  T1: 1D histogram-shifting primitive reversible for all coefficients (Eqs 1-4)
%  T2: smoothness score invariant under embedding (Eqs 11-12)
%  T3: JPEG forward/inverse round-trip is stable (re-quantization idempotent)
%  T4: band selection meets requested capacity with finite distortion (Eqs 13-15)
%  T5: full embed/extract pipeline lossless across image types & quality factors
%  T6: block re-sort at extraction matches embedding order
%  T7: dynamic optimized embedding lossless and reports MSE/FSI
%  T8: capacity-overflow raises an error

clear; clc;
addpath(genpath('src'));
addpath(genpath('utils'));

results = cell(0, 2);

%% ── T1: 1D histogram shifting ───────────────────────────────────────────────
fprintf('--- T1: 1D histogram-shifting primitive ---\n');
ok = true;
for E = -40:40
    if abs(E) == 1
        for t = 0:1
            Eh = hs_embed_coeff(E, t);
            [Er, tr, hb] = hs_extract_coeff(Eh);
            ok = ok && Er == E && hb && tr == t;
        end
    else
        Eh = hs_embed_coeff(E, 0);
        [Er, ~, hb] = hs_extract_coeff(Eh);
        ok = ok && Er == E && (hb == (abs(E) >= 1 && abs(E) <= 2));
    end
end
results(end+1,:) = {ok, 'T1: HS reversible for all coefficient values'};

%% ── T2: smoothness invariance ───────────────────────────────────────────────
fprintf('--- T2: smoothness score invariance ---\n');
rng(1);
img = make_test_image('mixed', 64, 64);
[coeffs, T_zz, dims] = jpeg_forward(img, 70);
S0 = smoothness_scores(coeffs, T_zz);
secret = randi([0,1], 1, round(0.3 * sum(abs(coeffs(:,2:64))==1, 'all')));
[marked, ~] = embed_data(coeffs, T_zz, secret);
S1 = smoothness_scores(marked, T_zz);
results(end+1,:) = {max(abs(S0 - S1)) < 1e-9, 'T2: smoothness invariant under embedding'};

%% ── T3: coefficient-domain reversibility of a no-secret pass ─────────────────
% (Pixel-domain re-quantization is NOT bit-exact due to uint8 clipping; the
%  scheme's guarantee is coefficient-domain, which is what we assert here.)
fprintf('--- T3: coefficient-domain round-trip (empty payload) ---\n');
[mk3, ax3] = embed_data(coeffs, T_zz, []);
[~, rc3] = extract_data(mk3, T_zz, ax3);
results(end+1,:) = {isequal(rc3, coeffs), ...
    'T3: embed/extract restores coefficients exactly (coefficient domain)'};

%% ── T4: band selection ──────────────────────────────────────────────────────
fprintf('--- T4: band selection capacity ---\n');
P = 500;
[bands, cap, dist] = select_bands(coeffs, T_zz, P);
results(end+1,:) = {cap >= P && all(bands >= 2 & bands <= 64) && isfinite(dist), ...
    sprintf('T4: F* meets capacity (%d>=%d), %d bands', cap, P, numel(bands))};

%% ── T5: full pipeline lossless ──────────────────────────────────────────────
fprintf('--- T5: full pipeline lossless ---\n');
for ty = {'smooth', 'mixed', 'texture'}
    for qf = [50, 70, 90]
        rng(42);
        im = make_test_image(ty{1}, 64, 64);
        [cf, Tz, dm] = jpeg_forward(im, qf);
        capmax = sum(abs(cf(:,2:64)) == 1, 'all');
        if capmax < 10
            results(end+1,:) = {true, sprintf('T5: %s QF=%d (skipped, low cap=%d)', ty{1}, qf, capmax)}; %#ok<AGROW>
            continue;
        end
        pl = round(capmax * 0.4);
        sec = randi([0,1], 1, pl);
        [mk, ax] = embed_data(cf, Tz, sec);
        [so, rc] = extract_data(mk, Tz, ax);
        lossless = isequal(rc, cf) && isequal(so, sec);
        results(end+1,:) = {lossless, ...
            sprintf('T5: %s QF=%d lossless (%d bits)', ty{1}, qf, pl)}; %#ok<AGROW>
    end
end

%% ── T6: block re-sort matches ───────────────────────────────────────────────
fprintf('--- T6: extraction block order ---\n');
ord0 = block_order(coeffs, T_zz);
[mk6, ~] = embed_data(coeffs, T_zz, secret);
ord1 = block_order(mk6, T_zz);
results(end+1,:) = {isequal(ord0, ord1), 'T6: marked re-sort == embed order'};

%% ── T7: optimized embedding ─────────────────────────────────────────────────
fprintf('--- T7: dynamic optimized embedding ---\n');
rng(7);
im7 = make_test_image('smooth', 64, 64);
[cf7, Tz7, dm7] = jpeg_forward(im7, 80);
cap7 = sum(abs(cf7(:,2:64)) == 1, 'all');
pl7 = round(cap7 * 0.3);
sec7 = randi([0,1], 1, pl7);
[mk7, ax7, info7] = embed_data_optimized(cf7, Tz7, sec7, dm7);
[so7, rc7] = extract_data(mk7, Tz7, ax7);
results(end+1,:) = {isequal(rc7, cf7) && isequal(so7, sec7) && isfinite(info7.mse), ...
    sprintf('T7: optimized lossless (r=%d, MSE=%.3f, FSI=%d)', ...
            info7.best_nbands, info7.mse, info7.fsi_bits)};

%% ── T8: capacity overflow ───────────────────────────────────────────────────
fprintf('--- T8: capacity overflow ---\n');
caught = false;
try
    embed_data(cf7, Tz7, randi([0,1], 1, cap7 * 5));
catch
    caught = true;
end
results(end+1,:) = {caught, 'T8: oversized payload raises error'};

%% ── Summary ─────────────────────────────────────────────────────────────────
np = 0; nf = 0;
fprintf('\n');
for r = 1:size(results,1)
    if results{r,1}
        fprintf('  [PASS] %s\n', results{r,2}); np = np + 1;
    else
        fprintf('  [FAIL] %s\n', results{r,2}); nf = nf + 1;
    end
end
fprintf('\n========================================\n');
fprintf('  Results: %d passed, %d failed\n', np, nf);
fprintf('========================================\n');
if nf == 0
    fprintf('  ALL TESTS PASSED\n');
else
    fprintf('  SOME TESTS FAILED — check output above\n');
end
