function [marked, aux, info] = embed_data_optimized(coeffs, T_zz, secret_bits, dims)
% EMBED_DATA_OPTIMIZED  Dynamic iterative selection of the embedding
%   configuration (the spirit of Algorithm 1 lines 5-10 / Eqs. 16-18):
%   search candidate frequency-band budgets and keep the one with the
%   smallest mean-squared error (MSE) between original and marked image
%   while meeting the target capacity P.
%
%   [marked, aux, info] = embed_data_optimized(coeffs, T_zz, secret_bits, dims)
%
%   dims : block-grid struct from jpeg_forward (for image-domain MSE)
%   info : struct with .best_nbands, .mse, .fsi_bits (file size increment),
%          and the per-candidate trace .trace (nbands, mse, fsi).
%
%   The paper iterates the two-dimensional histogram parameters (alpha,beta)
%   to find the optimal mapping gamma~(alpha,beta) = argmin MSE (Eq. 16).
%   Here the analogous control variable is the number of selected frequency
%   bands r (F* = {F*_1..F*_r}); fewer/lower-distortion bands reduce MSE and
%   FSI but may not meet capacity, so we evaluate the feasible candidates and
%   pick the minimum-MSE one (Eqs. 16-18).

secret_bits = secret_bits(:)';
payload = numel(secret_bits);

% Rank bands by unit distortion to get the candidate ordering (Eqs. 13-15).
[full_bands, full_cap] = select_bands(coeffs, T_zz, payload);
if full_cap < payload
    error('embed_data_optimized: capacity %d < payload %d.', full_cap, payload);
end

img0 = jpeg_inverse(coeffs, T_zz, dims);

% Candidate budgets: prefixes of the distortion-ranked band set that still
% meet capacity. select_bands already returns the minimal prefix meeting P;
% we additionally try a few larger budgets to confirm minimality of MSE.
order = block_order(coeffs, T_zz);
best = struct('mse', inf);
trace = [];

% Determine the full distortion-ranked list (not just the minimal prefix).
ranked = local_rank_bands(coeffs, T_zz);

for r = 1:numel(ranked)
    cand = sort(ranked(1:r));
    cap = local_capacity(coeffs, cand);
    if cap < payload
        continue;               % infeasible budget (Eq. 18 violated)
    end
    m = local_embed(coeffs, order, cand, secret_bits, payload);
    imgM = jpeg_inverse(m, T_zz, dims);
    mse = mean((double(img0(:)) - double(imgM(:))).^2);
    fsi = estimate_jpeg_bits(m) - estimate_jpeg_bits(coeffs);
    trace = [trace; r, mse, fsi]; %#ok<AGROW>
    if mse < best.mse
        best = struct('mse', mse, 'bands', cand, 'marked', m, 'fsi', fsi, 'r', r);
    end
    % Once feasible, a couple of extra budgets suffice to confirm the trend.
    if size(trace, 1) >= 1 && r >= find_first_feasible(ranked, coeffs, payload) + 3
        break;
    end
end

marked = best.marked;
aux = struct('payload', payload, 'bands', best.bands);
info = struct('best_nbands', best.r, 'mse', best.mse, ...
              'fsi_bits', best.fsi, 'trace', trace);
end

% ── local helpers ────────────────────────────────────────────────────────────
function ranked = local_rank_bands(coeffs, T_zz)
D = inf(1, 63);
for i = 2:64
    col = coeffs(:, i);
    Ci = sum(abs(col) == 1);
    Ji = sum(abs(col) > 1);
    if Ci > 0
        D(i-1) = (T_zz(i)^2) * Ji / Ci + 0.5 * (T_zz(i)^2);
    end
end
[Ds, idx] = sort(D, 'ascend');
ranked = idx(isfinite(Ds)) + 1;
end

function cap = local_capacity(coeffs, bands)
cap = 0;
for i = bands
    cap = cap + sum(abs(coeffs(:, i)) == 1);
end
end

function r0 = find_first_feasible(ranked, coeffs, payload)
cap = 0;
for r = 1:numel(ranked)
    cap = cap + sum(abs(coeffs(:, ranked(r))) == 1);
    if cap >= payload
        r0 = r; return;
    end
end
r0 = numel(ranked);
end

function marked = local_embed(coeffs, order, bands, secret_bits, payload)
marked = coeffs;
bi = 1;
for kk = 1:numel(order)
    k = order(kk);
    for i = bands
        E = marked(k, i);
        if abs(E) == 1
            if bi <= payload, t = secret_bits(bi); else, t = 0; end
            bi = bi + 1;
            marked(k, i) = hs_embed_coeff(E, t);
        elseif abs(E) > 1
            marked(k, i) = hs_embed_coeff(E, 0);
        end
    end
end
end
