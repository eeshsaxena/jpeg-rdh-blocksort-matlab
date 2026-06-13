function [bands, capacity, dist] = select_bands(coeffs, T_zz, payload)
% SELECT_BANDS  Optimal frequency-band selection by unit distortion
%               (Eqs. 13-15) under a target capacity P (Eq. 18).
%
%   [bands, capacity, dist] = select_bands(coeffs, T_zz, payload)
%
%   coeffs  : nblocks x 64 quantized DCT coefficients
%   T_zz    : 1 x 64 quantization-table values
%   payload : required embedding capacity P (bits)
%
%   bands    : sorted list of selected AC band indices (2..64) forming F*
%   capacity : total +-1 embedding capacity of the selected bands (>= P)
%   dist     : total unit distortion of F* (additive distortion model)
%
%   Unit band distortion (Eq. 13): D(F_i) = T_i^2 * J_i / C_i + 0.5*T_i^2
%     C_i = #{|coef| == 1} in band i,  J_i = #{|coef| > 1}.
%   Bands are added in ascending distortion until capacity >= P
%   (minimum additive distortion under the capacity constraint).

nbands = 63;
D   = inf(1, nbands);
cap = zeros(1, nbands);
for i = 2:64
    col   = coeffs(:, i);
    Ci    = sum(abs(col) == 1);
    Ji    = sum(abs(col) > 1);
    cap(i-1) = Ci;
    if Ci > 0
        D(i-1) = (T_zz(i)^2) * Ji / Ci + 0.5 * (T_zz(i)^2);   % Eq. 13
    end
end

[Dsort, idx] = sort(D, 'ascend');
bands = [];
capacity = 0;
dist = 0;
for r = 1:nbands
    if ~isfinite(Dsort(r))
        break;
    end
    bi = idx(r) + 1;            % back to zigzag index (2..64)
    bands(end+1) = bi;         %#ok<AGROW>
    capacity = capacity + cap(idx(r));
    dist = dist + Dsort(r);
    if capacity >= payload
        break;
    end
end
bands = sort(bands);

if capacity < payload
    warning('select_bands:capacity', ...
        'Max capacity %d < requested payload %d. Using all embeddable bands.', ...
        capacity, payload);
end
end
