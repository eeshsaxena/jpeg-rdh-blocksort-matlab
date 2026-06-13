function nbits = estimate_jpeg_bits(coeffs)
% ESTIMATE_JPEG_BITS  Estimate the JPEG entropy-coded size of the AC
%                     coefficients, as a proxy for File Size Increment (FSI).
%
%   nbits = estimate_jpeg_bits(coeffs)
%
%   coeffs : nblocks x 64 quantized DCT coefficients (zigzag)
%
%   Models the baseline JPEG AC coding cost: each non-zero AC coefficient
%   costs (run/size Huffman symbol ~ 4 bits, approximated) + size(category)
%   amplitude bits, where category = floor(log2(|coef|)) + 1. Zero-runs and
%   the EOB are folded into the per-symbol constant. This is not a bit-exact
%   JPEG encoder, but it tracks the file-size change caused by data
%   embedding (shifting +-1 -> +-2 raises some categories, and creates new
%   amplitude bits), which is what the paper's FSI metric measures.

HUFF_SYMBOL = 4;     % approx Huffman codeword length for a (run,size) symbol
nbits = 0;
ac = coeffs(:, 2:64);
nz = ac(ac ~= 0);
if isempty(nz)
    return;
end
cat = floor(log2(abs(nz))) + 1;        % JPEG amplitude category (size)
nbits = numel(nz) * HUFF_SYMBOL + sum(cat);
% End-of-block markers, one per block (approx 4 bits each)
nbits = nbits + size(coeffs, 1) * HUFF_SYMBOL;
end
