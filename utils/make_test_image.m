function img = make_test_image(type, M, N)
% MAKE_TEST_IMAGE  Synthetic 8-bit grayscale test images (dims multiple of 8).
%
%   img = make_test_image(type, M, N)
%
%   type : 'smooth'  - low-frequency content (many zero ACs, high capacity)
%          'texture' - high-frequency noise (few zero ACs, like Baboon)
%          'mixed'   - smooth + textured halves
%
%   Replace with imread('baboon.png') etc. to run on standard test images.

if nargin < 2, M = 256; end
if nargin < 3, N = 256; end
[cc, rr] = meshgrid(1:N, 1:M);
switch lower(type)
    case 'smooth'
        img = 110 + 50*sin(rr/40) + 40*cos(cc/55) + (rr+cc)/12;
    case 'texture'
        rng(1);
        img = 128 + 60*randn(M, N);
    case 'mixed'
        img = 110 + 50*sin(rr/40) + (rr+cc)/12;
        rng(2);
        noise = 128 + 60*randn(M, N);
        img(:, floor(N/2)+1:end) = noise(:, floor(N/2)+1:end);
    otherwise
        error('Unknown type: %s', type);
end
img = uint8(min(255, max(0, round(img))));
end
