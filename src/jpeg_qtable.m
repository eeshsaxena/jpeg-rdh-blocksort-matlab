function T = jpeg_qtable(qf)
% JPEG_QTABLE  Standard JPEG luminance quantization table scaled to a
%              quality factor (Annex K + IJG scaling).
%
%   T = jpeg_qtable(qf)   % qf in 1..100, returns 8x8 integer table
%
%   The non-zero quantization values T(s,w) drive the smoothness score
%   (Eq. 12) and the unit band distortion (Eq. 13) in the paper.

Q50 = [16 11 10 16 24 40 51 61;
       12 12 14 19 26 58 60 55;
       14 13 16 24 40 57 69 56;
       14 17 22 29 51 87 80 62;
       18 22 37 56 68 109 103 77;
       24 35 55 64 81 104 113 92;
       49 64 78 87 103 121 120 101;
       72 92 95 98 112 100 103 99];

if qf < 50
    s = 5000 / qf;
else
    s = 200 - 2 * qf;
end
T = floor((Q50 * s + 50) / 100);
T(T < 1) = 1;
end
