function Z = zigzag_index()
% ZIGZAG_INDEX  64x2 array of (row,col) 1-based positions in JPEG zigzag
%               scan order. Z(1,:) is the DC term; Z(2:64,:) are AC terms
%               ordered F_1..F_63 (the "frequency bands" of the paper).

Z = [1 1; 1 2; 2 1; 3 1; 2 2; 1 3; 1 4; 2 3;
     3 2; 4 1; 5 1; 4 2; 3 3; 2 4; 1 5; 1 6;
     2 5; 3 4; 4 3; 5 2; 6 1; 7 1; 6 2; 5 3;
     4 4; 3 5; 2 6; 1 7; 1 8; 2 7; 3 6; 4 5;
     5 4; 6 3; 7 2; 8 1; 8 2; 7 3; 6 4; 5 5;
     4 6; 3 7; 2 8; 3 8; 4 7; 5 6; 6 5; 7 4;
     8 3; 8 4; 7 5; 6 6; 5 7; 4 8; 5 8; 6 7;
     7 6; 8 5; 8 6; 7 7; 6 8; 7 8; 8 7; 8 8];
end
