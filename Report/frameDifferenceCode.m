% Morphological operations for Car sequence
Ob2 = bwareaopen(Ob, 50, 8);
Ob3 = bwmorph(Ob2, 'dilate');
Ob4 = imfill(Ob3, 'holes');
Ob5 = bwmorph(Ob4, 'erode', 2);

% Morphological operations for Highway sequence
ObFiltered = bwmorph(ObFiltered, 'bridge', 'Inf');
ObFiltered = imfill(ObFiltered, 'holes');
ObFiltered = bwmorph(ObFiltered, 'bridge', 'Inf');
ObFiltered = bwareaopen(ObFiltered, 8, 8);
ObFiltered = bwmorph(ObFiltered, 'dilate', 1);
ObFiltered = medfilt2(ObFiltered, [9 9]);
ObFiltered = imfill(ObFiltered, 'holes');
ObFiltered = bwmorph(ObFiltered, 'bridge', 'Inf');
ObFiltered = bwmorph(ObFiltered, 'erode', 1);