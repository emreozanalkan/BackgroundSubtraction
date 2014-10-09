% Morphological operations for Car sequence
foreground = medfilt2(foreground, [5 5]);
foreground = bwareaopen(foreground, 20, 8);

% Morphological operations for Highway sequence
foreground = bwareaopen(foreground, 30, 8);
foreground = medfilt2(foreground, [5 5]);
foreground = bwmorph(foreground, 'bridge', 'Inf');
foreground = imfill(foreground, 'holes');
se = strel('disk', 5);
foreground = imdilate(foreground, se);