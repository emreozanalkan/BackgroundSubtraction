% Morphological operations for Car sequence
foregroundFiltered = bwareaopen(foreground, 100, 8);
se = strel('disk', 13);
foregroundFiltered = imdilate(foregroundFiltered, se);
foregroundFiltered = bwmorph(foregroundFiltered, 'bridge', 'Inf');
foregroundFiltered = medfilt2(foregroundFiltered, [5 5]);
foregroundFiltered = imfill(foregroundFiltered, 'holes');
foregroundFiltered = bwmorph(foregroundFiltered, 'erode', 5);
foregroundFiltered = bwmorph(foregroundFiltered, 'remove');