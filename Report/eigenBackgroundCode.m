% Morphological operations for Car and Highway sequence
foregroundImage = reshape(foreground, [imageHeight, imageWidth]);
foregroundImage = bwareaopen(foregroundImage, 16, 8);
foregroundImage = bwmorph(foregroundImage, 'bridge', 'Inf');
se = strel('disk', 7);
foregroundImage = imdilate(foregroundImage, se);
foregroundImage = medfilt2(foregroundImage, [9 9]);
foregroundImage = imfill(foregroundImage, 'holes');
foregroundImage = bwmorph(foregroundImage, 'erode', 5);