%     [image, map] = imread('input\small.png');
%     image(:,:, 4) = 255;
%     image_asd = image(:);
%     image = image(:,:, 1:3);
%     figure;
%     imshow(image);
%     
%     
%     %image_r = reshape(image, 512,512,4);
%     
%     %image_r = image_r(:,:,1:3);
%     
%     %imshow(image_r);

[C,I] = max(accum(:));
C
I;
[I1,I2] = ind2sub(size(accum),I)
