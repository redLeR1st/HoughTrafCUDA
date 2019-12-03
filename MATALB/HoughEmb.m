img = imread('input\input_2048.png');
close all;
    [h, w, ~] = size(img);

% img = gpuArray(img);

tic
BW = rgb2gray(img);
BW = imbinarize(BW, 0.5);


[H,T,R] = hough(BW,'RhoResolution',0.5,'Theta',-90:0.5:89);
toc
figure;
imshow(BW);