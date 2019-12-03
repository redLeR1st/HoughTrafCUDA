close all;
%
im = imread('input\input_128.png');

im = gpuArray(im);

figure
imshow(im)
%% Initializing other parameters
theta = ((-90:90)./180) .* pi;
D = sqrt(size(im,1).^2 + size(im,2).^2);
HS = zeros(ceil(2.*D),numel(theta));
[y,x] = find(im);
y = y - 1;
x = x - 1;
figure
rho = cell(1,numel(x));

tic
for i = 1: numel(x)
    rho{i} = x(i).*cos(theta) + y(i).*sin(theta); % [-sqrt(2),sqrt(2)]*D rho interval
end
toc

% for i = 1: numel(x)
%     plot(theta,-rho{i})
%     hold on
% end

% for i = 1:numel(x)
%     rho{i} = rho{i} + D; % mapping rho from 0 to 2*sqrt(2)
%     rho{i} = floor(rho{i}) + 1;
%     for j = 1:numel(rho{i})
%         HS(rho{i}(j),j) = HS(rho{i}(j),j) + 1; 
%     end
% end
% figure
% imshow(HS)

