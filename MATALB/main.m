function  [accum] = main( I, num_of_lines )
    close all;
    
    DEG2RAD = 3.14159265358979323846/180;
    %//#define N           512
    %#define BLOCKDIM    16
    N = 512;
    BLOCKDIM = 16;
    image = imread(I);
    
    [img_h, img_w, ~] = size(image);
    
    if img_h > img_w
        N = img_h;
    else
        N = img_w;
    end
    
    image(:,:,4) = 255;
    
    %image = permute(image, [2 1 3]);
    image = image(:)';
    image = gpuArray(image);
    
    % 1. Create CUDAKernel object.
    computeAccum = parallel.gpu.CUDAKernel('computeAccum.ptx','computeAccum.cu');
    drawAccum = parallel.gpu.CUDAKernel('drawAccum.ptx','drawAccum.cu');
    findMaxInAccum = parallel.gpu.CUDAKernel('findMaxInAccum.ptx','findMaxInAccum.cu');
    getLineFromAccum = parallel.gpu.CUDAKernel('getLineFromAccum.ptx','getLineFromAccum.cu');
    plotLines = parallel.gpu.CUDAKernel('plotLines.ptx','plotLines.cu');

    % 2. Set object properties.
    computeAccum.ThreadBlockSize = [BLOCKDIM, BLOCKDIM, 1];
    computeAccum.GridSize = [int16((N + BLOCKDIM - 1) / BLOCKDIM), int16((N + BLOCKDIM - 1) / BLOCKDIM), 1];
    drawAccum.ThreadBlockSize = [BLOCKDIM, BLOCKDIM 1];
    drawAccum.GridSize = [int16((N + BLOCKDIM - 1) / BLOCKDIM), int16((N + BLOCKDIM - 1) / BLOCKDIM), 1];
    findMaxInAccum.ThreadBlockSize = [BLOCKDIM, BLOCKDIM 1];
    findMaxInAccum.GridSize = [int16((N + BLOCKDIM - 1) / BLOCKDIM), int16((N + BLOCKDIM - 1) / BLOCKDIM), 1];
    getLineFromAccum.ThreadBlockSize = [BLOCKDIM, BLOCKDIM 1];
    getLineFromAccum.GridSize = [int16((N + BLOCKDIM - 1) / BLOCKDIM), int16((N + BLOCKDIM - 1) / BLOCKDIM), 1];
    plotLines.ThreadBlockSize = [BLOCKDIM, BLOCKDIM 1];
    plotLines.GridSize = [int16((N + BLOCKDIM - 1) / BLOCKDIM), int16((N + BLOCKDIM - 1) / BLOCKDIM), 1];

    
%   int w_accum = 180;
% 
% 	int N = _img_h > _img_w ? _img_h : _img_w;
% 
% 	double hough_h = ((sqrt(2.0) * (double)N) / 2.0);
% 	int h_accum = hough_h * 2.0; 
    w_accum = 180;
    
    hough_h = ((sqrt(2.0) * double(N)) / 2.0)
    h_accum = int16(hough_h * 2.0); 
    
    bw_image = uint8(zeros(img_h , img_w, 4));
    bw_image = permute(bw_image, [2 1 3]);
    bw_image = gpuArray(bw_image(:)');
    
    accum = uint32(zeros(h_accum , w_accum));
    accum = accum';
    accum = gpuArray(accum(:)');
    
    b = 50;
    g = 50;
    r = 255;

    tic
    % __global__ void computeAccum(unsigned char* result, unsigned char* bw_image, unsigned int* accum, int w, int h, int w_accum, int h_accum, double hough_h);
    [result, bw, accum] = feval(computeAccum, image, bw_image, accum, img_w, img_h, w_accum, h_accum, hough_h);
    
    
    %show_img(result, img_h, img_w, 4);
    %show_img(bw, img_h, img_w, 4);
    %show_img(accum, w_accum, h_accum, 0);
    points = gpuArray(int32(zeros(4,1)));
    max_val = gpuArray(int32(zeros(1,1)));
    
    for i = 1:num_of_lines
       % findMaxInAccum << <gridDim, blockDim >> > (dev_accum, w_accum, h_accum, dev_points, dev_max); 
        [accum, points, max_val] = feval(findMaxInAccum, accum, w_accum, h_accum, points, max_val);

        %getLineFromAccum << <gridDim, blockDim >> > (dev_accum, w_accum, h_accum, dev_points, dev_max);
        [accum, points, max_val] = feval(getLineFromAccum, accum, w_accum, h_accum, points, max_val);
        
        %max_val
        %points
        
        x1 = int16(0);
        y1 = int16(0);
        x2 = int16(0);
        y2 = int16(0);
		x = int16(points(1));
		y = int16(points(2));

        if x >= 45 && x <= 135
		
			%y = (r - x cos(t)) / sin(t)  
			x1 = 0;
			y1 = ((double(y - (h_accum / 2))) - ((x1 - (img_w / 2)) * cos(double(x * DEG2RAD)))) / sin(double(x * DEG2RAD)) + (img_h / 2);
			x2 = img_w - 0;
			y2 = ((double(y - (h_accum / 2))) - ((x2 - (img_w / 2)) * cos(double(x * DEG2RAD)))) / sin(double(x * DEG2RAD)) + (img_h / 2);
		
		else
		
			%x = (r - y sin(t)) / cos(t);  
			y1 = 0;
			x1 = ((double(y - (h_accum / 2))) - ((y1 - (img_h / 2)) * sin(double(x * DEG2RAD)))) / cos(double(x * DEG2RAD)) + (img_w / 2);
			y2 = img_h - 0;
			x2 = ((double(y - (h_accum / 2))) - ((y2 - (img_h / 2)) * sin(double(x * DEG2RAD)))) / cos(double(x * DEG2RAD)) + (img_w / 2);
        end
        points(1) = int32(x1);
		points(2) = int32(y1);
		points(3) = int32(x2);
		points(4) = int32(y2)
        
        if b + 30 < 255 && r - 20 >= 0
			b = b + 30;
			r = r + 20;
        end
        %plotLines << <gridDim, blockDim >> > (dev_result, _img_w, _img_h, dev_points, b, g, r);
		%plotLines << <gridDim, blockDim >> > (dev_bw_image, _img_w, _img_h, dev_points, b, g, r);
        [result] = feval(plotLines, result, img_w, img_h, points, b, g, r);
        [bw] = feval(plotLines, bw, img_w, img_h, points, b, g, r);
        
        %show_img(result, img_h, img_w, 4);
        %show_img(bw, img_h, img_w, 4);
        
    end
    toc
    accum_img = gpuArray(uint8(zeros(h_accum, w_accum, 4)));
    %drawAccum << <gridDim, blockDim >> > (dev_accum, dev_image_accum, w_accum, h_accum, dev_max);
    [valami, accum_img, megvalami] = feval(drawAccum, accum, accum_img, w_accum, h_accum, max_val);
    show_img(accum, w_accum, h_accum, 0);
    show_img(accum_img, h_accum, w_accum, 4);
    show_img(result, img_h, img_w, 4);
    show_img(bw, img_h, img_w, 4);
    accum = reshape(accum, h_accum, w_accum);
    
    
    
    % 3. Call feval with defined inputs.
    % g1 = gpuArray(in1); % Input gpuArray.
    % g2 = gpuArray(in2); % Input gpuArray.
    %
    % result = feval(k,g1,g2);
end

