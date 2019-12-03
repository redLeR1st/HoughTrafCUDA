#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"

#include "ImageIO.h"

#include <iostream>
#include <ctime>
#include <fstream>

using namespace std;

//#define N           512
#define BLOCKDIM    16


#define M_PI 3.14159265358979323846   // pi
#define DEG2RAD (M_PI/180.0f)

__global__ void computeAccum(unsigned char* result, unsigned char* bw_image, unsigned int* accum, int w, int h, int w_accum, int h_accum, double hough_h);
__global__ void findMaxInAccum(unsigned int* accum, int w_accum, int h_accum, int* dev_points, int* max);
__global__ void plotLines(unsigned char* result, int w, int h, int* points, unsigned char b, unsigned char g, unsigned char r);
__global__ void getLineFromAccum(unsigned int* accum, int w_accum, int h_accum, int* dev_points, int* max);
__global__ void drawAccum(unsigned int* accum, unsigned char* image_accum, int w_accum, int h_accum, int* max);

int main(int argc, char** argv)
{

	int number_of_lines = std::atoi(argv[2]);

	int _img_w;
	int _img_h;

	unsigned char* result_temp = new unsigned char[2048 * 2048 * 4];
	
	char* name_of_the_input = argv[1];
	//cudaHostAlloc(&result_temp, sizeof(unsigned char) * 2048 * 2048 * 4, cudaHostAllocMapped);
	//cudaHostRegister(&result_temp, sizeof(unsigned char) * 2048 * 2048 * 4, cudaHostRegisterDefault);
	// READING STUFF
	readRGBImageFromFile(name_of_the_input, result_temp, _img_w, _img_h);


	unsigned char* result = new unsigned char[_img_w * _img_h * 4];
	unsigned char* dev_result;

	unsigned char* bw_image = new unsigned char[_img_w * _img_h * 4];
	unsigned char* dev_bw_image;

	int w_accum = 180;

	int N = _img_h > _img_w ? _img_h : _img_w;

	double hough_h = ((sqrt(2.0) * (double)N) / 2.0);
	int h_accum = hough_h * 2.0; 

	unsigned int* accum = new unsigned int[w_accum * h_accum];
	unsigned int* dev_accum;
	unsigned char* image_accum = new unsigned char[w_accum * h_accum * 4];
	unsigned char* dev_image_accum;


	int* points = new int[4];
	int* dev_points;

	int* max = new int;
	int* dev_max;

	// OPTIMIZATION######################################
	

	//cudaHostAlloc(&result, sizeof(unsigned char) * _img_w * _img_h * 4, cudaHostAllocMapped);
	//cudaHostRegister(&result, sizeof(unsigned char) * _img_w * _img_h * 4, cudaHostRegisterDefault);
	//
	//cudaHostAlloc(&bw_image, sizeof(unsigned char) * _img_w * _img_h * 4, cudaHostAllocMapped);
	//cudaHostRegister(&bw_image, sizeof(unsigned char) * _img_w * _img_h * 4, cudaHostRegisterDefault);
	//
	//cudaHostAlloc(&accum, sizeof(unsigned int) * w_accum * h_accum, cudaHostAllocMapped);
	//cudaHostRegister(&accum, sizeof(unsigned int) * w_accum * h_accum, cudaHostRegisterDefault);
	//
	//cudaHostAlloc(&image_accum, sizeof(unsigned char) * w_accum * h_accum * 4, cudaHostAllocMapped);
	//cudaHostRegister(&image_accum, sizeof(unsigned char) * w_accum * h_accum * 4, cudaHostRegisterDefault);
	//
	//cudaHostAlloc(&points, sizeof(int) * 4, cudaHostAllocMapped);
	//cudaHostRegister(&points, sizeof(int) * 4, cudaHostRegisterDefault);
	//
	//cudaHostAlloc(&max, sizeof(int), cudaHostAllocMapped);
	//cudaHostRegister(&max, sizeof(int), cudaHostRegisterDefault);
	
	// OPTIMIZATION######################################*/
	

	cudaMalloc((void**)&dev_result, _img_h * _img_w * 4 * sizeof(unsigned char));

	cudaMalloc((void**)&dev_bw_image, _img_h * _img_w * 4 * sizeof(unsigned char));

	cudaMalloc((void**)&dev_accum, w_accum * h_accum * sizeof(unsigned int));
	cudaMemset(dev_accum, 0, w_accum * h_accum * sizeof(unsigned int));

	cudaMalloc((void**)&dev_image_accum, w_accum * h_accum * 4 * sizeof(unsigned char));

	cudaMalloc((void**)&dev_points, 4 * sizeof(int));
	cudaMemset(dev_points, 0, 4 * sizeof(int));

	cudaMalloc((void**)&dev_max, sizeof(int));

	dim3 blockDim = dim3(BLOCKDIM, BLOCKDIM, 1);
	dim3 gridDim = dim3((N + BLOCKDIM - 1) / BLOCKDIM, (N + BLOCKDIM - 1) / BLOCKDIM, 1);

	std::cout << "BLOCK: " << BLOCKDIM << std::endl;
	std::cout << "GRID : " << (N + BLOCKDIM - 1) / BLOCKDIM << std::endl;
	
	cudaMemcpy(dev_result, result_temp, _img_w * _img_h * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);

	unsigned char b = 50;
	unsigned char g = 50;
	unsigned char r = 255;

	// ALGORITHM
	int threshlod = 20;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	cudaEventRecord(start); //-------------------------------------------START

	computeAccum << <gridDim, blockDim >> > (dev_result, dev_bw_image, dev_accum, _img_w, _img_h, w_accum, h_accum, hough_h); // count the accum image
	for (int i = 0; i < number_of_lines; i++) {
		cudaMemset(dev_max, -999999, sizeof(int));
		findMaxInAccum << <gridDim, blockDim >> > (dev_accum, w_accum, h_accum, dev_points, dev_max);

		//if (i == 0) {
		//	drawAccum << <gridDim, blockDim >> > (dev_accum, dev_image_accum, w_accum, h_accum, dev_max);
		//	cudaMemcpy(image_accum, dev_image_accum, w_accum * h_accum * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		//	char* accum_name = "0_accum.png";
		//	accum_name[0] = '0' + i;
		//	writeRGBImageToFile(accum_name, image_accum, w_accum, h_accum);
		//}
		
		getLineFromAccum << <gridDim, blockDim >> > (dev_accum, w_accum, h_accum, dev_points, dev_max);

		cudaMemcpy(max, dev_max, sizeof(int), cudaMemcpyDeviceToHost);
		std::cout << "max: " << (*max) << std::endl;

		if ((*max) < threshlod) {
			std::cout << "no more line above the threshold: " << threshlod << std::endl;
			break;
		}

		cudaMemcpy(points, dev_points, 4 * sizeof(int), cudaMemcpyDeviceToHost);

		int x1, y1, x2, y2;
		x1 = y1 = x2 = y2 = 0;
		int x = points[0];
		int y = points[1];

		std::cout << "x " << x << std::endl;
		std::cout << "y " << y << std::endl;

		if (x >= 45 && x <= 135)
		{
			//y = (r - x cos(t)) / sin(t)  
			x1 = 0;
			y1 = ((double)(y - (h_accum / 2)) - ((x1 - (_img_w / 2)) * cos(x * DEG2RAD))) / sin(x * DEG2RAD) + (_img_h / 2);
			x2 = _img_w - 0;
			y2 = ((double)(y - (h_accum / 2)) - ((x2 - (_img_w / 2)) * cos(x * DEG2RAD))) / sin(x * DEG2RAD) + (_img_h / 2);
		}
		else
		{
			//x = (r - y sin(t)) / cos(t);  
			y1 = 0;
			x1 = ((double)(y - (h_accum / 2)) - ((y1 - (_img_h / 2)) * sin(x * DEG2RAD))) / cos(x * DEG2RAD) + (_img_w / 2);
			y2 = _img_h - 0;
			x2 = ((double)(y - (h_accum / 2)) - ((y2 - (_img_h / 2)) * sin(x * DEG2RAD))) / cos(x * DEG2RAD) + (_img_w / 2);
		}
		points[0] = x1;
		points[1] = y1;
		points[2] = x2;
		points[3] = y2;

		if (b + 30 < 255 && r - 20 >= 0) {
			b += 30;
			r -= 20;
		}

		// cudaMemcpy(dev_points, points, 4 * sizeof(int), cudaMemcpyHostToDevice);
		// plotLines << <gridDim, blockDim >> > (dev_result, _img_w, _img_h, dev_points, b, g, r);
		// plotLines << <gridDim, blockDim >> > (dev_bw_image, _img_w, _img_h, dev_points, b, g, r);

		

		for (int i = 0; i < 4; i++) {
			std::cout << points[i] << " ";
		}
		std::cout << "\n";
	}

	cudaEventRecord(stop); // ---------------------------------------------STOP
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	// WRITE OUT STUFF
	cudaMemcpy(result, dev_result, _img_h * _img_w * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	writeRGBImageToFile("result.png", result, _img_w, _img_h);

	// WRITE OUT STUFF
	cudaMemcpy(bw_image, dev_bw_image, _img_h * _img_w * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	writeRGBImageToFile("result_bw.png", bw_image, _img_w, _img_h);

	// write accum to a file
	cudaMemcpy(accum, dev_accum, w_accum * h_accum * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	fstream myfile;
	
	myfile.open("example.txt", fstream::out);
	
	for (int i = 0; i < N; i++) //This variable is for each row below the x 
	{
		for (int j = 0; j < 180; j++)
		{
			
			int tid = i * 180 + j;

			myfile << accum[tid] << " ";
		}
		myfile<< ";"<< std::endl;
	}
	myfile.close();
	cout << "TIME: " << milliseconds << endl;
	cout << "SIZE: " << N << endl;
	cout << "hough: " << hough_h << endl;

	return 0;
}

__global__ void plotLines(unsigned char* result, int w, int h, int* points, unsigned char blue, unsigned char green, unsigned char read) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	int tid = y * w + x;
	
	if (x >= w || y >= h)
		return;

	int x1 = points[0];
	int y1 = points[1];
	int x2 = points[2];
	int y2 = points[3];

	//ax + by + c = 0
	int a = y1 - y2;
	int b = x2 - x1;
	int c = (x1 - x2) * y1 + (y2 - y1) * x1;

	double diff = a * x + b * y + c;

	if (-10 < diff && diff < 500) {
		result[tid * 4 + 0] = blue;
		result[tid * 4 + 1] = green;
		result[tid * 4 + 2] = read;
		result[tid * 4 + 3] = 255;
	}

	return;
}


__global__ void getLineFromAccum(unsigned int* accum, int w_accum, int h_accum, int* dev_points, int* max) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	int tid = y * w_accum + x;

	if (x >= w_accum || y >= h_accum)
		return;

	int temp_max;
	if (max[0] == (int)accum[tid]) {
		dev_points[0] = x;
		dev_points[1] = y;
		// DELETE THE LINE FROM ACCU
		
		int filter_size = 30;
		for (int i = x - filter_size / 2; i < x + filter_size / 2; i++) {
			for (int j = y - filter_size / 2; j < y + filter_size / 2; j++) {
				if (i < w_accum && j < h_accum) {
					int tid_temp = j * w_accum + i;
					accum[tid_temp] = -1;
				}
			}
		}

	}

	return;
}

__global__ void findMaxInAccum(unsigned int* accum, int w_accum, int h_accum, int* dev_points, int* max)
{

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	int tid = y * w_accum + x;

	if (x >= w_accum || y >= h_accum)
		return;

	int old = (int)accum[tid];
	atomicMax(&max[0], (int)accum[tid]);

	if (old == max[0]) {
	
		atomicExch(&dev_points[0], x);
		atomicExch(&dev_points[1], y);

	}

	return;
}

__global__ void computeAccum(unsigned char* result, unsigned char* bw_image, unsigned int* accum, int w, int h, int w_accum, int h_accum, double hough_h)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	int tid = y * w + x;
	
	if (x >= w || y >= h)
		return;

    double center_x = w / 2;
    double center_y = h / 2;

	if (result[tid * 4] > 128 && result[tid * 4 + 1]  > 128 && result[tid * 4 + 2] > 128) {
		
		for (int t = 0; t < 180; t++) {
			double r = (((double)x - center_x) * cos((double)t * DEG2RAD)) + (((double)y - center_y) * sin((double)t * DEG2RAD));
			//accum[(int)((round(r + hough_h) * 180.0)) + t]++;
			atomicAdd(&accum[(int)((round(r + hough_h) * 180.0)) + t], 1);
		}
		bw_image[tid * 4 + 0] = 255;
		bw_image[tid * 4 + 1] = 255;
		bw_image[tid * 4 + 2] = 255;
		bw_image[tid * 4 + 3] = 255;
	}
	else {
		bw_image[tid * 4 + 0] = 0;
		bw_image[tid * 4 + 1] = 0;
		bw_image[tid * 4 + 2] = 0;
		bw_image[tid * 4 + 3] = 255;
	}
	

	return;
}

__global__ void drawAccum(unsigned int* accum, unsigned char* image_accum, int w_accum, int h_accum, int* max) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	int tid = y * w_accum + x;

	if (x >= w_accum || y >= h_accum)
		return;
	int pixel_value = 0;
	if (!accum[tid] == 0) {
		image_accum[tid * 4 + 0] = (unsigned char)((*max) / accum[tid]) * 255;
		image_accum[tid * 4 + 1] = (unsigned char)((*max) / accum[tid]) * 255;
		image_accum[tid * 4 + 2] = (unsigned char)((*max) / accum[tid]) * 255;
		image_accum[tid * 4 + 3] = 255;
	}
	else if (accum[tid] == -1) {
		image_accum[tid * 4 + 0] = 255;
		image_accum[tid * 4 + 1] = (unsigned char)((*max) / accum[tid]) * 255;
		image_accum[tid * 4 + 2] = (unsigned char)((*max) / accum[tid]) * 255;
		image_accum[tid * 4 + 3] = 255;
	}
	else {
		image_accum[tid * 4 + 0] = 0;
		image_accum[tid * 4 + 1] = 0;
		image_accum[tid * 4 + 2] = 0;
		image_accum[tid * 4 + 3] = 255;
	}

}