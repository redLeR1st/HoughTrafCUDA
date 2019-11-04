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

__global__ void rippleKernel(unsigned char* result, unsigned int* accum, int w, int h, int w_accum, int h_accum, double hough_h);
__global__ void getLines(unsigned int* accum, int w_accum, int h_accum, int* dev_points, int* max);
__global__ void plotLines(unsigned char* result, int w, int h, int* points);
__global__ void findMax(unsigned int* accum, int w_accum, int h_accum, int* dev_points, int* max);

int main(int argc, char** argv)
{

	int _img_w;
	int _img_h;

	unsigned char* result_temp = new unsigned char[2048 * 2048 * 4];

	std::cout << "DEBUG0" << std::endl;
	
	char* name_of_the_input = argv[1];
	// READING STUFF
	readRGBImageFromFile(name_of_the_input, result_temp, _img_w, _img_h);


	unsigned char* result = new unsigned char[_img_w * _img_h * 4];


	int N = _img_h > _img_w ? _img_h : _img_w;

	unsigned char* dev_result;

	int w_accum = 180;

	double hough_h = ((sqrt(2.0) * (double)N) / 2.0);
	int h_accum = hough_h * 2.0; // -r -> +r 

	unsigned int* accum = new unsigned int[w_accum * h_accum];
	unsigned int* dev_accum;

	int* points = new int[4];
	int* dev_points;

	int* max = new int;
	int* dev_max;

	cudaMalloc((void**)&dev_result, _img_h * _img_w * 4 * sizeof(unsigned char));

	cudaMalloc((void**)&dev_accum, w_accum * h_accum * sizeof(unsigned int));
	cudaMemset(dev_accum, 0, w_accum * h_accum * sizeof(unsigned int));

	cudaMalloc((void**)&dev_points, 4 * sizeof(int));
	cudaMemset(dev_points, 0, 4 * sizeof(int));

	cudaMalloc((void**)&dev_max, sizeof(int));
	cudaMemset(dev_max, -999999, sizeof(int));

	dim3 blockDim = dim3(BLOCKDIM, BLOCKDIM, 1);
	dim3 gridDim = dim3((N + BLOCKDIM - 1) / BLOCKDIM, (N + BLOCKDIM - 1) / BLOCKDIM, 1);

	
	
	cudaMemcpy(dev_result, result_temp, _img_w * _img_h * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);
	std::cout << "DEBUG1" << std::endl;
	// ALGORITHM
	rippleKernel << <gridDim, blockDim >> > (dev_result, dev_accum, _img_w, _img_h, w_accum, h_accum, hough_h);
	getLines << <gridDim, blockDim >> > (dev_accum, w_accum, h_accum, dev_points, dev_max);
	findMax << <gridDim, blockDim >> > (dev_accum, w_accum, h_accum, dev_points, dev_max);

	cudaMemcpy(points, dev_points, 4 * sizeof(int), cudaMemcpyDeviceToHost);

	int x1, y1, x2, y2;
	x1 = y1 = x2 = y2 = 0;
	int x = points[0];
	int y = points[1];

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

	cudaMemcpy(dev_points, points, 4 * sizeof(int), cudaMemcpyHostToDevice);
	plotLines << <gridDim, blockDim >> > (dev_result, _img_w, _img_h, dev_points);

	// WRITE OUT STUFF
	cudaMemcpy(result, dev_result, _img_h * _img_w * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	writeRGBImageToFile("result.png", result, _img_w, _img_h);
	
	cudaMemcpy(max, dev_max, sizeof(int), cudaMemcpyDeviceToHost);
	
	std::cout << "max: " << (*max) << std::endl;

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
	// ------------------



	for (int i = 0; i < 4; i++) {
		std::cout << points[i] << " ";
	}
	std::cout << "\n";

	//cudaMemcpy(accum, dev_accum, w_accum * h_accum * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	//writeRGBImageToFile("accum.png", accum, w_accum, h_accum);

	//writeRGBImageToFile("result.png", result, N, N);
	

	return 0;
}

__global__ void plotLines(unsigned char* result, int w, int h, int* points) {

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
		result[tid * 4 + 0] = 255;
		result[tid * 4 + 1] = 0;
		result[tid * 4 + 2] = 0;
		result[tid * 4 + 3] = 255;
	}

	return;
}


__global__ void findMax(unsigned int* accum, int w_accum, int h_accum, int* dev_points, int* max) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	int tid = y * w_accum + x;

	if (x >= w_accum || y >= h_accum)
		return;

	int temp_max;
	if (max[0] == (int)accum[tid]) {
		atomicExch(&dev_points[0], x);
		atomicExch(&dev_points[1], y);
	}
	
	//atomicExch(&temp_max, max[0]);
	//
	//atomicCAS(&max[0], (int)accum[tid], x);
	//atomicExch(&dev_points[0], max[0]);
	//atomicExch(&max[0], temp_max);
	//
	//atomicCAS(&max[0], (int)accum[tid], y);
	//atomicExch(&dev_points[1], max[0]);
	//atomicExch(&max[0], temp_max);

	return;
}

__global__ void getLines(unsigned int* accum, int w_accum, int h_accum, int* dev_points, int* max)
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
	
		//dev_points[0] = x1;
		//dev_points[1] = y1;
		//dev_points[2] = x2;
		//dev_points[3] = y2;
	}

	return;
}

__global__ void rippleKernel(unsigned char* result, unsigned int* accum, int w, int h, int w_accum, int h_accum, double hough_h)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	int tid = y * w + x;
	
	if (x >= w || y >= h)
		return;

    double center_x = w / 2;
    double center_y = h / 2;

	if (result[tid * 4] > 128 && result[tid * 4 + 1]  > 128 && result[tid * 4 + 2] > 128 && result[tid * 4 + 3] > 128) {
		
		for (int t = 0; t < 180; t++) {
			double r = (((double)x - center_x) * cos((double)t * DEG2RAD)) + (((double)y - center_y) * sin((double)t * DEG2RAD));
			//accum[(int)((round(r + hough_h) * 180.0)) + t]++;
			atomicAdd(&accum[(int)((round(r + hough_h) * 180.0)) + t], 1);
		}

		
		//result[tid * 4 + 1] = 0;
		//result[tid * 4 + 2] = 0;
		//result[tid * 4 + 3] = 255;
	}
	
	/*
	float dist = sqrtf((x - w / 2.0f) * (x - w / 2.0f) + (y - h / 2.0f) * (y - h / 2.0f));
	float value = (cosf(dist / waveLength * CUDART_PI_F * 2) + 1) * 127;

	if (x < w && y < h)
	{
		result[tid * 4] = value;
		result[tid * 4 + 1] = value;
		result[tid * 4 + 2] = value;
		result[tid * 4 + 3] = 255;
	}
	*/
	return;
}




















/*#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
*/