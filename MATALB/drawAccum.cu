
// draw ACC
__global__ void drawAccum(unsigned int* accum, unsigned char* image_accum, int w_accum, int h_accum, int* max) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	int tid = y * w_accum + x;

    int _r = w_accum * h_accum * 0;
    int _g = w_accum * h_accum * 1;
    int _b = w_accum * h_accum * 2;
    int _a = w_accum * h_accum * 3;	

	if (x >= w_accum || y >= h_accum)
		return;
	int pixel_value = 0;
	if (!accum[tid] == 0) {
		image_accum[tid + _r] = (unsigned char)((*max) / accum[tid]) * 255;
		image_accum[tid + _g] = (unsigned char)((*max) / accum[tid]) * 255;
		image_accum[tid + _b] = (unsigned char)((*max) / accum[tid]) * 255;
		image_accum[tid + _a] = 255;
	}
	else if (accum[tid] == -1) {
		image_accum[tid + _r] = 255;
		image_accum[tid + _g] = (unsigned char)((*max) / accum[tid]) * 255;
		image_accum[tid + _b] = (unsigned char)((*max) / accum[tid]) * 255;
		image_accum[tid + _a] = 255;
	}
	else {
		image_accum[tid + _r] = 0;
		image_accum[tid + _g] = 0;
		image_accum[tid + _b] = 0;
		image_accum[tid + _a] = 255;
	}

}