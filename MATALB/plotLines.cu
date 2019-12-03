__global__ void plotLines(unsigned char* result, int w, int h, int* points, unsigned char blue, unsigned char green, unsigned char read) {

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	int tid = y * w + x;
    
    int _r = w * h * 0;
    int _g = w * h * 1;
    int _b = w * h * 2;
    int _a = w * h * 3;	

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
        result[tid + _b] = blue;
		result[tid + _g] = green;
		result[tid + _r] = read;
		result[tid + _a] = 255;
	}

	return;
}
