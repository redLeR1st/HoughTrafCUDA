


// Accum
#define M_PI 3.14159265358979323846   // pi
#define DEG2RAD (M_PI/180.0f)
__global__ void computeAccum(unsigned char* result, unsigned char* bw_image, unsigned int* accum, int w, int h, int w_accum, int h_accum, double hough_h)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	int tid = y * w + x;
    int _r = w * h * 0;
    int _g = w * h * 1;
    int _b = w * h * 2;
    int _a = w * h * 3;	

	if (x >= w || y >= h)
		return;

    double center_x = w / 2;
    double center_y = h / 2;

	if (result[tid + _r] > 128 && result[tid + _g] > 128 && result[tid + _b] > 128) {
		
		for (int t = 0; t < 180; t++) {
			double r = (((double)x - center_x) * cos((double)t * DEG2RAD)) + (((double)y - center_y) * sin((double)t * DEG2RAD));
			//accum[(int)((round(r + hough_h) * 180.0)) + t]++;
            int help = (int)((round(r + hough_h) * 180.0)) + t;
            int ind = 0;
            int alma = help / h_accum;
            int korte = help % h_accum;

            ind = alma + (korte * w_accum);
            atomicAdd(&accum[(int)((round(r + hough_h) * 180.0)) + t], 1);
            //atomicAdd(&accum[ind], 1);
		}
		bw_image[tid + _r] = 255;
		bw_image[tid + _g] = 255;
        bw_image[tid + _b] = 255;
		bw_image[tid + _a] = 255;
	}
	else {
		bw_image[tid + _r] = 0;
		bw_image[tid + _g] = 0;
        bw_image[tid + _b] = 0;
		bw_image[tid + _a] = 255;
	}
	

	return;
}

