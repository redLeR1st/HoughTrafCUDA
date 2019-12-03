
//MAx
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