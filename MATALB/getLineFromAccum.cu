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
					accum[tid_temp] = 0;
				}
			}
		}

	}

	return;
}