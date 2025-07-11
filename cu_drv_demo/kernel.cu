__global__ void kernel_function(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= 2;
    }
    printf("hello world idx %u : %f\n",idx,data[idx]);
}