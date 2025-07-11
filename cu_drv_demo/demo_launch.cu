#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK_ERROR(call) \
{\
    const CUresult error = call;\
    if (error != CUDA_SUCCESS) { \
        const char* errorName = NULL;\
        cuGetErrorName(error,&errorName);\
        const char* errorString = NULL;\
        cuGetErrorString(error,&errorString);\
        printf("CUDA error Name : %s , line %d\n description : %s \n", errorName, __LINE__, errorString); \
        exit(error); \
    }\
}    

int main() {
    CUresult err;

    // Initialize CUDA
    err = cuInit(0);
    CUDA_CHECK_ERROR(err);

    // Create CUDA context
    CUcontext cuContext;
    err = cuCtxCreate(&cuContext, 0, 0);
    CUDA_CHECK_ERROR(err);

    // Load module
    CUmodule cuModule;
    err = cuModuleLoad(&cuModule, "kernel.ptx");
    CUDA_CHECK_ERROR(err);

    // Get kernel function
    CUfunction cuFunction;
    err = cuModuleGetFunction(&cuFunction, cuModule, "_Z15kernel_functionPfi");
    CUDA_CHECK_ERROR(err);

    // Allocate device memory
    CUdeviceptr d_data;
    int size = 256;
    err = cuMemAlloc(&d_data, size * sizeof(float));
    float *h_data = (float *)malloc(size*sizeof(float));
    for(int i = 0;i < size;i++)
        h_data[i] = i;
    cuMemcpyHtoD(d_data, h_data, size*sizeof(float));
    CUDA_CHECK_ERROR(err);

    // Set kernel parameters
    void *args[] = { &d_data, &size };
    err = cuLaunchKernel(cuFunction, 1, 1, 1,  // grid dims
                         256, 1, 1,           // block dims
                         0, NULL,             // shared mem and stream
                         args, 0);            // kernel arguments
    CUDA_CHECK_ERROR(err);

    // Synchronize to wait for kernel completion
    err = cuCtxSynchronize();
    CUDA_CHECK_ERROR(err);

    // Cleanup
    cuMemFree(d_data);
    cuCtxDestroy(cuContext);

    return 0;
}
