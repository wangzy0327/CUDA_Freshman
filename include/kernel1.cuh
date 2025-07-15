#include <stdio.h>
//M=N=K=2048:
//GPU mySgemm Average elasped time: 0.008981 second, performance: 1912.996305 GFLOPS.
__global__  __launch_bounds__(1024)
void mysgemm_v1(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    if(ix < M && iy < N){
      float tmp = 0;
      for(int i = 0;i < K;i++){
        tmp += A[i*M+ix] * B[iy*K+i];
      }
      C[ix+iy*M] = alpha * tmp + beta * C[ix+iy*M];
    }
}