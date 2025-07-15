#include <stdio.h>
//M=N=K=2048:
//GPU mySgemm Average elasped time: 0.009025 second, performance: 1903.689545 GFLOPS.
__global__  __launch_bounds__(1024)
void mysgemm_v2(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    // 计算块指针偏移
    const float* Ablock = &A[(bx << 5) + 0 * lda];
    const float* Bblock = &B[0 + (by << 5) * ldb];
    float* Cblock = &C[(bx << 5) + (by << 5) * ldc];
    
    float tmp = 0.0f;
    
    for (int k_count = 0; k_count < K; k_count++) {
        tmp += Ablock[tx + k_count * lda] * Bblock[k_count + ty * ldb];
    }
    
    Cblock[tx + ty * ldc] = alpha * tmp + beta * Cblock[tx + ty * ldc];    
}