#include <stdio.h>
#define MS 32
#define NS 32
#define KS 32
//M=N=K=2048
//GPU mySgemm Average elasped time: 0.009039 second, performance: 1900.730759 GFLOPS.
__global__  __launch_bounds__(1024)
void mysgemm_v3(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){
    int tx = threadIdx.x, ty = threadIdx.y;
    int ix = blockIdx.x * blockDim.x + tx;  // 全局行索引 (0 ≤ ix < M)
    int iy = blockIdx.y * blockDim.y + ty;  // 全局列索引 (0 ≤ iy < N)

    if (ix < M && iy < N) {
        __shared__ float sa[MS * KS];
        __shared__ float sb[NS * KS];
        float tmp = 0.0f;

        for (int k_count = 0; k_count < K; k_count += KS) {
            // 加载 A 的分块到 sa（列主序）
            if (k_count + ty < K) {
                sa[tx + ty * MS] = A[(k_count + ty) * M + ix];
            } else {
                sa[tx + ty * MS] = 0.0f;  // 超出 K 的部分补 0
            }

            // 加载 B 的分块到 sb（列主序）
            if (k_count + tx < K) {
                sb[ty + tx * NS] = B[iy * K + (k_count + tx)];
            } else {
                sb[ty + tx * NS] = 0.0f;  // 超出 K 的部分补 0
            }
            __syncthreads();

            // 计算局部矩阵乘法
            //#pragma unroll
            //不进行循环展开
            for (int i = 0; i < KS; i+=1) {
                tmp += sa[tx + i * MS] * sb[ty + i * NS];
            }
            __syncthreads();
        }

        // 更新 C（考虑 alpha 和 beta）
        C[iy * M + ix] = alpha * tmp + beta * C[iy * M + ix];
    }
}