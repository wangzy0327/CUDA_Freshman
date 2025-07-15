#include <stdio.h>
#define MS 32
#define NS 32
#define KS 32
#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
#define sa4(i,j) sa4[((j)<<5) + (i)]
#define sb4(i,j) sb4[((j)<<5) + (i)]
//M=N=K=2048
//GPU mySgemm Average elasped time: 0.006504 second, performance: 2641.574403 GFLOPS.
//转换为1维计算索引 很容易出错，且是列主序 最好不要这样写
__global__  __launch_bounds__(1024)
void mysgemm_v4(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int ix = blockIdx.x * blockDim.x + tx;  // 全局行索引 (0 ≤ ix < M)
    int iy = blockIdx.y * blockDim.y + ty;  // 全局列索引 (0 ≤ iy < N)

    if (ix < M && iy < N) {
        __shared__ float sa[MS * KS];
        //填充 bank conflict
        __shared__ float sb[NS * (KS+1)];
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
                sb[ty + tx * (NS+1)] = B[iy * K + (k_count + tx)];
            } else {
                sb[ty + tx * (NS+1)] = 0.0f;  // 超出 K 的部分补 0
            }
            __syncthreads();

            // 计算局部矩阵乘法 循环展开
            #pragma unroll
            for (int i = 0; i < KS; i+=1) {
                tmp += sa[tx + i * MS] * sb[ty + i * (NS+1)];
                // tmp += sa[tx + (i+1) * MS] * sb[ty + (i+1) * (NS+1)];
                // tmp += sa[tx + (i+2) * MS] * sb[ty + (i+2) * (NS+1)];
                // tmp += sa[tx + (i+3) * MS] * sb[ty + (i+3) * (NS+1)];
            }
            __syncthreads();
        }

        // 更新 C（考虑 alpha 和 beta）
        C[iy * M + ix] = alpha * tmp + beta * C[iy * M + ix];
    }
}

// cache blocking version, without register-level data re-use
// with memory coelascing on shared memory
__global__  __launch_bounds__(1024)
void mysgemm_v4_ano(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    int row = tx&31, col = tx>>5;
    A = &A((bx<<5),0);
    B = &B(0,(by<<5));
    C = &C((bx<<5),(by<<5));
    __shared__ float sa4[MS*KS];
    __shared__ float sb4[KS*NS];
    float tmp=0.;
    for (int k_count = 0; k_count<K; k_count+=KS){
        sa4(row,col)=A(row,col);
        sb4(col,row)=B(row,col);
        A+=(lda<<5);B+=32;
        __syncthreads();
        for (int inner_k_count=0;inner_k_count<KS;inner_k_count++){
            tmp += sa4(row,inner_k_count) * sb4(col,inner_k_count);
        }
        __syncthreads();
    }
    C(row,col) = alpha * tmp + beta*C(row,col);
}