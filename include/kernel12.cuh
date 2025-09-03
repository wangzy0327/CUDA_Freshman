#include <stdio.h>
#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
#define sa12(i,j) sa12[((j)<<7) + (i)]
#define sb12(i,j) sb12[((j)<<7) + (i)]
#define MS_12 128
#define NS_12 128
#define KS_12 16
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32

#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;


//v1 += v2 * s3, vector scaling
#define vscal(v1, v2, s3)\
    v1.x+=v2.x*s3;\
    v1.y+=v2.y*s3;\
    v1.z+=v2.z*s3;\
    v1.w+=v2.w*s3;
//v1 = alpha * v2 + beta * v3, simd fma
#define simd_axpby(v1, alpha, v2, beta, v3)\
    v1.x=alpha*v2.x+beta*v3.x;\
    v1.y=alpha*v2.y+beta*v3.y;\
    v1.z=alpha*v2.z+beta*v3.z;\
    v1.w=alpha*v2.w+beta*v3.w;

//M=N=K=2048 MS=NS=128 KS=8 2048 half(8k) SM share mem(96k) block share mem(48k) Block dim  = 256 per SM max 24 block
//grid.dim = 2048*2048/128/128 = 1024 block / 80 SM  per SM = 12.8 block ,  12 block can active improve SM warp active numbers
// block max share memory 48KB  128x16x2x2(half)=8K 
__global__  __launch_bounds__(256)
void mysgemm_v12_ano(int M, int N, int K, float alpha, const half* A, const half* B, float beta, float* C){
// 列主序矩阵的领先维度（LD = 行数，与列主序存储匹配）
    int lda = M;    // A是M×K列主序，领先维度=行数M
    int ldb = K;    // B是K×N列主序，领先维度=行数K
    int ldc = M;    // C是M×N列主序，领先维度=行数M

    // 计算当前Warp在M和N维度的索引（与原逻辑一致）
    // warpSize 定义是在cuda_bf16.h,cuda_fp16.h中
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // 声明列主序的WMMA片段
    nvcuda::wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> a_frag;  // A为列主序
    nvcuda::wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> b_frag;  // B为列主序
    nvcuda::wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    nvcuda::wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    nvcuda::wmma::fill_fragment(acc_frag, 0.0f);

    // 沿K维度循环（每次处理16个元素）
    for (int k = 0; k < K; k += WMMA_K) {
        // 计算A和B子块的列主序地址索引
        int a_row = warpM * WMMA_M;       // A子块的起始行（M维度）
        int a_col = k;                    // A子块的起始列（K维度）
        int b_row = k;                    // B子块的起始行（K维度）
        int b_col = warpN * WMMA_N;       // B子块的起始列（N维度）

        // 边界检查（确保子块在矩阵范围内）
        if (a_row + WMMA_M <= M && a_col + WMMA_K <= K && 
            b_row + WMMA_K <= K && b_col + WMMA_N <= N) {

            // 加载A子块（列主序：地址 = 行 + 列×领先维度）
            nvcuda::wmma::load_matrix_sync(a_frag, &A[a_row + a_col * lda], lda);
            // 加载B子块（列主序）
            nvcuda::wmma::load_matrix_sync(b_frag, &B[b_row + b_col * ldb], ldb);

            // 矩阵乘法：acc = A*B + acc
            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // 计算C和输出结果的索引
    int c_row = warpM * WMMA_M;  // C子块的起始行（M维度）
    int c_col = warpN * WMMA_N;  // C子块的起始列（N维度）

    // 边界检查
    if (c_row + WMMA_M <= M && c_col + WMMA_N <= N) {
        // 加载C子块（列主序）
        nvcuda::wmma::load_matrix_sync(c_frag, &C[c_row + c_col * ldc], ldc, nvcuda::wmma::mem_col_major);

        // 计算：C = alpha*acc + beta*C
        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        // 存储结果（列主序）
        nvcuda::wmma::store_matrix_sync(&C[c_row + c_col * ldc], c_frag, ldc, nvcuda::wmma::mem_col_major);
    }
}

// M=N=K=6144:
// GPU cublas Average elasped time: 0.004700 second, performance: 98699.585016 GFLOPS.
// GPU mySgemm Average elasped time: 0.005153 second, performance: 90013.569027 GFLOPS.
// 核函数：基于WMMA的矩阵乘法（列主序，无填充，无half2向量操作）
__global__ void mysgemm_v12(int M, int N, int K, float alpha, const half* A, const half* B, float beta, float* C) {
    // 列主序矩阵的领先维度（LD = 行数，与列主序存储匹配）
    int lda = M;    // A是M×K列主序，领先维度=行数M
    int ldb = K;    // B是K×N列主序，领先维度=行数K
    int ldc = M;    // C是M×N列主序，领先维度=行数M

    // 计算当前Warp在M和N维度的索引（与原逻辑一致）
    // warpSize 定义是在cuda_bf16.h,cuda_fp16.h中
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // 声明列主序的WMMA片段
    nvcuda::wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> a_frag;  // A为列主序
    nvcuda::wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> b_frag;  // B为列主序
    nvcuda::wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    nvcuda::wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    nvcuda::wmma::fill_fragment(acc_frag, 0.0f);

    // 沿K维度循环（每次处理16个元素）
    for (int k = 0; k < K; k += WMMA_K) {
        // 计算A和B子块的列主序地址索引
        int a_row = warpM * WMMA_M;       // A子块的起始行（M维度）
        int a_col = k;                    // A子块的起始列（K维度）
        int b_row = k;                    // B子块的起始行（K维度）
        int b_col = warpN * WMMA_N;       // B子块的起始列（N维度）

        // 边界检查（确保子块在矩阵范围内）
        if (a_row + WMMA_M <= M && a_col + WMMA_K <= K && 
            b_row + WMMA_K <= K && b_col + WMMA_N <= N) {

            // 加载A子块（列主序：地址 = 行 + 列×领先维度）
            nvcuda::wmma::load_matrix_sync(a_frag, &A[a_row + a_col * lda], lda);
            // 加载B子块（列主序）
            nvcuda::wmma::load_matrix_sync(b_frag, &B[b_row + b_col * ldb], ldb);

            // 矩阵乘法：acc = A*B + acc
            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // 计算C和输出结果的索引
    int c_row = warpM * WMMA_M;  // C子块的起始行（M维度）
    int c_col = warpN * WMMA_N;  // C子块的起始列（N维度）

    // 边界检查
    if (c_row + WMMA_M <= M && c_col + WMMA_N <= N) {
        // 加载C子块（列主序）
        nvcuda::wmma::load_matrix_sync(c_frag, &C[c_row + c_col * ldc], ldc, nvcuda::wmma::mem_col_major);

        // 计算：C = alpha*acc + beta*C
        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        // 存储结果（列主序）
        nvcuda::wmma::store_matrix_sync(&C[c_row + c_col * ldc], c_frag, ldc, nvcuda::wmma::mem_col_major);
    }
}