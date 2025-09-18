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
    //获取每个block处理的数据块
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    A = &A(bx<<7,0);
    B = &B(0,by<<7);
    C = &C(bx<<7,by<<7);
    //256/32 8 warp per warp(warpSize=32) handle 128x128/8 = 128x16 (8x(16x16) 8 wmma
    __shared__ half sa12[KS_12*MS_12];
    __shared__ half sb12[KS_12*NS_12];
    int lane_id = tx&31; //[0-31]
    int warp_id = tx>>5; //[0-7]

    //128x16 share mem, per thread handle (128x16/256) = 8
    int row_a = (tx%64)<<1, col_a = (tx/64)<<2; // row_a [0,2,4,...,126], col_a [0,4,8,12]
    int row_b = (tx%2)<<3, col_b = (tx/2);  //row_b [0,8], col_b [0,1,2,...127]
    // int row_c = (tx<<4)<<3, col_c = (tx>>4)<<3; // 128x128/256 = 64 (8x8) row_c [0,8,16,...,120] col_c [0,8,16,...,120]
    half2 Av[4],Bv[4];
    // 声明列主序的WMMA片段
    nvcuda::wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> a_frag;  // A为列主序
    nvcuda::wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> b_frag;  // B为行主序 - 共享内存转置原数据块
    nvcuda::wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    nvcuda::wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    for(int k_count = 0;k_count < K;k_count+=KS_12){
        Av[0] = *((half2*)&A(row_a,col_a));
        Av[1] = *((half2*)&A(row_a,col_a+1));
        Av[2] = *((half2*)&A(row_a,col_a+2));
        Av[3] = *((half2*)&A(row_a,col_a+3));
        Bv[0] = *((half2*)&B(row_b,col_b));
        Bv[1] = *((half2*)&B(row_b+2*1,col_b));
        Bv[2] = *((half2*)&B(row_b+2*2,col_b));
        Bv[3] = *((half2*)&B(row_b+2*3,col_b));
        ((half2*)sa12)[row_a+col_a*4] = Av[0];
        ((half2*)sa12)[row_a+col_a*4+1] = Av[1];
        ((half2*)sa12)[row_a+col_a*4+2] = Av[2];
        ((half2*)sa12)[row_a+col_a*4+3] = Av[3];
        #pragma unroll
        for(int i = 0;i < 4;i++){
            sb12(col_b,row_b+i*2) = Bv[i].x;
            sb12(col_b,row_b+i*2+1) = Bv[i].y;
        }
        A += (lda<<4);B+=16;
        __syncthreads();
        #pragma unroll
        // 沿K维度循环（每次处理16个元素 WMMA_K大小）
        for(int inner_k = 0;inner_k < KS_12;inner_k+=WMMA_K){
            //这里KS_12 == WMMA_K ,如果KS_12是WMMA_K的数倍可以这样
            //share memory sa(16,128)KSxMS(列主序)  sb(16,128) KSxNS(行主序)
            nvcuda::wmma::fill_fragment(acc_frag, 0.0f);
            // 由于block内有8个warp,按照一个warp wmma处理16x16数据块,128x128/(16x16)/8 = 8,每个warp需要处理8个wmma块, 按照每个warp从sa中取1块，从sb中取8块来处理 (warp_id)
            // 加载sa子块（列主序：地址 = 行 + 列×领先维度）
            nvcuda::wmma::load_matrix_sync(a_frag, &sa12(warp_id*WMMA_M,0), MS_12);
            for(int i = 0;i < tx/warpSize;i++){
                nvcuda::wmma::load_matrix_sync(b_frag, &sb12(i*WMMA_N,0), NS_12);
                
                // 矩阵乘法：acc = A*B + acc
                nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
                // 加载C子块（列主序）
                nvcuda::wmma::load_matrix_sync(c_frag, &C(warp_id*WMMA_M, i*WMMA_N), ldc, nvcuda::wmma::mem_col_major);

                // 计算：C = alpha*acc + beta*C
                for (int j = 0; j < c_frag.num_elements; j++) {
                    c_frag.x[j] = alpha * acc_frag.x[j] + beta * c_frag.x[j];
                }

                // 存储结果（列主序）
                nvcuda::wmma::store_matrix_sync(&C(warp_id*WMMA_M, i*WMMA_N), c_frag, ldc, nvcuda::wmma::mem_col_major);                
            }
        }
    }
}


#define OFFSET(row, col, ld) ((row) * (ld) + (col))
// 列主序地址计算宏：A[i][j] = i + j * lda（lda为领先维度=行数）
// #define OFFSET_COL_MAJOR(row, col, lda) ((row) + (col) * (lda))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
// 针对const指针（如输入矩阵A和B）
#define CONST_FLOAT4(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])
// M=N=K=6144:
// GPU cublas Average elasped time: 0.004692 second, performance: 98870.572320 GFLOPS. cublasSgemmEx 使用TensorCore mma half,half,float
// GPU mySgemm Average elasped time: 0.010297 second, performance: 45048.741998 GFLOPS.
__global__  __launch_bounds__(256)
void mysgemm_v12_ano2(int M, int N, int K, float alpha, const half* a, const half* b, float beta, float* c) {

    const int BM = 128;
    const int BN = 256;
    const int BK = 32;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid >> 5;

    const int APAD = 8;
    const int BPAD = 8;

    __shared__ half s_a[BM][BK + APAD];
    __shared__ half s_b[BK][BN + BPAD];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2][4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[2][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c[4][4];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_c[i][j], 0.0f);
        }
    }

    int load_a_smem_m = (tid >> 2) << 1;
    int load_a_smem_k = (tid &  3) << 3;
    int load_b_smem_k = (tid >> 5) << 2;
    int load_b_smem_n = (tid & 31) << 3;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    int comp_c_frag_m = wid &  1;
    int comp_c_frag_n = wid >> 1;

    for (int bk = 0; bk < K / BK; bk++) {
        FLOAT4(s_a[load_a_smem_m    ][load_a_smem_k]) = CONST_FLOAT4(a[load_a_gmem_addr        ]);
        FLOAT4(s_a[load_a_smem_m + 1][load_a_smem_k]) = CONST_FLOAT4(a[load_a_gmem_addr +     K]);
        FLOAT4(s_b[load_b_smem_k    ][load_b_smem_n]) = CONST_FLOAT4(b[load_b_gmem_addr        ]);
        FLOAT4(s_b[load_b_smem_k + 1][load_b_smem_n]) = CONST_FLOAT4(b[load_b_gmem_addr +     N]);
        FLOAT4(s_b[load_b_smem_k + 2][load_b_smem_n]) = CONST_FLOAT4(b[load_b_gmem_addr + 2 * N]);
        FLOAT4(s_b[load_b_smem_k + 3][load_b_smem_n]) = CONST_FLOAT4(b[load_b_gmem_addr + 3 * N]);

        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;

        __syncthreads();

        wmma::load_matrix_sync(frag_a[0][0], &s_a[comp_c_frag_m * 64     ][ 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][1], &s_a[comp_c_frag_m * 64 + 16][ 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][2], &s_a[comp_c_frag_m * 64 + 32][ 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[0][3], &s_a[comp_c_frag_m * 64 + 48][ 0], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][0], &s_a[comp_c_frag_m * 64     ][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][1], &s_a[comp_c_frag_m * 64 + 16][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][2], &s_a[comp_c_frag_m * 64 + 32][16], BK + APAD);
        wmma::load_matrix_sync(frag_a[1][3], &s_a[comp_c_frag_m * 64 + 48][16], BK + APAD);

        wmma::load_matrix_sync(frag_b[0][0], &s_b[ 0][comp_c_frag_n * 64     ], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][1], &s_b[ 0][comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][2], &s_b[ 0][comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[0][3], &s_b[ 0][comp_c_frag_n * 64 + 48], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][0], &s_b[16][comp_c_frag_n * 64     ], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][1], &s_b[16][comp_c_frag_n * 64 + 16], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][2], &s_b[16][comp_c_frag_n * 64 + 32], BN + BPAD);
        wmma::load_matrix_sync(frag_b[1][3], &s_b[16][comp_c_frag_n * 64 + 48], BN + BPAD);

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                wmma::mma_sync(frag_c[i][j], frag_a[0][i], frag_b[0][j], frag_c[i][j]);
                wmma::mma_sync(frag_c[i][j], frag_a[1][i], frag_b[1][j], frag_c[i][j]);
            }
        }

        __syncthreads();
    }

    int store_c_gmem_m = by * BM + comp_c_frag_m * 64;
    int store_c_gmem_n = bx * BN + comp_c_frag_n * 64;
    int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::store_matrix_sync(&c[store_c_gmem_addr + i * 16 * N + j * 16], frag_c[i][j], N, wmma::mem_row_major);
        }
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