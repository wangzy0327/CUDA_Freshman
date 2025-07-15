#include <stdio.h>
#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
#define sa7(i,j) sa7[((j)<<6) + (i)]
#define sb7(i,j) sb7[((j)<<6) + (i)]
#define MS_7 64
#define NS_7 64
#define KS_7 64
//M=N=K=2048
//GPU mySgemm Average elasped time: 0.002224 second, performance: 7723.7613 GFLOPS.
// TILE 64x64 cache blocking version, without register-level data re-use
// with memory coelascing on shared memory
// more workloads per thread. 4x4 micro kernel.
__global__  __launch_bounds__(256)
void mysgemm_v7(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    //获取每个block处理的数据块
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    A = &A(bx<<6,0);
    B = &B(0,by<<6);
    C = &C(bx<<6,by<<6);
    //64x64/256 per thread handle 16 data split
    __shared__ float sa7[KS_7*MS_7];
    __shared__ float sb7[KS_7*NS_7];
    int row = tx%64;  // [0,1,2,3,...,63]
    int col = tx/64;  // [0,1,2,3] -> *16+i
    int loop = 16;
    float tmp[4][4] = {0.};
    //循环展开
    int A_idx[4]; // 0_idx [0,1] [32,33] ... 1_idx [2,3] [34,35] ... 16_idx [30,31] [62,63]
    int B_idx[4]; // 0_idx [0,1] [32,33] ... 1_idx [2,3] [34,35] ... 16_idx [30,31] [62,63]
    A_idx[0] = tx%16*2; A_idx[1] = A_idx[0]+1; A_idx[2] = A_idx[0]+32; A_idx[3] = A_idx[0]+33;
    B_idx[0] = tx/16*2; B_idx[1] = B_idx[0]+1; B_idx[2] = B_idx[0]+32; B_idx[3] = B_idx[0]+33;
    for(int k_count = 0;k_count < K;k_count+=KS_7){
        #pragma unroll
        for(int i = 0;i < loop;i++){
            // if(i == 0){
            //     printf("Matrix A (iy:ix) = (%d,%d) handle (%d,%d)  \n",row,col,row,col);
            //     printf("Matrix B (iy:ix) = (%d,%d) handle (%d,%d) transpose to (%d,%d) \n",row,col,row,(row+col+i)%NS_7,(row+col+i)%NS_7,row);
            // }
            sa7(row,col*loop+i) = A(row,col*loop+i);
            sb7((row+col*loop+i)%NS_7,row) = B(row,(row+col*loop+i)%NS_7);
        }
        A += (lda<<6); B += 64;
        __syncthreads();
        #pragma unroll
        for(int inner_k = 0;inner_k < KS_7;inner_k++){
            for(int i = 0;i < 4;i++){
                for(int j = 0;j < 4;j++){
                    tmp[i][j] += sa7(A_idx[i],inner_k) * sb7(B_idx[j],inner_k);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i = 0;i < 4;i++){
        for(int j = 0;j < 4;j++){
            C(A_idx[i],B_idx[j]) = alpha * tmp[i][j] + beta * C(A_idx[i], B_idx[j]);
        }
    }
}

//M=N=K=2048
//GPU mySgemm Average elasped time: 0.002201 second, performance: 7804.604 GFLOPS.
// TILE 64x64 cache blocking version, without register-level data re-use
// with memory coelascing on shared memory
// more workloads per thread. 4x4 micro kernel.
// adopt vetorized load/store when mul-add operator
__global__  __launch_bounds__(256)
void mysgemm_v7_plus(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){
        int lda = M, ldb = K, ldc = M;
    //获取每个block处理的数据块
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    A = &A(bx<<6,0);
    B = &B(0,by<<6);
    C = &C(bx<<6,by<<6);
    //64x64/256 per thread handle 16 data split
    __shared__ float sa7[KS_7*MS_7];
    __shared__ float sb7[KS_7*NS_7];
    int row = tx%64;  // [0,1,2,3,...,63]
    int col = tx/64;  // [0,1,2,3] -> *16+i
    int loop = 16;
    float2 tmp[4][2] = {0.};
    //循环展开
    int A_idx[4]; // 0_idx [0,1] [32,33] ... 1_idx [2,3] [34,35] ... 16_idx [30,31] [62,63]
    int B_idx[4]; // 0_idx [0,1] [32,33] ... 1_idx [2,3] [34,35] ... 16_idx [30,31] [62,63]
    A_idx[0] = tx%16*2; A_idx[1] = A_idx[0]+1; A_idx[2] = A_idx[0]+32; A_idx[3] = A_idx[0]+33;
    B_idx[0] = tx/16*2; B_idx[1] = B_idx[0]+1; B_idx[2] = B_idx[0]+32; B_idx[3] = B_idx[0]+33;
    for(int k_count = 0;k_count < K;k_count+=KS_7){
        #pragma unroll
        for(int i = 0;i < loop;i++){
            // if(i == 0){
            //     printf("Matrix A (iy:ix) = (%d,%d) handle (%d,%d)  \n",row,col,row,col);
            //     printf("Matrix B (iy:ix) = (%d,%d) handle (%d,%d) transpose to (%d,%d) \n",row,col,row,(row+col+i)%NS_7,(row+col+i)%NS_7,row);
            // }
            sa7(row,col*loop+i) = A(row,col*loop+i);
            sb7((row+col*loop+i)%NS_7,row) = B(row,(row+col*loop+i)%NS_7);
        }
        A += (lda<<6); B += 64;
        __syncthreads();
        #pragma unroll
        for(int inner_k = 0;inner_k < KS_7;inner_k++){
            for(int i = 0;i < 4;i++){
                for(int j = 0;j < 4;j+=2){
                    tmp[i][j/2].x += sa7(A_idx[i],inner_k) * sb7(B_idx[j],inner_k);
                    tmp[i][j/2].y += sa7(A_idx[i],inner_k) * sb7(B_idx[j+1],inner_k);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i = 0;i < 4;i++){
        for(int j = 0;j < 4;j+=2){
            C(A_idx[i],B_idx[j]) = alpha * tmp[i][j/2].x + beta * C(A_idx[i], B_idx[j]);
            C(A_idx[i],B_idx[j+1]) = alpha * tmp[i][j/2].y + beta * C(A_idx[i], B_idx[j]);
        }
    }
}