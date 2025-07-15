#include <cuda_runtime.h>
#include <stdio.h>
#include <cublas_v2.h>
#include "freshman.h"
#define TILEX 32
#define TILEY 32

#define CEIL_DIV(m,n) ( (m) + (n) - 1 ) / (n)

#define MS 32
#define NS 32
#define KS 32

#define MS_4 32
#define NS_4 32
#define KS_4 32

void sgemm_CPU(int M, int N, int K, float alpha, const float * MatA, const float * MatB, float beta, float * MatC)
{   
    //列主序
    for(int i = 0;i < M;i++){
        for(int j = 0;j < N;j++){
            const float* A = MatA + i;
            const float* B = MatB + j*K;
            // float* C = MatC + j*M;
            float sum = 0.0f;
            for(int k = 0;k < K;k++){
                // sum += MatA[i+k*M] * MatB[k+j*K];
                sum += A[k*M] * B[k];
                // C[i] += A[k*M] * B[k];
            }
            MatC[i+j*M] = alpha * sum + beta * MatC[i+j*M];
        }
    }
}

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

__global__  __launch_bounds__(1024)
void mysgemm_v3(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
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
            for (int i = 0; i < KS; i+=4) {
                tmp += sa[tx + i * MS] * sb[ty + i * NS];
                tmp += sa[tx + (i+1) * MS] * sb[ty + (i+1) * NS];
                tmp += sa[tx + (i+2) * MS] * sb[ty + (i+2) * NS];
                tmp += sa[tx + (i+3) * MS] * sb[ty + (i+3) * NS];
            }
            __syncthreads();
        }

        // 更新 C（考虑 alpha 和 beta）
        C[iy * M + ix] = alpha * tmp + beta * C[iy * M + ix];
    }
}

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

//bug
__global__ __launch_bounds__(256)
// __global__ __launch_bounds__(64)
 void mysgemm_v4_4(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){
    //blockDim = 256 != 1024 MS_4 x NS_4 = 32 x 32  per thread (32x32)/256=4
    // per block = 32 x 32
    int tx = threadIdx.x;
    //处理 数据索引
    int ix = blockIdx.x * MS_4 + (tx % MS_4);  //全局索引 (0 ≤ ix < M)
    int iy = blockIdx.y * NS_4 + (tx / MS_4 * 4); //一次处理4列

    if (ix < M && iy < N) {
        // printf("(iy:ix) = (%d,%d) handle (%d,%d),(%d,%d),(%d,%d),(%d,%d) \n",iy,ix,iy,ix,iy*(M),ix,iy*(M+1),ix,iy*(M+2),ix,iy*(M+3),ix);        
        __shared__ float sa[MS_4 * KS_4]; //32x32 1kx4=4k
        //填充 bank conflict
        __shared__ float sb[NS_4 * KS_4]; //32x32 1kx4=4k
        float tmp[4] = {0.,0.,0.,0.};
        for(int k_count = 0;k_count < K;k_count+=KS_4){
            int j = tx / MS_4;
            int i = tx % MS_4;
            int n_j = tx / KS_4;
            int n_i = tx % KS_4;
            sa[MS_4*j+i] = A[(k_count + j)*M+ix];
            sa[MS_4*(j+8)+i] = A[(k_count + j+8)*M+ix];
            sa[MS_4*(j+8*2)+i] = A[(k_count + j+8*2)*M+ix];
            sa[MS_4*(j+8*3)+i] = A[(k_count + j+8*3)*M+ix];
            sb[n_i*NS_4+n_j] = B[(iy )*K+k_count+ix];
            sb[n_i*NS_4+n_j+8] = B[(iy + 1)*K+k_count+ix];
            sb[n_i*NS_4+n_j+8*2] = B[(iy + 2)*K+k_count+ix];
            sb[n_i*NS_4+n_j+8*3] = B[(iy + 3)*K+k_count+ix];
            __syncthreads();
            for(int v = 0; v < KS_4;v++){
                // per thread handle 1M 4N
                // iy*4+[0-3]
                //                                                          0..1          ...    31        ... 32  33               ...   255
                // each thread M_idx = tx%32,tx/32*4+ N_idx = (tx/32)*4+[0-3] (0,[0,3]) (1,[0,3]) ... (31,[0,3]) ... (0,[4,7]),(1,[8,11]) ... (0,[28,31])
                //  int n_idx = j*4;  MS_4==NS_4 
                // printf("share mem (sy:sx) = (%d,%d) handle (%d,%d),(%d,%d),(%d,%d),(%d,%d) \n",MS_4*v+i,NS_4*v+(j*4),MS_4*v+i,NS_4*v+j*4,MS_4*v+i,NS_4*v+j*4+1,MS_4*v+i,NS_4*v+j*4+2,MS_4*v+i,NS_4*v+j*4+3); 
                tmp[0] += sa[MS_4*v+i] * sb[NS_4*v+j];
                tmp[1] += sa[MS_4*v+i] * sb[NS_4*v+j+8];
                tmp[2] += sa[MS_4*v+i] * sb[NS_4*v+j+8*2];
                tmp[3] += sa[MS_4*v+i] * sb[NS_4*v+j+8*3];
            }
            __syncthreads();
        }
        // if(iy == 0)
        //     printf("result (iy:ix) = (%d,%d)=%f, (iy+1:ix) = (%d,%d)=%f, (iy+2:ix) = (%d,%d)=%f, (iy+3:ix) = (%d,%d)=%f   \n",iy,ix,tmp[0],(iy+1),ix,tmp[1],(iy+2),ix,tmp[2],(iy+3),ix,tmp[3]);
        // 更新 C（考虑 alpha 和 beta）
        C[iy * M + ix] = alpha * tmp[0] + beta * C[iy * M + ix];
        C[(iy+1) * M + ix] = alpha * tmp[1] + beta * C[(iy+1) * M + ix];
        C[(iy+2) * M + ix] = alpha * tmp[2] + beta * C[(iy+2) * M + ix];
        C[(iy+3) * M + ix] = alpha * tmp[3] + beta * C[(iy+3) * M + ix];
    }
}


#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
#define sa5(i,j) sa5[((j)<<5) + (i)]
#define sb5(i,j) sb5[(((j)<<5)+1) + (i)]
#define MS 32
#define NS 32
#define KS 32
// cache blocking version, without register-level data re-use
// with memory coelascing on shared memory
// more workloads per thread. 4x1 micro kernel.
__global__  __launch_bounds__(256)
void mysgemm_v5(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    //row 为 要处理数据的行 每个block数据为1024,blockDim.x=256(每个block线程数),每个thread处理4行数据,数据列为[0-31],32列
    int row1 = (tx&7)<<2, row2 = row1+1, row3 = row1+2, row4 = row1+3, col = tx>>3;
    A = &A((bx<<5),0);  //当前block处理的A的数据块起始位置
    B = &B(0,(by<<5));
    C = &C((bx<<5),(by<<5));
    __shared__ float sa5[MS*KS];
    __shared__ float sb5[KS*NS];
    float Cres[4] = {0., 0., 0., 0.};
    float b00;
    for (int k_count = 0; k_count<K; k_count+=KS){
        sa5(row1,col)=A(row1,col);
        sa5(row2,col)=A(row2,col);
        sa5(row3,col)=A(row3,col);
        sa5(row4,col)=A(row4,col);
        sb5(col,row1)=B(row1,col);
        sb5(col,row2)=B(row2,col);
        sb5(col,row3)=B(row3,col);
        sb5(col,row4)=B(row4,col);
        A+=(lda<<5);B+=32;
        __syncthreads();
        #pragma unroll
        for (int inner_k_count=0;inner_k_count<KS;inner_k_count++){
            b00 = sb5(col,inner_k_count);
            Cres[0] += sa5(row1,inner_k_count) * b00;
            Cres[1] += sa5(row2,inner_k_count) * b00;
            Cres[2] += sa5(row3,inner_k_count) * b00;
            Cres[3] += sa5(row4,inner_k_count) * b00;
        }
        __syncthreads();
    }
    C(row1,col) = alpha * Cres[0] + beta*C(row1,col);
    C(row2,col) = alpha * Cres[1] + beta*C(row2,col);
    C(row3,col) = alpha * Cres[2] + beta*C(row3,col);
    C(row4,col) = alpha * Cres[3] + beta*C(row4,col);
}

__global__  __launch_bounds__(256)
void mysgemm_v5_ano(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    //获取每个block处理的数据块
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    A = &A(bx<<5,0);
    B = &B(0,by<<5);
    C = &C(bx<<5,by<<5);
    __shared__ float sa5[KS*MS];
    __shared__ float sb5[KS*(NS+1)];
    int row0,row1,row2,row3;
    int col = tx>>3;   // tx / 8  0...31
    row0 = (tx&7)<<2;  // 0, 4, 8, 12 ...
    row1 = row0 + 1;
    row2 = row0 + 2;
    row3 = row0 + 3;
    float tmp[4] = {0.,0.,0.,0.};
    float b00;
    for(int k_count = 0;k_count < K;k_count+=KS){
        sa5(row0,col) = A(row0,col);
        sa5(row1,col) = A(row1,col);
        sa5(row2,col) = A(row2,col);
        sa5(row3,col) = A(row3,col);
        sb5(col,row0) = B(row0,col);
        sb5(col,row1) = B(row1,col);
        sb5(col,row2) = B(row2,col);
        sb5(col,row3) = B(row3,col);
        A += (lda<<5); B += 32;
        __syncthreads();
        //循环展开
        #pragma unroll
        for(int inner_k = 0;inner_k < KS;inner_k++){
            b00 = sb5(col,inner_k);
            tmp[0] += sa5(row0,inner_k) * b00;
            tmp[1] += sa5(row1,inner_k) * b00;
            tmp[2] += sa5(row2,inner_k) * b00;
            tmp[3] += sa5(row3,inner_k) * b00;
        }
        __syncthreads();
    }
    C(row0,col) = alpha * tmp[0] + beta * C(row0,col);
    C(row1,col) = alpha * tmp[1] + beta * C(row1,col);
    C(row2,col) = alpha * tmp[2] + beta * C(row2,col);
    C(row3,col) = alpha * tmp[3] + beta * C(row3,col);
}


__global__  __launch_bounds__(256)
void mysgemm_v5_ano2(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    //获取每个block处理的数据块
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    A = &A(bx<<5,0);
    B = &B(0,by<<5);
    C = &C(bx<<5,by<<5);
    __shared__ float sa5[KS*MS];
    __shared__ float sb5[KS*(NS+1)];
    int row = tx&0x1F;  // 0...31
    int col0,col1,col2,col3;
    col0 = (tx>>5)*4;  // col0 ∈ {0,4,8,12,16,20,24,28}
    col1 = col0 + 1;
    col2 = col0 + 2;
    col3 = col0 + 3;
    float tmp[4] = {0.,0.,0.,0.};
    float a00;
    for(int k_count = 0;k_count < K;k_count+=KS){
        sa5(row,col0) = A(row,col0);
        sa5(row,col1) = A(row,col1);
        sa5(row,col2) = A(row,col2);
        sa5(row,col3) = A(row,col3);
        sb5(col0,row) = B(row,col0);
        sb5(col1,row) = B(row,col1);
        sb5(col2,row) = B(row,col2);
        sb5(col3,row) = B(row,col3);
        A += (lda<<5); B += 32;
        __syncthreads();
        //循环展开
        #pragma unroll
        for(int inner_k = 0;inner_k < KS;inner_k++){
            a00 = sa5(row,inner_k);
            tmp[0] += sb5(col0,inner_k) * a00;
            tmp[1] += sb5(col1,inner_k) * a00;
            tmp[2] += sb5(col2,inner_k) * a00;
            tmp[3] += sb5(col3,inner_k) * a00;
        }
        __syncthreads();
    }
    C(row,col0) = alpha * tmp[0] + beta * C(row,col0);
    C(row,col1) = alpha * tmp[1] + beta * C(row,col1);
    C(row,col2) = alpha * tmp[2] + beta * C(row,col2);
    C(row,col3) = alpha * tmp[3] + beta * C(row,col3);
}

//切分方向不同，减少 A B 往share mem 写数据冲突
__global__  __launch_bounds__(256)
void mysgemm_v5_ano_pro(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    //获取每个block处理的数据块
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    A = &A(bx<<5,0);
    B = &B(0,by<<5);
    C = &C(bx<<5,by<<5);
    __shared__ float sa5[KS*MS];
    __shared__ float sb5[KS*(NS+1)];
    int row = tx&0x1F;  // 0...31
    int col0,col1,col2,col3;
    col0 = (tx>>5)<<2;  // col0 ∈ {0,4,8,12,16,20,24,28} -> col0 = {0,1,2,3,4,5,6,7}
    col1 = col0 + 1;
    col2 = col0 + 2;
    col3 = col0 + 3;
    float tmp[4] = {0.,0.,0.,0.};
    float a00;
    for(int k_count = 0;k_count < K;k_count+=KS){
        sa5(row,col0) = A(row,col0);
        sa5(row,col1) = A(row,col1);
        sa5(row,col2) = A(row,col2);
        sa5(row,col3) = A(row,col3);
        sb5(col0,row) = B(row,col0);
        sb5(col1,row) = B(row,col1);
        sb5(col2,row) = B(row,col2);
        sb5(col3,row) = B(row,col3);
        A += (lda<<5); B += 32;
        __syncthreads();
        //循环展开
        #pragma unroll
        for(int inner_k = 0;inner_k < KS;inner_k++){
            a00 = sb5(row,inner_k);
            tmp[0] += sa5(col0,inner_k) * a00;
            tmp[1] += sa5(col1,inner_k) * a00;
            tmp[2] += sa5(col2,inner_k) * a00;
            tmp[3] += sa5(col3,inner_k) * a00;
        }
        __syncthreads();
    }
    C(col0,row) = alpha * tmp[0] + beta * C(col0,row);
    C(col1,row) = alpha * tmp[1] + beta * C(col1,row);
    C(col2,row) = alpha * tmp[2] + beta * C(col2,row);
    C(col3,row) = alpha * tmp[3] + beta * C(col3,row);
    // C(row,col0) = alpha * tmp[0] + beta * C(row,col0);
    // C(row,col1) = alpha * tmp[1] + beta * C(row,col1);
    // C(row,col2) = alpha * tmp[2] + beta * C(row,col2);
    // C(row,col3) = alpha * tmp[3] + beta * C(row,col3);
}

#define sb5(i,j) sb5[(((j)<<5)) + (i)]
__global__  __launch_bounds__(256)
void mysgemm_v5_ano2_pro(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    //获取每个block处理的数据块
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    A = &A(bx<<5,0);
    B = &B(0,by<<5);
    C = &C(bx<<5,by<<5);
    __shared__ float sa5[KS*MS];
    __shared__ float sb5[KS*NS];
    int row = tx&0x1F;  // 0...31
    int col0,col1,col2,col3;
    col0 = (tx>>5)*4;  // col0 ∈ {0,4,8,12,16,20,24,28}
    col1 = col0 + 1;
    col2 = col0 + 2;
    col3 = col0 + 3;
    float tmp[4] = {0.,0.,0.,0.};
    float a00;
    for(int k_count = 0;k_count < K;k_count+=KS){
        sa5(row,col0) = A(row,col0);
        sa5(row,col1) = A(row,col1);
        sa5(row,col2) = A(row,col2);
        sa5(row,col3) = A(row,col3);
        // printf("(iy:ix) = (%d,%d) handle (%d,%d) transpose to (%d,%d) \n",row,col0,row,(row+col0)%NS,(row+col0)%NS,row);
        sb5((row+col0)%NS,row) = B(row,(row+col0)%NS);
        sb5((row+col1)%NS,row) = B(row,(row+col1)%NS);
        sb5((row+col2)%NS,row) = B(row,(row+col2)%NS);
        sb5((row+col3)%NS,row) = B(row,(row+col3)%NS);
        A += (lda<<5); B += 32;
        __syncthreads();
        //循环展开
        #pragma unroll
        for(int inner_k = 0;inner_k < KS;inner_k++){
            a00 = sa5(row,inner_k);
            tmp[0] += sb5(col0,inner_k) * a00;
            tmp[1] += sb5(col1,inner_k) * a00;
            tmp[2] += sb5(col2,inner_k) * a00;
            tmp[3] += sb5(col3,inner_k) * a00;
        }
        __syncthreads();
    }
    C(row,col0) = alpha * tmp[0] + beta * C(row,col0);
    C(row,col1) = alpha * tmp[1] + beta * C(row,col1);
    C(row,col2) = alpha * tmp[2] + beta * C(row,col2);
    C(row,col3) = alpha * tmp[3] + beta * C(row,col3);
}


#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
#define sa6(i,j) sa6[((j)<<5) + (i)]
#define sb6(i,j) sb6[(((j)<<5)) + (i)]
#define MS_6 32
#define NS_6 32
#define KS_6 32
// cache blocking version, without register-level data re-used
// with memory coelascing on shared memory
// more workloads per thread. 4x1 micro kernel.
// adopt vetorized load/store
__global__  __launch_bounds__(256)
void mysgemm_v6(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    int row1 = (tx&7)<<2, row2 = row1+1, row3 = row1+2, row4 = row1+3, col = tx>>3;
    A = &A((bx<<5),0);
    B = &B(0,(by<<5));
    C = &C((bx<<5),(by<<5));
    __shared__ float sa6[MS_6*KS_6];
    __shared__ float sb6[KS_6*NS_6];
    float4 Av, Bv, Cv, Cres;
    Cres.x = 0., Cres.y = 0., Cres.z = 0., Cres.w = 0.;
    float b00;
    for (int k_count = 0; k_count<K; k_count+=KS_6){
        Av = *((float4 *)(&A(row1,col)));
        Bv = *((float4 *)(&B(row1,col)));
        ((float4 *)sa6)[tx] = Av;
        sb6(col,row1)=Bv.x;
        sb6(col,row2)=Bv.y;
        sb6(col,row3)=Bv.z;
        sb6(col,row4)=Bv.w;
        A+=(lda<<5);B+=32;
        __syncthreads();
        #pragma unroll
        for (int inner_k_count=0;inner_k_count<KS_6;inner_k_count++){
            b00 = sb6(col, inner_k_count);
            Cres.x += sa6(row1,inner_k_count) * b00;
            Cres.y += sa6(row2,inner_k_count) * b00;
            Cres.z += sa6(row3,inner_k_count) * b00;
            Cres.w += sa6(row4,inner_k_count) * b00;
        }
        __syncthreads();
    }
    Cv = *((float4 *)(&C(row1,col)));
    Cres.x = alpha * Cres.x + beta * Cv.x;
    Cres.y = alpha * Cres.y + beta * Cv.y;
    Cres.z = alpha * Cres.z + beta * Cv.z;
    Cres.w = alpha * Cres.w + beta * Cv.w;
    *(float4 *)(&(C(row1,col))) = Cres;
}

#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
#define sa7(i,j) sa7[((j)<<6) + (i)]
#define sb7(i,j) sb7[(((j)<<6)) + (i)]
#define MS_7 64
#define NS_7 64
#define KS_7 64
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



#define sa7(i,j) sa6[(((j)<<6)) + (i)]
#define sb7(i,j) sb6[(((j)<<6)) + (i)]
#define MS_7 64
#define NS_7 64
#define KS_7 64
#define MS_7_PAD 65    // 带填充的行大小（64+1）
#define KS_7_PAD 65    // 带填充的行大小（64+1）
// cache blocking version, without register-level data re-use
// with memory coelascing on shared memory
// more workloads per thread. 4x1 micro kernel.
__global__  __launch_bounds__(256)
void mysgemm_v7_ano(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    //row 为 要处理数据的行 每个block数据为4096,blockDim.x=256(每个block线程数),每个thread处理16行数据,数据列为[0-63],64列
    // 1. 计算线程处理的行和列基础索引（4行×4列=16元素）
    // 行基础索引：覆盖0~60（步长4），每个线程处理row_base ~ row_base+3（4行）
    int row_base = (tx & 15) << 2;  // tx&15取0~15，<<2后为0,4,8,...,60（共16个基础行）
    // 列基础索引：覆盖0~60（步长4），每个线程处理col_base ~ col_base+3（4列）
    int col_base = (tx >> 4) << 2;  // tx>>4取0~15，<<2后为0,4,8,...,60（共16个基础列）

    // 2. 定位当前block在全局矩阵中的起始地址
    A = &A((bx << 6), 0);          // A的当前块起始：行=bx*64，列=0（A是M×K矩阵）
    B = &B(0, (by << 6));          // B的当前块起始：行=0，列=by*64（B是K×N矩阵）
    C = &C((bx << 6), (by << 6));  // C的当前块起始：行=bx*64，列=by*64（C是M×N矩阵）

    // 3. 共享内存：存储A和B的子块（每个block私有）
    __shared__ float sa6[MS_7  * KS_7];  // 64×64=4096元素，存储A的子块（MS_6行×KS_6列）
    __shared__ float sb6[KS_7  * NS_7];  // 64×64=4096元素，存储B的子块（KS_6行×NS_6列）

    // 4. 寄存器：存储当前线程计算的16个结果（4×4）
    float Cres[4][4] = {0.0f};  // Cres[i][j]对应(row_base+i, col_base+j)的结果

    // 5. 分块遍历K维度，累加计算
    for (int k_count = 0; k_count < K; k_count += KS_7) {
        // 5.1 加载A和B的子块到共享内存（每个线程加载4×4=16元素）
        #pragma unroll  // 展开循环提高效率
        for (int i = 0; i < 4; i++) {  // 遍历4行
            #pragma unroll
            for (int j = 0; j < 4; j++) {  // 遍历4列
                int row = row_base + i;    // 行索引（0~63，不越界）
                int col = col_base + j;    // 列索引（0~63，不越界）
                // 加载A的元素到sa6：A的子块是MS_6×KS_6，行=row，列=col（当前K分块内的列）
                sa7(row, col) = A(row, col);
                // 加载B的元素到sb6：B的子块是KS_6×NS_6，行=col（当前K分块内的行），列=row
                sb7(col, row) = B(col, row);
            }
        }

        // 移动到下一个K分块（更新A和B的指针）
        A += lda * KS_7;  // A的列增加KS_6（lda是每行元素数，所以+lda*KS_6等价于列+KS_6）
        B += KS_7;        // B的行增加KS_6（B是行优先，每行K元素，+KS_6等价于行+KS_6）

        __syncthreads();  // 等待所有线程加载完共享内存

        // 5.2 计算当前K分块的贡献（累加矩阵乘法）
        #pragma unroll
        for (int k = 0; k < KS_7; k++) {  // 遍历当前K分块内的元素
            #pragma unroll
            for (int i = 0; i < 4; i++) {  // 4行
                #pragma unroll
                for (int j = 0; j < 4; j++) {  // 4列
                    // 矩阵乘法：C[i][j] += A[i][k] * B[k][j]
                    Cres[i][j] += sa7(row_base + i, k) * sb7(k, col_base + j);
                }
            }
        }

        __syncthreads();  // 等待当前K分块计算完成
    }

    // 6. 将结果写回全局内存（应用alpha和beta）
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int row = row_base + i;
            int col = col_base + j;
            C(row, col) = alpha * Cres[i][j] + beta * C(row, col);
        }
    }
}

#define sa7(i,j) sa7[((j)<<6) + (i)]
#define sb7(i,j) sb7[((j)<<6) + (i)]
#define MS_7 64
#define NS_7 64
#define KS_7 16
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
#define vload(v1,addr)\
    v1 = *((float4 *)(addr));
#define vstore(addr,v1)\
    *((float4 *)(addr)) = v1;
// cache blocking version, without register-level data re-use
// with memory coelascing on shared memory
// more workloads per thread. 4x4 micro kernel.
// adopt vetorized load/store
__global__  __launch_bounds__(256)
void mysgemm_v7_pro(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    int row_a = (tx&15)<<2, col_a = tx>>4;
    int row_b = (tx&3)<<2, col_b = tx>>2;
    int col_c = col_a<<2;
    int lda16 = lda<<4;
    A = &A((bx<<6),0);
    B = &B(0,(by<<6));
    C = &C((bx<<6),(by<<6));//the TB size is 64.
    __shared__ float sa7[1024];
    __shared__ float sb7[1024];
    float4 Av, Bv, Cv[4], Cres[4];
    memset(Cres, 0, sizeof(Cres));
    for (int k_count = 0; k_count<K; k_count+=KS_7){
        vload(Av, &A(row_a,col_a))
        vload(Bv, &B(row_b,col_b))
        ((float4 *)sa7)[tx] = Av;
        sb7(col_b,row_b)=Bv.x;
        sb7(col_b,row_b+1)=Bv.y;
        sb7(col_b,row_b+2)=Bv.z;
        sb7(col_b,row_b+3)=Bv.w;
        A+=lda16;B+=16;
        __syncthreads();
        #pragma unroll
        for (int inner_k_count=0;inner_k_count<KS_7;inner_k_count++){
            vload(Av, &sa7(row_a,inner_k_count))
            vload(Bv, &sb7(col_c,inner_k_count))
            vscal(Cres[0], Av, Bv.x)
            vscal(Cres[1], Av, Bv.y)
            vscal(Cres[2], Av, Bv.z)
            vscal(Cres[3], Av, Bv.w)
        }
        __syncthreads();
    }
    vload(Cv[0], &C(row_a,col_c))
    vload(Cv[1], &C(row_a,col_c+1))
    vload(Cv[2], &C(row_a,col_c+2))
    vload(Cv[3], &C(row_a,col_c+3))
    simd_axpby(Cres[0],alpha,Cres[0],beta,Cv[0])
    simd_axpby(Cres[1],alpha,Cres[1],beta,Cv[1])
    simd_axpby(Cres[2],alpha,Cres[2],beta,Cv[2])
    simd_axpby(Cres[3],alpha,Cres[3],beta,Cv[3])

    vstore(&C(row_a,col_c), Cres[0])
    vstore(&C(row_a,col_c+1), Cres[1])
    vstore(&C(row_a,col_c+2), Cres[2])
    vstore(&C(row_a,col_c+3), Cres[3])
}

void test_mysemm_v1(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){
    // cudaDeviceSynchronize();
    int blockX = 32, blockY = 32;
    dim3 blockDim(blockX,blockY);
    dim3 gridDim(CEIL_DIV(M,blockX),CEIL_DIV(N,blockY));
    mysgemm_v1<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    cudaDeviceSynchronize();
    // cudaStreamSynchronize(0);
}

void test_mysemm_v2(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){
    // cudaDeviceSynchronize();
    int blockX = 32, blockY = 32;
    dim3 blockDim(blockX,blockY);
    dim3 gridDim(CEIL_DIV(M,blockX),CEIL_DIV(N,blockY));
    mysgemm_v2<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    cudaDeviceSynchronize();
}

void test_mysemm_v3(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){
    // cudaDeviceSynchronize();
    int blockX = 32, blockY = 32;
    dim3 blockDim(blockX,blockY);
    dim3 gridDim(CEIL_DIV(M,blockX),CEIL_DIV(N,blockY));
    mysgemm_v3<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    cudaDeviceSynchronize();
}

void test_mysemm_v4(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){
    // cudaDeviceSynchronize();
    int blockX = MS, blockY = NS;
    dim3 blockDim(blockX,blockY);
    dim3 gridDim(CEIL_DIV(M,blockX),CEIL_DIV(N,blockY));
    mysgemm_v4<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    cudaDeviceSynchronize();
}

void test_mysemm_v5(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){
    // cudaDeviceSynchronize();
    int blockX = MS, blockY = NS;
    // dim3 blockDim(1024);
    dim3 blockDim(256);//x4
    // dim3 blockDim(64);//x4
    dim3 gridDim(CEIL_DIV(M,blockX),CEIL_DIV(N,blockY));
    mysgemm_v5_ano2_pro<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    cudaDeviceSynchronize();
}

void test_mysemm_v6(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){
    // cudaDeviceSynchronize();
    int blockX = MS_6, blockY = NS_6;
    // dim3 blockDim(1024);
    dim3 blockDim(256);//x4
    // dim3 blockDim(64);//x4
    dim3 gridDim(CEIL_DIV(M,blockX),CEIL_DIV(N,blockY));
    mysgemm_v6<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    cudaDeviceSynchronize();
}

void test_mysemm_v7(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){
    // cudaDeviceSynchronize();
    int blockX = 64, blockY = 64;
    // dim3 blockDim(1024);
    dim3 blockDim(256);//x4
    // dim3 blockDim(64);//x4
    dim3 gridDim(CEIL_DIV(M,blockX),CEIL_DIV(N,blockY));
    mysgemm_v7<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    cudaDeviceSynchronize();
}


int main(int argc,char **argv)
{
  // set up device
  initDevice(0);
  int kernel=1;
  if(argc>=2)
    kernel=atoi(argv[1]);
  int SIZE[24];
  for (int i=0;i<24;i++) SIZE[i]=(i+1)<<8;
  if (kernel<0||kernel>11) {
    printf("Please enter a valid kernel number (0-11).\n");
    exit(-2);
  }
  int m, n, k,max_size;
  int N=1, upper_limit;
  if (kernel<=7&&kernel!=0) upper_limit=8;
  else upper_limit=(sizeof(SIZE)/sizeof(int));
  max_size=SIZE[upper_limit-1];
  float* A_host = NULL,*B_host = NULL, *C_host = NULL,*C_from_dev = NULL,*C_from_dev_lib = NULL;
  float* A_dev = NULL,*B_dev = NULL,*C_dev = NULL,*C_dev_lib = NULL;
  float alpha = 1.0, beta = 0.;//two arbitary input parameters

  int nElem = max_size*max_size;
  int nBytes = sizeof(float)*max_size*max_size;

  CHECK(cudaHostAlloc((float**)&A_host,nBytes,cudaHostAllocDefault));
  CHECK(cudaHostAlloc((float**)&B_host,nBytes,cudaHostAllocDefault));
  CHECK(cudaHostAlloc((float**)&C_host,nBytes,cudaHostAllocDefault));
  CHECK(cudaHostAlloc((float**)&C_from_dev,nBytes,cudaHostAllocDefault));
  CHECK(cudaHostAlloc((float**)&C_from_dev_lib,nBytes,cudaHostAllocDefault));
  //Malloc
  // A_host=(float*)malloc(nBytes);
  // B_host=(float*)malloc(nBytes);
  // C_host=(float*)malloc(nBytes);
  // C_from_dev=(float*)malloc(nBytes);
  // C_from_dev_lib=(float*)malloc(nBytes);

  CHECK(cudaMalloc((float**)&A_dev,nBytes));
  CHECK(cudaMalloc((float**)&B_dev,nBytes));
  CHECK(cudaMalloc((float**)&C_dev,nBytes));
  CHECK(cudaMalloc((float**)&C_dev_lib,nBytes));

  memset(C_host,0,nBytes);
  memset(C_from_dev,0,nBytes);
  memset(C_from_dev_lib,0,nBytes);
  cudaMemset(C_dev,0,nBytes);
  cudaMemset(C_dev_lib,0,nBytes);

  initialData(A_host,nElem);
  initialData(B_host,nElem);

  CHECK(cudaMemcpyAsync(A_dev,A_host,nBytes,cudaMemcpyHostToDevice,0));
  CHECK(cudaMemcpyAsync(B_dev,B_host,nBytes,cudaMemcpyHostToDevice,0));
  // CHECK(cudaMemcpy(A_dev,A_host,nBytes,cudaMemcpyHostToDevice));
  // CHECK(cudaMemcpy(B_dev,B_host,nBytes,cudaMemcpyHostToDevice));

  cudaSharedMemConfig MemConfig = cudaSharedMemBankSizeFourByte;
  cublasHandle_t handle; cublasCreate(&handle);
  cudaEvent_t beg_lib, end_lib, beg, end;
  cudaEventCreate(&beg); cudaEventCreate(&end);
  cudaEventCreate(&beg_lib); cudaEventCreate(&end_lib);
  float elapsed_time;
  
  printf("--------------------------------------------\n");
  

  for(int i_count = upper_limit-1;i_count < upper_limit;i_count++){
//   for(int i_count = 0;i_count < upper_limit;i_count++){
//   for(int i_count = 0;i_count < 1;i_count++){
    m=n=k=SIZE[i_count];
    printf("\nM=N=K=%d:\n",m);
    cudaEventRecord(beg_lib);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A_dev, m, B_dev, k, &beta, C_dev_lib, m);
    cudaEventRecord(end_lib);
    cudaEventSynchronize(beg_lib);
    cudaEventSynchronize(end_lib);
    cudaEventElapsedTime(&elapsed_time, beg_lib, end_lib);
    elapsed_time /= 1000.;
    printf("GPU cublas Average elasped time: %f second, performance: %f GFLOPS.\n", elapsed_time,2.*1e-9*m*n*k/elapsed_time);
    if(i_count < 3){
      //数据量低于 3*256 时，计算CPU校对结果
      double iStart=cpuSecond();
      sgemm_CPU(m, n, k, alpha, A_host, B_host, beta, C_host);
      double iElaps=cpuSecond()-iStart;
      printf("CPU Execution Time elapsed %f sec\n",iElaps);
    }
    CHECK(cudaMemcpyAsync(C_from_dev_lib, C_dev_lib, nBytes, cudaMemcpyDeviceToHost,0));
    // CHECK(cudaMemcpy(C_from_dev_lib, C_dev_lib, nBytes, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    elapsed_time = 0.0f;
    cudaEventRecord(beg);
    for(int n_count = 0;n_count < N;n_count++){
      switch (kernel)
      {
        case 0: cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A_dev, m, B_dev, k, &beta, C_dev, m);break;
        case 1: test_mysemm_v1(m,n,k,alpha,A_dev,B_dev,beta,C_dev);break;
        case 2: test_mysemm_v2(m,n,k,alpha,A_dev,B_dev,beta,C_dev);break;
        case 3: test_mysemm_v3(m,n,k,alpha,A_dev,B_dev,beta,C_dev);break;
        case 4: test_mysemm_v4(m,n,k,alpha,A_dev,B_dev,beta,C_dev);break;
        case 5: test_mysemm_v5(m,n,k,alpha,A_dev,B_dev,beta,C_dev);break;
        case 6: test_mysemm_v6(m,n,k,alpha,A_dev,B_dev,beta,C_dev);break;
        case 7: test_mysemm_v7(m,n,k,alpha,A_dev,B_dev,beta,C_dev);break;
        default:
          break;
      }
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.;
    printf("GPU mySgemm Average elasped time: %f second, performance: %f GFLOPS.\n", elapsed_time/N,2.*1e-9*N*m*n*k/elapsed_time);
    fflush(stdout);
    CHECK(cudaMemcpyAsync(C_from_dev, C_dev, nBytes, cudaMemcpyDeviceToHost,0));
    // CHECK(cudaMemcpy(C_from_dev, C_dev, nBytes, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    // if(i_count < 3)
    //   checkResult(C_host,C_from_dev_lib,nElem);
    // else
    checkResult(C_from_dev,C_from_dev_lib,nElem);
    // printMatrix(C_from_dev_lib,n,1);
  }

  cudaFree(A_dev);
  cudaFree(B_dev);
  cudaFree(C_dev);
  cudaFree(C_dev_lib);
  cudaFreeHost(A_host);
  cudaFreeHost(B_host);
  cudaFreeHost(C_host);
  cudaFreeHost(C_from_dev);
  cudaFreeHost(C_from_dev_lib);
  cublasDestroy_v2(handle);
  return 0;
}