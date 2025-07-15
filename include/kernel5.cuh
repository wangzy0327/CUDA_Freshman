#include <stdio.h>
#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
#define sa5(i,j) sa5[((j)<<5) + (i)]
#define sb5(i,j) sb5[((j)<<5) + (i)]
#define sb5_padding(i,j) sb5_padding[(((j)<<5)+1) + (i)]
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

//M=N=K=2048
//GPU mySgemm Average elasped time: 0.005195 second, performance: 3306.87 GFLOPS.
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
    __shared__ float sb5_padding[KS*(NS+1)];
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
        sb5_padding(col,row0) = B(row0,col);
        sb5_padding(col,row1) = B(row1,col);
        sb5_padding(col,row2) = B(row2,col);
        sb5_padding(col,row3) = B(row3,col);
        A += (lda<<5); B += 32;
        __syncthreads();
        //循环展开
        #pragma unroll
        for(int inner_k = 0;inner_k < KS;inner_k++){
            b00 = sb5_padding(col,inner_k);
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

//M=N=K=2048
//GPU mySgemm Average elasped time: 0.005396 second, performance: 3183.891133 GFLOPS.
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
    __shared__ float sb5_padding[KS*(NS+1)];
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
        sb5_padding(col0,row) = B(row,col0);
        sb5_padding(col1,row) = B(row,col1);
        sb5_padding(col2,row) = B(row,col2);
        sb5_padding(col3,row) = B(row,col3);
        A += (lda<<5); B += 32;
        __syncthreads();
        //循环展开
        #pragma unroll
        for(int inner_k = 0;inner_k < KS;inner_k++){
            a00 = sa5(row,inner_k);
            tmp[0] += sb5_padding(col0,inner_k) * a00;
            tmp[1] += sb5_padding(col1,inner_k) * a00;
            tmp[2] += sb5_padding(col2,inner_k) * a00;
            tmp[3] += sb5_padding(col3,inner_k) * a00;
        }
        __syncthreads();
    }
    C(row,col0) = alpha * tmp[0] + beta * C(row,col0);
    C(row,col1) = alpha * tmp[1] + beta * C(row,col1);
    C(row,col2) = alpha * tmp[2] + beta * C(row,col2);
    C(row,col3) = alpha * tmp[3] + beta * C(row,col3);
}

//M=N=K=2048
//GPU mySgemm Average elasped time: 0.004798 second, performance: 3581.00 GFLOPS.
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
    __shared__ float sb5_padding[KS*(NS+1)];
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
        sb5_padding(col0,row) = B(row,col0);
        sb5_padding(col1,row) = B(row,col1);
        sb5_padding(col2,row) = B(row,col2);
        sb5_padding(col3,row) = B(row,col3);
        A += (lda<<5); B += 32;
        __syncthreads();
        //循环展开
        #pragma unroll
        for(int inner_k = 0;inner_k < KS;inner_k++){
            a00 = sb5_padding(row,inner_k);
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

//M=N=K=2048
//GPU mySgemm Average elasped time: 0.003511 second, performance: 4892.787679 GFLOPS.
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