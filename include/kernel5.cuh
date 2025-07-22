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
        if(k_count == 0 && bx ==0 && by == 0){
            printf("threadIdx %d = transpose to (%d,%d) bank(%d),(%d,%d) bank(%d),(%d,%d) bank(%d),(%d,%d) bank(%d) \n",tx,(row+col0)%NS,row,((row<<5)+((row+col0)%NS))%32,(row+col1)%NS,row,((row<<5)+((row+col1)%NS))%32,(row+col2)%NS,row,((row<<5)+((row+col2)%NS))%32,(row+col3)%NS,row,((row<<5)+((row+col3)%NS))%32);
        }
        sa5(row,col0) = A(row,col0);
        sa5(row,col1) = A(row,col1);
        sa5(row,col2) = A(row,col2);
        sa5(row,col3) = A(row,col3);

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
/*

mysgemm_v5_ano2_pro

warp内bank store 不冲突

threadIdx 0 = transpose to (0,0) bank(0),(1,0) bank(1),(2,0) bank(2),(3,0) bank(3) 
threadIdx 1 = transpose to (1,1) bank(1),(2,1) bank(2),(3,1) bank(3),(4,1) bank(4) 
threadIdx 2 = transpose to (2,2) bank(2),(3,2) bank(3),(4,2) bank(4),(5,2) bank(5) 
threadIdx 3 = transpose to (3,3) bank(3),(4,3) bank(4),(5,3) bank(5),(6,3) bank(6) 
threadIdx 4 = transpose to (4,4) bank(4),(5,4) bank(5),(6,4) bank(6),(7,4) bank(7) 
threadIdx 5 = transpose to (5,5) bank(5),(6,5) bank(6),(7,5) bank(7),(8,5) bank(8) 
threadIdx 6 = transpose to (6,6) bank(6),(7,6) bank(7),(8,6) bank(8),(9,6) bank(9) 
threadIdx 7 = transpose to (7,7) bank(7),(8,7) bank(8),(9,7) bank(9),(10,7) bank(10) 
threadIdx 8 = transpose to (8,8) bank(8),(9,8) bank(9),(10,8) bank(10),(11,8) bank(11) 
threadIdx 9 = transpose to (9,9) bank(9),(10,9) bank(10),(11,9) bank(11),(12,9) bank(12) 
threadIdx 10 = transpose to (10,10) bank(10),(11,10) bank(11),(12,10) bank(12),(13,10) bank(13) 
threadIdx 11 = transpose to (11,11) bank(11),(12,11) bank(12),(13,11) bank(13),(14,11) bank(14) 
threadIdx 12 = transpose to (12,12) bank(12),(13,12) bank(13),(14,12) bank(14),(15,12) bank(15) 
threadIdx 13 = transpose to (13,13) bank(13),(14,13) bank(14),(15,13) bank(15),(16,13) bank(16) 
threadIdx 14 = transpose to (14,14) bank(14),(15,14) bank(15),(16,14) bank(16),(17,14) bank(17) 
threadIdx 15 = transpose to (15,15) bank(15),(16,15) bank(16),(17,15) bank(17),(18,15) bank(18) 
threadIdx 16 = transpose to (16,16) bank(16),(17,16) bank(17),(18,16) bank(18),(19,16) bank(19) 
threadIdx 17 = transpose to (17,17) bank(17),(18,17) bank(18),(19,17) bank(19),(20,17) bank(20) 
threadIdx 18 = transpose to (18,18) bank(18),(19,18) bank(19),(20,18) bank(20),(21,18) bank(21) 
threadIdx 19 = transpose to (19,19) bank(19),(20,19) bank(20),(21,19) bank(21),(22,19) bank(22) 
threadIdx 20 = transpose to (20,20) bank(20),(21,20) bank(21),(22,20) bank(22),(23,20) bank(23) 
threadIdx 21 = transpose to (21,21) bank(21),(22,21) bank(22),(23,21) bank(23),(24,21) bank(24) 
threadIdx 22 = transpose to (22,22) bank(22),(23,22) bank(23),(24,22) bank(24),(25,22) bank(25) 
threadIdx 23 = transpose to (23,23) bank(23),(24,23) bank(24),(25,23) bank(25),(26,23) bank(26) 
threadIdx 24 = transpose to (24,24) bank(24),(25,24) bank(25),(26,24) bank(26),(27,24) bank(27) 
threadIdx 25 = transpose to (25,25) bank(25),(26,25) bank(26),(27,25) bank(27),(28,25) bank(28) 
threadIdx 26 = transpose to (26,26) bank(26),(27,26) bank(27),(28,26) bank(28),(29,26) bank(29) 
threadIdx 27 = transpose to (27,27) bank(27),(28,27) bank(28),(29,27) bank(29),(30,27) bank(30) 
threadIdx 28 = transpose to (28,28) bank(28),(29,28) bank(29),(30,28) bank(30),(31,28) bank(31) 
threadIdx 29 = transpose to (29,29) bank(29),(30,29) bank(30),(31,29) bank(31),(0,29) bank(0) 
threadIdx 30 = transpose to (30,30) bank(30),(31,30) bank(31),(0,30) bank(0),(1,30) bank(1) 
threadIdx 31 = transpose to (31,31) bank(31),(0,31) bank(0),(1,31) bank(1),(2,31) bank(2)

warp间流水线并行

threadIdx 32 = transpose to (4,0) bank(4),(5,0) bank(5),(6,0) bank(6),(7,0) bank(7) 
threadIdx 33 = transpose to (5,1) bank(5),(6,1) bank(6),(7,1) bank(7),(8,1) bank(8) 
threadIdx 34 = transpose to (6,2) bank(6),(7,2) bank(7),(8,2) bank(8),(9,2) bank(9) 
threadIdx 35 = transpose to (7,3) bank(7),(8,3) bank(8),(9,3) bank(9),(10,3) bank(10) 
threadIdx 36 = transpose to (8,4) bank(8),(9,4) bank(9),(10,4) bank(10),(11,4) bank(11) 
threadIdx 37 = transpose to (9,5) bank(9),(10,5) bank(10),(11,5) bank(11),(12,5) bank(12) 
threadIdx 38 = transpose to (10,6) bank(10),(11,6) bank(11),(12,6) bank(12),(13,6) bank(13) 
threadIdx 39 = transpose to (11,7) bank(11),(12,7) bank(12),(13,7) bank(13),(14,7) bank(14) 
threadIdx 40 = transpose to (12,8) bank(12),(13,8) bank(13),(14,8) bank(14),(15,8) bank(15)



*/