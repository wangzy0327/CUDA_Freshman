#include <stdio.h>
#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
#define sa7(i,j) sa7[((j)<<6) + (i)]
#define sb7(i,j) sb7[((j)<<6) + (i)]
#define MS_7 64
#define NS_7 64
#define KS_7 64
#define KS_7_2 16
//M=N=K=2048
//M=N=K=2048 MS=NS=KS=64 8192 float(16k) SM share mem(96k) block share mem(48k) Block dim  = 256 per SM max 6block
//grid.dim = 2048*2048/64/64 = 1024 block / 80 SM  per SM = 12.8 block , only 6 block can active because of block share memory alloc limit
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
            #pragma unroll
            for(int i = 0;i < 4;i++){
                #pragma unroll
                for(int j = 0;j < 4;j++){
                    tmp[i][j] += sa7(A_idx[i],inner_k) * sb7(B_idx[j],inner_k);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i = 0;i < 4;i++){
        #pragma unroll
        for(int j = 0;j < 4;j++){
            C(A_idx[i],B_idx[j]) = alpha * tmp[i][j] + beta * C(A_idx[i], B_idx[j]);
        }
    }
}



//M=N=K=2048 MS=NS=KS=64 8192 float(16k) SM share mem(96k) block share mem(48k) Block dim  = 256 per SM max 6block
//grid.dim = 2048*2048/64/64 = 1024 block / 80 SM  per SM = 12.8 block , only 6 block can active because of block share memory alloc limit
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
            #pragma unroll
            for(int i = 0;i < 4;i++){
                #pragma unroll
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
//M=N=K=2048 MS=NS=64 KS=16 2048 float(8k) SM share mem(96k) block share mem(48k) Block dim  = 256 per SM max 12 block
//grid.dim = 2048*2048/64/64 = 1024 block / 80 SM  per SM = 12.8 block ,  12 block can active    
// cache blocking version, without register-level data re-use
// with memory coelascing on shared memory
// more workloads per thread. 4x4 micro kernel.
// adopt vetorized load/store
__global__  __launch_bounds__(256)
void mysgemm_v7_ano(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    // sa7(i,j) sa7[((j)<<6) + (i)] j [0-15] i [0-64]
    // sb7(i,j) sb7[((j)<<6) + (i)] j [0-15] i [0-64]
    int row_a = tx%32, col_a = ((tx/32)&7)<<1; //[0,1,...,31] col_a [0,2,4,...14,]
    // int row_b = tx%32, col_b = ((tx>>5)&7)<<1;  //16 x 64 row_b [0,1,2,...,31]  col_b [0,2,...14]
    int row_b[2],col_b[2];
    row_b[0] = tx%32;
    row_b[1] = row_b[0]+32;  // row_b[0] [0,1,...,31] row_b[1] [32,33,...,63]
    col_b[0] = ((tx>>5)&7)<<1; col_b[1] = col_b[0]+1; // col_b[0] [0,2,...,14] col_b[1] [1,3,5,...,15]
    int row_s = (tx&0x0F)<<2; // [0,4,8,...,60] 16
    int col_s = ((tx>>4)&0x0F)<<2; // [0,4,8,...,60] 16
    int lda16 = lda<<4;
    A = &A((bx<<6),0);
    B = &B(0,(by<<6));
    C = &C((bx<<6),(by<<6));//the TB size is 64.
    __shared__ float sa7[KS_7_2*MS_7];
    __shared__ float sb7[KS_7_2*NS_7];
    float4 Av, Bv, Cv[4], Cres[4];
    memset(Cres, 0, sizeof(Cres));
    for (int k_count = 0; k_count<K; k_count+=KS_7_2){
        // if(k_count == 0 && bx == 0 && by == 0){
        //     // printf("Matrix A (%d) = (%d,%d) handle (%d,%d),(%d,%d),(%d,%d),(%d,%d)  \n",tx,row_a,col_a,row_a,col_a,row_a+32,col_a,row_a,col_a+1,row_a+32,col_a+1);
        //     // printf("Matrix B %d (%d,%d) =  handle (%d,%d),(%d,%d),(%d,%d),(%d,%d) transpose to (%d,%d),(%d,%d),(%d,%d),(%d,%d) \n",(tx),(row_b,col_b),
        //     printf("k_count : %d ,block :(%d,%d) kernel threadIdx : %d (%d,%d) =  (%d,%d),(%d,%d),(%d,%d),(%d,%d) transpose to (%d,%d) bank(%d),(%d,%d) bank(%d),(%d,%d) bank(%d),(%d,%d) bank(%d) \n",k_count, bx,by, tx,\
        //     row_b[0],col_b[0],\
        //     (col_b[0]+row_b[0]%16)%KS_7_2,row_b[0],\
        //     (col_b[0]+row_b[1]%16)%KS_7_2,row_b[1],\
        //     (col_b[1]+row_b[0]%16)%KS_7_2,row_b[0],\
        //     (col_b[1]+row_b[1]%16)%KS_7_2,row_b[1],\
        //     row_b[0],(col_b[0]+row_b[0]%16)%KS_7_2,row_b[0]%32,\
        //     row_b[1],(col_b[0]+row_b[1]%16)%KS_7_2,row_b[1]%32,\
        //     row_b[0],(col_b[1]+row_b[0]%16)%KS_7_2,row_b[0]%32,\
        //     row_b[1],(col_b[1]+row_b[1]%16)%KS_7_2,row_b[1]%32);
        // }
        sa7(row_a,col_a) = A(row_a,col_a);
        sa7(row_a+32,col_a) = A(row_a+32,col_a);
        sa7(row_a,col_a+1) = A(row_a,col_a+1);
        sa7(row_a+32,col_a+1) = A(row_a+32,col_a+1);

        sb7(row_b[0],(col_b[0]+row_b[0]/16)%KS_7_2) = B((col_b[0]+row_b[0]/16)%KS_7_2,row_b[0]);
        sb7(row_b[1],(col_b[0]+row_b[1]/16)%KS_7_2) = B((col_b[0]+row_b[1]/16)%KS_7_2,row_b[1]);
        sb7(row_b[0],(col_b[1]+row_b[0]/16)%KS_7_2) = B((col_b[1]+row_b[0]/16)%KS_7_2,row_b[0]);
        sb7(row_b[1],(col_b[1]+row_b[1]/16)%KS_7_2) = B((col_b[1]+row_b[1]/16)%KS_7_2,row_b[1]);             
        A+=lda16;B+=16;
        __syncthreads();
        #pragma unroll
        for (int inner_k_count=0;inner_k_count<KS_7_2;inner_k_count++){
            //四路bank
            vload(Av, &sa7(row_s,inner_k_count))
            vload(Bv, &sb7(col_s,inner_k_count))
            vscal(Cres[0], Av, Bv.x)
            vscal(Cres[1], Av, Bv.y)
            vscal(Cres[2], Av, Bv.z)
            vscal(Cres[3], Av, Bv.w)
        }
        __syncthreads();
    }
    vload(Cv[0], &C(row_s,col_s))
    vload(Cv[1], &C(row_s,col_s+1))
    vload(Cv[2], &C(row_s,col_s+2))
    vload(Cv[3], &C(row_s,col_s+3))
    simd_axpby(Cres[0],alpha,Cres[0],beta,Cv[0])
    simd_axpby(Cres[1],alpha,Cres[1],beta,Cv[1])
    simd_axpby(Cres[2],alpha,Cres[2],beta,Cv[2])
    simd_axpby(Cres[3],alpha,Cres[3],beta,Cv[3])

    vstore(&C(row_s,col_s), Cres[0])
    vstore(&C(row_s,col_s+1), Cres[1])
    vstore(&C(row_s,col_s+2), Cres[2])
    vstore(&C(row_s,col_s+3), Cres[3])
}

// M=N=K=2048: GPU mySgemm Average elasped time: 0.001972 second, performance: 8713.456222 GFLOPS.
__global__  __launch_bounds__(256)
void mysgemm_v7_ano_plus(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    int row_a = tx%64, col_a = (tx/64)<<2; //[0,1,...,63] col_a [0,4,8,12]
    int col_b = tx%64, row_b = (tx/64)<<2; // [0,4,8,12]
    int row_s = (tx&0x0F)<<2; // [0,4,8,...,60] 16
    int col_s = ((tx>>4)&0x0F)<<2; // [0,4,8,...,60] 16
    int lda16 = lda<<4;
    A = &A((bx<<6),0);
    B = &B(0,(by<<6));
    C = &C((bx<<6),(by<<6));//the TB size is 64.
    __shared__ float sa7[KS_7_2*MS_7];
    __shared__ float sb7[KS_7_2*NS_7];
    float4 Av, Bv, Cv[4], Cres[4];
    memset(Cres, 0, sizeof(Cres));
    for (int k_count = 0; k_count<K; k_count+=KS_7_2){
      
        sa7(row_a,col_a) = A(row_a,col_a);
        sa7(row_a,col_a+1) = A(row_a,col_a+1);
        sa7(row_a,col_a+2) = A(row_a,col_a+2);
        sa7(row_a,col_a+3) = A(row_a,col_a+3);

        Bv = *((float4*)(&B(row_b,col_b)));
        sb7(col_b,row_b) = Bv.x;
        sb7(col_b,row_b+1) = Bv.y;
        sb7(col_b,row_b+2) = Bv.z;
        sb7(col_b,row_b+3) = Bv.w;
        // sb7(col_b,row_b) = B(row_b,col_b);
        // sb7(col_b,row_b+1) = B(row_b+1,col_b);
        // sb7(col_b,row_b+2) = B(row_b+2,col_b);
        // sb7(col_b,row_b+3) = B(row_b+3,col_b);

        A+=lda16;B+=16;
        __syncthreads();
        #pragma unroll
        for (int inner_k_count=0;inner_k_count<KS_7_2;inner_k_count++){
            //四路bank
            vload(Av, &sa7(row_s,inner_k_count))
            vload(Bv, &sb7(col_s,inner_k_count))
            vscal(Cres[0], Av, Bv.x)
            vscal(Cres[1], Av, Bv.y)
            vscal(Cres[2], Av, Bv.z)
            vscal(Cres[3], Av, Bv.w)
        }
        __syncthreads();
    }
    vload(Cv[0], &C(row_s,col_s))
    vload(Cv[1], &C(row_s,col_s+1))
    vload(Cv[2], &C(row_s,col_s+2))
    vload(Cv[3], &C(row_s,col_s+3))
    simd_axpby(Cres[0],alpha,Cres[0],beta,Cv[0])
    simd_axpby(Cres[1],alpha,Cres[1],beta,Cv[1])
    simd_axpby(Cres[2],alpha,Cres[2],beta,Cv[2])
    simd_axpby(Cres[3],alpha,Cres[3],beta,Cv[3])

    vstore(&C(row_s,col_s), Cres[0])
    vstore(&C(row_s,col_s+1), Cres[1])
    vstore(&C(row_s,col_s+2), Cres[2])
    vstore(&C(row_s,col_s+3), Cres[3])
}


//M=N=K=2048 MS=NS=64 KS=16 2048 float(8k) SM share mem(96k) block share mem(48k) Block dim  = 256 per SM max 12 block
//grid.dim = 2048*2048/64/64 = 1024 block / 80 SM  per SM = 12.8 block ,  12 block can active improve SM warp active numbers
//64x64x16
//M=N=K=2048:
// GPU cublas Average elasped time: 0.001439 second, performance: 11937.892956 GFLOPS.
// GPU mySgemm Average elasped time: 0.001863 second, performance: 9220.625443 GFLOPS.

__global__  __launch_bounds__(256)
void mysgemm_v7_ano2(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    A = &A((bx<<6),0);
    B = &B(0,(by<<6));
    C = &C((bx<<6),(by<<6));//the TB size is 64.    
    //64x16=1024 16x64=1024
    __shared__ float sa7[KS_7_2*MS_7],sb7[KS_7_2*NS_7];
    int row_a = (tx&15)<<2, col_a = tx>>4; // row_a = [0,4,...60] col_a = [0,1,2... 15]
    int row_b = (tx&3)<<2, col_b = tx>>2;  // row_b = [0,4,8,12] col_b = [0,1,2,...63]
    int col_c = col_a<<2;
    //64x16/256 = 4 per thread 4 float data
    float4 Av,Bv,Cv[4],Cres[4];
    memset(Cres,0,sizeof(Cres));
    for(int k_count = 0;k_count < K;k_count+=KS_7_2){
        vload(Av, &A(row_a,col_a))
        vload(Bv, &B(row_b,col_b))
        ((float4 *)sa7)[tx] = Av;
        sb7(col_b,row_b)=Bv.x;
        sb7(col_b,row_b+1)=Bv.y;
        sb7(col_b,row_b+2)=Bv.z;
        sb7(col_b,row_b+3)=Bv.w;
        A+=(lda<<4);B+=16;
        __syncthreads();
        #pragma unroll
        for (int inner_k_count=0;inner_k_count<KS_7_2;inner_k_count++){
            vload(Av, &sa7(row_a,inner_k_count));
            vload(Bv, &sb7(col_c,inner_k_count));
            vscal(Cres[0], Av, Bv.x);
            vscal(Cres[1], Av, Bv.y);
            vscal(Cres[2], Av, Bv.z);
            vscal(Cres[3], Av, Bv.w);
        }
        __syncthreads();
    }
    vload(Cv[0], &C(row_a,col_c));
    vload(Cv[1], &C(row_a,col_c+1));
    vload(Cv[2], &C(row_a,col_c+2));
    vload(Cv[3], &C(row_a,col_c+3));
    simd_axpby(Cres[0],alpha,Cres[0],beta,Cv[0]);
    simd_axpby(Cres[1],alpha,Cres[1],beta,Cv[1]);
    simd_axpby(Cres[2],alpha,Cres[2],beta,Cv[2]);
    simd_axpby(Cres[3],alpha,Cres[3],beta,Cv[3]);

    vstore(&C(row_a,col_c), Cres[0])
    vstore(&C(row_a,col_c+1), Cres[1])
    vstore(&C(row_a,col_c+2), Cres[2])
    vstore(&C(row_a,col_c+3), Cres[3])
}