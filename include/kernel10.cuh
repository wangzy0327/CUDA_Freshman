#include <stdio.h>
#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
#define ptr_A(i,j) ptr_A[(i) + (j)*lda]
#define ptr_B(i,j) ptr_B[(i) + (j)*ldb]
#define sa10(i,j) sa10[((j)<<7) + (i)]
#define sb10(i,j) sb10[((j)<<7) + (i)]
#define MS_10 128
#define NS_10 128
#define KS_10 8


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

//M=N=K=2048 MS=NS=128 KS=8 2048 float(8k) SM share mem(96k) block share mem(48k) Block dim  = 256 per SM max 12 block
//grid.dim = 2048*2048/64/64 = 1024 block / 80 SM  per SM = 12.8 block ,  12 block can active improve SM warp active numbers
//128x8x8
//M=N=K=2048: GPU mySgemm Average elasped time: 0.002054 second, performance: 8365.732796 GFLOPS.
//M=N=K=6144:GPU mySgemm Average elasped time: 0.044191 second, performance: 10496.541926 GFLOPS.

//默认每次执行内存操作(mem)后进行计算(compute)有延迟(stall),默认SM 的warp scheduler为了降低stall 空闲时间会从当前正在执行warp(处在内存延迟)切换到其他warp执行
//v9 就存在上述问题，v10为了解决执行stall问题，采用prefetch(预取)然后执行下一个计算
//这里的ptr_A,ptr_B表示当前thread 处理 全局内存的局部变量指针，不去统一修改A(全局变量指针)，减少线程块间同步和读写不一致问题
/*内存访问的流水线化
GPU 的全局内存访问（如 vload）会被编译器转换为 显式的内存加载指令（如 LDG）。这些指令是非阻塞的：线程发起加载请求后，无需等待数据返回即可继续执行后续指令（直到实际使用数据时才会阻塞）。
在 v10 中，预取操作（如 vload(pref_Av, &ptr_A(...))）会触发内存加载，但线程可以继续执行后续计算指令（如之前的 Cres 累加），直到下一次需要用到预取的数据时才会同步。
*/
//M=N=K=6144:  GPU mySgemm Average elasped time: 0.038512 second, performance: 12044.545381 GFLOPS.
__global__  __launch_bounds__(256)
void mysgemm_v10_ano(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    //获取每个block处理的数据块
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    A = &A(bx<<7,0);
    B = &B(0,by<<7);
    C = &C(bx<<7,by<<7);
    //128x128/256 per thread handle 64 data split 64/float4 = 16
    //128x8 share mem / 256 = 4
    __shared__ float sa10[KS_10*MS_10];
    __shared__ float sb10[KS_10*NS_10];
    int row_a = (tx%32)<<2, col_a = tx/32; //row_a [0,4,8,...,120,124] col_a [0,1,2,...,7]
    int col_b = tx%NS_10, row_b = (tx>>7)<<2; //col_b [0-127] row_b [0,4]
    int lane_id = tx&31; //[0-31]
    int warp_id = tx>>5; // [0-7]
    //warp 处理 64x32(colxrow) 每个block内warp的维度是(2x4)
    //每个block 内 warp索引
    int warp_row = warp_id&3;    //[0-3]  总共8个warp(256/32) row 4, col 2
    int warp_col = warp_id>>2;   //[0-1]
    //每个warp 内 thread索引
    //thread 处理 8x8 (colxrow) 每个warp内thread的维度是((64/8)x(32/8))=(8x4)
    int row_w =  lane_id&3; //[0-3]
    int col_w = lane_id>>2;//[0-7]
    //每个thread内 (先确定warp的索引(x warp索引数据的偏移量)，再根据warp内的索引算出整体数据位置索引)
    int row_c = (warp_row<<5) + (row_w<<3);
    int col_c = (warp_col<<6) + (col_w<<3);
    const float* ptr_A, *ptr_B;
    //double buffer
    float4 Av1[2],Bv1[2],Av2[2],Bv2[2],Cv[16],Cres[16];
    float4 pref_Av, pref_Bv;
    int K_upper = K>>3;
    int lda8 = (lda<<3);
    memset(Cres, 0, sizeof(Cres));
    //预取
    pref_Av = *((float4*)&A(row_a,col_a));
    pref_Bv = *((float4*)&B(row_b,col_b));
    ((float4*)sa10)[tx] = pref_Av;
    sb10(col_b,row_b) = pref_Bv.x;
    sb10(col_b,row_b+1) = pref_Bv.y;
    sb10(col_b,row_b+2) = pref_Bv.z;
    sb10(col_b,row_b+3) = pref_Bv.w;
    //同步为了下一步读取
    __syncthreads();
    vload(Av1[0],&sa10(row_c,0));
    vload(Av2[0],&sa10(row_c+4,0));
    vload(Bv1[0],&sb10(col_c,0));
    vload(Bv2[0],&sb10(col_c+4,0));    
    for(int k_count = 0;k_count < K_upper;k_count++){
        /*packing A and B into shared memory*/
        int inc = (k_count+1)%K_upper;
        // ptr_A = const_cast<float*>(A + inc * (lda<<3));
        ptr_A = (A + inc * lda8);
        ptr_B = (B + inc * 8);
        //下面vload与后面的compute可以重叠
        vload(pref_Av, &ptr_A(row_a,col_a));
        vload(pref_Bv, &ptr_B(row_b,col_b));
        // A +=(lda<<3); B+=8;
        // __syncthreads();
        #pragma unroll
        for(int inner_k = 0;inner_k < KS_10;inner_k++){
            int next_inner_k = (inner_k + 1)%KS_10;
            //加载 double buffer 的另一个
            //计算inner_k 加载inner_k+1
            //编译器/手写循环的依赖
            //如果 vload 和 vscal 在同一个 inner_k 循环里并且顺序写死，那么编译器会保持顺序执行，这样不会提前进入下一轮
            if(next_inner_k != 0){
                vload(Av1[(inner_k+1)&1],&sa10(row_c,next_inner_k));
                vload(Av2[(inner_k+1)&1],&sa10(row_c+4,next_inner_k));
                vload(Bv1[(inner_k+1)&1],&sb10(col_c,next_inner_k));
                vload(Bv2[(inner_k+1)&1],&sb10(col_c+4,next_inner_k));
            }

            //fma
            vscal(Cres[0],Av1[(inner_k)&1],Bv1[(inner_k)&1].x);
            vscal(Cres[1],Av2[(inner_k)&1],Bv1[(inner_k)&1].x);
            vscal(Cres[2],Av1[(inner_k)&1],Bv1[(inner_k)&1].y);
            vscal(Cres[3],Av2[(inner_k)&1],Bv1[(inner_k)&1].y);            
            vscal(Cres[4],Av1[(inner_k)&1],Bv1[(inner_k)&1].z);
            vscal(Cres[5],Av2[(inner_k)&1],Bv1[(inner_k)&1].z); 
            vscal(Cres[6],Av1[(inner_k)&1],Bv1[(inner_k)&1].w);
            vscal(Cres[7],Av2[(inner_k)&1],Bv1[(inner_k)&1].w);  

            vscal(Cres[8],Av1[(inner_k)&1],Bv2[(inner_k)&1].x);
            vscal(Cres[9],Av2[(inner_k)&1],Bv2[(inner_k)&1].x);
            vscal(Cres[10],Av1[(inner_k)&1],Bv2[(inner_k)&1].y);
            vscal(Cres[11],Av2[(inner_k)&1],Bv2[(inner_k)&1].y);            
            vscal(Cres[12],Av1[(inner_k)&1],Bv2[(inner_k)&1].z);
            vscal(Cres[13],Av2[(inner_k)&1],Bv2[(inner_k)&1].z); 
            vscal(Cres[14],Av1[(inner_k)&1],Bv2[(inner_k)&1].w);
            vscal(Cres[15],Av2[(inner_k)&1],Bv2[(inner_k)&1].w);                         
        }
        __syncthreads();
        ((float4 *)sa10)[tx] = pref_Av;
        sb10(col_b,row_b)=pref_Bv.x;
        sb10(col_b,row_b+1)=pref_Bv.y;
        sb10(col_b,row_b+2)=pref_Bv.z;
        sb10(col_b,row_b+3)=pref_Bv.w;
        __syncthreads();
        vload(Av1[0], &sa10(row_c,0))
        vload(Av2[0], &sa10(row_c+4,0))
        vload(Bv1[0], &sb10(col_c,0))
        vload(Bv2[0], &sb10(col_c+4,0))        
    }
    #pragma unroll
    for(int i = 0;i < 8;i++){
        vload(Cv[i*2],&C(row_c,col_c+i));
        vload(Cv[i*2+1],&C(row_c+4,col_c+i));
    }
    #pragma unroll
    for(int i = 0;i < 16;i++){
        simd_axpby(Cres[i],alpha,Cres[i],beta,Cv[i]);
    }
    #pragma unroll
    for(int i = 0;i < 8;i++){
        vstore(&C(row_c,col_c+i), Cres[i*2]);
        vstore(&C(row_c+4,col_c+i), Cres[i*2+1]);
    }
}

