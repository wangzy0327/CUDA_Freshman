#include <cuda_runtime.h>
#include <stdio.h>
#include <cublas_v2.h>
#include "freshman.h"
#include "mysgemms.cuh"
#define TILEX 32
#define TILEY 32

#define CEIL_DIV(m,n) ( (m) + (n) - 1 ) / (n)

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

void testThreadIdx(){
  int tx_size = 256;
  int bx = 0;int by = 0;
  for(int tx = 0;tx < tx_size;tx++){
    int row_b = tx%32, col_b = ((tx>>5)&7)<<1;  //16 x 64 row_b [0,1,2,...,31]  col_b [0,2,...,14]
    printf("kernel threadIdx : %d (%d,%d) =  (%d,%d),(%d,%d),(%d,%d),(%d,%d) transpose to (%d,%d),(%d,%d),(%d,%d),(%d,%d) \n", tx,\
    col_b,row_b,\
    row_b,(row_b%16+col_b)%KS_7_2,\
    row_b,(row_b%16+col_b+1)%KS_7_2,\
    row_b+32,(row_b%16+col_b)%KS_7_2,\
    row_b+32,(row_b%16+col_b+1)%KS_7_2,\
    (row_b%16+col_b)%KS_7_2,row_b,\
    (row_b%16+col_b+1)%KS_7_2,row_b,\
    (row_b%16+col_b)%KS_7_2,row_b+32,\
    (row_b%16+col_b+1)%KS_7_2,row_b+32);
  }
}

void testThreadIdx2(){
  int tx_size = 256;
  for(int tx = 0;tx < tx_size;tx++){
    int row = tx&0x1F;  // 0...31
    int col0,col1,col2,col3;
    col0 = (tx>>5)*4;  // col0 ∈ {0,4,8,12,16,20,24,28}
    col1 = col0 + 1;
    col2 = col0 + 2;
    col3 = col0 + 3;
    printf("threadIdx2 %d = transpose to (%d,%d),(%d,%d),(%d,%d),(%d,%d) \n",tx,(row+col0)%NS,row,(row+col1)%NS,row,(row+col2)%NS,row,(row+col3)%NS,row);
  }
}

void test_mysgemm_v1(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){
    // cudaDeviceSynchronize();
    int blockX = 32, blockY = 32;
    dim3 blockDim(blockX,blockY);
    dim3 gridDim(CEIL_DIV(M,blockX),CEIL_DIV(N,blockY));
    mysgemm_v1<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    cudaDeviceSynchronize();
    // cudaStreamSynchronize(0);
}

void test_mysgemm_v2(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){
    // cudaDeviceSynchronize();
    int blockX = 32, blockY = 32;
    dim3 blockDim(blockX,blockY);
    dim3 gridDim(CEIL_DIV(M,blockX),CEIL_DIV(N,blockY));
    mysgemm_v2<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    cudaDeviceSynchronize();
}

void test_mysgemm_v3(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){
    // cudaDeviceSynchronize();
    int blockX = 32, blockY = 32;
    dim3 blockDim(blockX,blockY);
    dim3 gridDim(CEIL_DIV(M,blockX),CEIL_DIV(N,blockY));
    mysgemm_v3<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    cudaDeviceSynchronize();
}

void test_mysgemm_v4(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){
    // cudaDeviceSynchronize();
    int blockX = 32, blockY = 32;
    dim3 blockDim(blockX,blockY);
    dim3 gridDim(CEIL_DIV(M,blockX),CEIL_DIV(N,blockY));
    mysgemm_v4<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    cudaDeviceSynchronize();
}

void test_mysgemm_v5(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){
    // cudaDeviceSynchronize();
    int blockX = 32, blockY = 32;
    // dim3 blockDim(1024);
    dim3 blockDim(256);//x4
    // dim3 blockDim(64);//x4
    dim3 gridDim(CEIL_DIV(M,blockX),CEIL_DIV(N,blockY));
    mysgemm_v5_ano2_pro<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    cudaDeviceSynchronize();
}

void test_mysgemm_v6(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){
    // cudaDeviceSynchronize();
    int blockX = 32, blockY = 32;
    // dim3 blockDim(1024);
    dim3 blockDim(256);//x4
    // dim3 blockDim(64);//x4
    dim3 gridDim(CEIL_DIV(M,blockX),CEIL_DIV(N,blockY));
    mysgemm_v6<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    cudaDeviceSynchronize();
}

void test_mysgemm_v7(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float* C){
    cudaDeviceSynchronize();
    int blockX = 64, blockY = 64;
    // dim3 blockDim(1024);
    dim3 blockDim(256);//x4
    // dim3 blockDim(64);//x4
    dim3 gridDim(CEIL_DIV(M,blockX),CEIL_DIV(N,blockY));
    mysgemm_v7_ano_plus<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
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
  // testThreadIdx();
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
  CHECK(cudaMemset(C_dev,0,nBytes));
  CHECK(cudaMemset(C_dev_lib,0,nBytes));

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
  

  // for(int i_count = upper_limit-1;i_count < upper_limit;i_count++){
  // for(int i_count = 0;i_count < upper_limit;i_count++){
  for(int i_count = 0;i_count < 1;i_count++){
    m=n=k=SIZE[i_count];
    printf("\nM=N=K=%d:\n",m);
    //warmup cuBLAS 库在第一次调用时需要初始化内部状态（如加载内核、分配内部缓冲区等），这会带来额外的开销
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A_dev, m, B_dev, k, &beta, C_dev_lib, m);
    cudaEventRecord(beg_lib);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A_dev, m, B_dev, k, &beta, C_dev_lib, m);
    cudaEventRecord(end_lib);
    cudaEventSynchronize(beg_lib);
    cudaEventSynchronize(end_lib);
    cudaEventElapsedTime(&elapsed_time, beg_lib, end_lib);
    elapsed_time /= 1000.;
    printf("GPU cublas Average elasped time: %f second, performance: %f GFLOPS.\n", elapsed_time,2.*1e-9*m*n*k/elapsed_time);
    if(i_count < 0){
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
        case 1: test_mysgemm_v1(m,n,k,alpha,A_dev,B_dev,beta,C_dev);break;
        case 2: test_mysgemm_v2(m,n,k,alpha,A_dev,B_dev,beta,C_dev);break;
        case 3: test_mysgemm_v3(m,n,k,alpha,A_dev,B_dev,beta,C_dev);break;
        case 4: test_mysgemm_v4(m,n,k,alpha,A_dev,B_dev,beta,C_dev);break;
        case 5: test_mysgemm_v5(m,n,k,alpha,A_dev,B_dev,beta,C_dev);break;
        case 6: test_mysgemm_v6(m,n,k,alpha,A_dev,B_dev,beta,C_dev);break;
        case 7: test_mysgemm_v7(m,n,k,alpha,A_dev,B_dev,beta,C_dev);break;
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
  // testThreadIdx();
  // testThreadIdx2();
  return 0;
}