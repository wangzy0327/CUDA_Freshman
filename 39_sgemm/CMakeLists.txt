add_executable(sgemm sgemm.cu)
# 查找 CUDA Toolkit（包括 cuBLAS）
find_package(CUDAToolkit REQUIRED)

# 链接 CUDA 运行时和 cuBLAS
target_link_libraries(sgemm PRIVATE CUDA::cudart CUDA::cublas)