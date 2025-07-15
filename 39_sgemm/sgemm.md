# How to optimize SGEMM on NVIDIA GPUs

"HPC is all about reducing data movement". Optimizing GEMM on GPU and CPU platforms share the same idea: to hide the memory latency with massive parallelism, cache-/register-level data re-use, and manual prefetching. On CPUs, both instruction-level and data-level parallelisms are exploited as well as delicate prefetching schemes are designed to hide the memory latency. Meanwhile, we partition the input matrices and pack them before computing to ensure a "smooth and low-latancy" computing kernel. The prefetching for matrix ```C``` is especially critical to the CPU GEMM performance.

On GPUs, we also need to take advantage of the low-latency "cache" --- shared memory. There are rich opportunities on GPUs for us to exploit data re-use on both shared memory level and register level. More details could be found in the official document of [CUTLASS](https://github.com/NVIDIA/cutlass/blob/master/media/docs/efficient_gemm.md).

All questions are encouraged to send to [wangzy0327@qq.com](mailto:wangzy0327@qq.com).

# Hardware platforms and software configurations

* We compiled the program with ```gcc 7.5.0``` under Ubuntu 18.04.5 LTS.
* NVIDIA cuBLAS version: ```CUDA cuBLAS 11.3.1.68```.

# How to run

Just three steps.

* We first run ```cmake -B cmake-build && cmake --build cmake-build -j```.
* Second, run the binary using ```./cmake-build/39_sgemm/sgemm [kernel_number]```, where ```kernel_number``` selects the kernel for benchmark. ```0``` represents NVIDIA cuBLAS and ```1-11``` represent 11 kernels demonstrating the optimizing strategies.

# Step-wise Optimizations

Here we take the column-major implemetation for SGEMM. Both A and B are not transposed.

## Kernel 1 (naive version)

[source code](../include/kernel1.cuh)

Kernel 1 is the most naive implementation of SGEMM in CUDA. This is the triple-for-loop implementation with register re-use when updating ```C(i,j)```. In this version, each threa block (TB) is responsible for a ```32x32``` sub-block of ```C```, and each thread computes only a single element of the ```C``` matrix.

## Kernel 2 (Kernel1 + 32x32x32 tiling)

[source code](../include/kernel2.cuh)

Kernel2 partitions the matrix ```A``` and matrix ```B``` into ```32x32``` blocks. These ```32x32``` blocks are loaded into shared memory before being loaded for GEMM computation. When loading the data into shared memory (this is called as packing in CPU GEMM), each thread is responsible to load/store one element and we set 1024 threads per TB using ```__launch_bounds__(1024)```. After packing is completed, all the threads are synchronized and then start to compute for their own element. Since each TB is still to compute a ```32x32``` matrix ```C```, each thread remains to take a single element of ```C```.
In short, this version adds cache blocking upon [the previous version](../include/kernel1.cuh), with the parameter set ```{Ms,Ns,Ks}={32,32,32}```.

## Kernel 3 (minor update on Kernel2)

[source code](../include/kernel3.cuh)

We bring a simple optimization upon [kernel 2](../include/kernel2.cuh) here: storing ```threadIdx.x``` before re-using it massively, in order to reduce living registers and benefit the compiler optimization. The performance slightly improves in this step.

## Kernel 4 (kernel 3 + reducing bank conflictions on shared memory)

[source code](../include/kernel4.cuh)

In the previous version, the memory access on the shared memory is not ideal. We re-ordered the memory access pattern on the shared memory: making the shared memory col-major but transposing matrix ```B``` when packing it into the shared memory. This doubles the performance. kernel4 uses loop unrolling based on kernel3, and the performance is improved to a certain extent.

## Kernel 5 (kernel4 + 4x1 micro kernel)

[source code](../include/kernel5.cuh)

In this step, we ask each thread to compute 4 elements for the ```C``` matrix. Therefore, we now have 256 threads in a TB to compute the ```32x32``` matrix ```C``` that the TB is responsible for. Using the CPU-GEMM language, the micro kernel's shape is: ```4x1```: that is to say, after the packing routine completes, each thread loads a ```4x1``` A and an ```1x1``` B and computes ```C(4x1)``` += ```A(4x1)*B(1x1)```.
Starting from this step, we restrict 256 threads for each TB.
mysgemm_v5_ano2_pro uses the method of reading global memory diagonally and writing shared memory diagonally (although the B matrix is not merged when reading (col is different), writing to shared memory completely avoids bank conflict), so there is no write conflict, and the performance will be greatly improved. In addition, since there is no write conflict and no padding is used, the subsequent 4x1 micro kernel calculates the sum of products (if padding is used, 4 is not in the same column, and there will be a serious conflict with the other 4 columns in the same warp), but this problem does not exist here.

For detailed performance comparison, please see [here](./sgemm-metrics.txt).

## Kernel 6 (kernel5 + vectorized load/store)

[source code](../include/kernel6.cuh)

Since our target machine supports a 128-bit transaction from the DRAM, we can apply the vectorized load operation using the ```float4``` data type. kernel6 performs vector load/store based on kernel5 native version

## Kernel 7 ( Ms,Ns,Ks= 64,64,64 tilling)

[source code](../include/kernel7.cuh)

Kernel7 partitions the matrix ```A``` and matrix ```B``` into ```64x64``` blocks. These ```64x64``` blocks are loaded into shared memory before being loaded for GEMM computation. When loading the data into shared memory (this is called as packing in CPU GEMM), each thread is responsible to load/store one element and we set 1024 threads per TB using ```__launch_bounds__(256)```. After packing is completed, all the threads are synchronized and then start to compute for their own element. Since each TB is still to compute a ```64x64``` matrix ```C```, each thread remains to take a single element of ```C```.
In short, this version adds cache blocking upon [the previous version](../include/kernel6.cuh), with the parameter set ```{Ms,Ns,Ks}={64,64,64}```.
mysgemm_v7 achieves a leapfrog performance improvement through the combination strategy of "increasing block size to improve computing memory access ratio + inheriting shared memory conflict optimization （memory diagonally load/store）+ reducing scheduling overhead"
The mysgemm_v7_plus version uses float2 vectorization for shared memory matrix muladd operations, which improves performance compared to mysgemm_v7
