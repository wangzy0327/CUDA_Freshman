nvcc -arch=sm_70 -ptx kernel.cu -o kernel.ptx
nvcc -arch=sm_70 -lcuda -lcudart -o demo_launch.out demo_launch.cu