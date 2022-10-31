#include <stdio.h>
#include "utils/utils.h"

// ~TODO 3~
// Modify the kernel below such as each element of the 
// array will be now equal to 0 if it is an even number
// or 1, if it is an odd number
__global__ void kernel_parity_id(int *a, int N)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		a[i] %= 2;
	}
}

// ~TODO 4~
// Modify the kernel below such as each element will
// be equal to the BLOCK ID this computation takes
// place.
__global__ void kernel_block_id(int *a, int N)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		a[i] = blockIdx.x;
	}
}

// ~TODO 5~
// Modify the kernel below such as each element will
// be equal to the THREAD ID this computation takes
// place.
__global__ void kernel_thread_id(int *a, int N)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
	{
		a[i] = threadIdx.x;
	}
}

int main(void) {
    int nDevices;

    // Get the number of CUDA-capable GPU(s)
    cudaGetDeviceCount(&nDevices);
    printf("Device count: %d\n", nDevices);

    // ~TODO 1~
    // For each device, show some details in the format below, 
    // then set as active device the first one (assuming there
    // is at least CUDA-capable device). Pay attention to the
    // type of the fields in the cudaDeviceProp structure.
    //
    // Device number: <i>
    //      Device name: <name>
    //      Total memory: <mem>
    //      Memory Clock Rate (KHz): <mcr>
    //      Memory Bus Width (bits): <mbw>
    // 
    // Hint: look for cudaGetDevicePropers and cudaSetDevice in
    // the Cuda Toolkit Documentation.

    struct cudaDeviceProp prop;
    int ret;
    for (int i = 0; i < nDevices; ++i)
    {
    	ret = cudaGetDeviceProperties(&prop, i);
		if (ret != cudaSuccess)
		{
			exit(1);
		}
		printf("Device name: %s\nTotal memory: %d\nMemory Clock Rate (KHz): %d\nMemory Bus Width (bits): %d\n",
			prop.name, prop.totalGlobalMem, prop.clockRate, prop.memoryBusWidth);
    }

    if (nDevices)
    {
		ret = cudaSetDevice(0);
		if (ret != cudaSuccess)
		{
			exit(1);
		}
    }
    else
    {
		exit(1);
    }
    

    // ~TODO 2~
    // With information from example_2.cu, allocate an array with
    // integers (where a[i] = i). Then, modify the three kernels
    // above and execute them using 4 blocks, each with 4 threads.
    // Hint: num_elements = block_size * block_no (see example_2)
    //
    // You can use the fill_array_int(int *a, int n) function (from utils)
    // to fill your array as many times you want.

    const int N = 16;
    int *a_host = (int*)malloc(N * sizeof(int));
    int *a_device = NULL;
    cudaMalloc((void**)&a_device, N * sizeof(int));
    if (a_host == NULL || a_device == NULL)
    {
		exit(1);
    }
    fill_array_int(a_host, N);
    cudaMemcpy(a_device, a_host, N * sizeof(int), cudaMemcpyHostToDevice);
   
  
    // ~TODO 3~
    // Execute kernel_parity_id kernel and then copy from 
    // the device to the host; call cudaDeviceSynchronize()
    // after a kernel execution for safety purposes.
    //
    // Uncomment the line below to check your results

    int blocks_no = 4, block_size = 4;
    kernel_parity_id<<<blocks_no, block_size>>>(a_device, N);
    cudaDeviceSynchronize();

    cudaMemcpy(a_host, a_device, N * sizeof(int), cudaMemcpyDeviceToHost);

    

    check_task_1(3, a_host);

    // ~TODO 4~
    // Execute kernel_block_id kernel and then copy from 
    // the device to the host;
    //
    // Uncomment the line below to check your results

    kernel_block_id<<<blocks_no, block_size>>>(a_device, N);
    cudaDeviceSynchronize();

    cudaMemcpy(a_host, a_device, N * sizeof(int), cudaMemcpyDeviceToHost);


    check_task_1(4, a_host);

    // ~TODO 5~
    // Execute kernel_thread_id kernel and then copy from 
    // the device to the host;
    //
    // Uncomment the line below to check your results

    kernel_thread_id<<<blocks_no, block_size>>>(a_device, N);
    cudaDeviceSynchronize();

    cudaMemcpy(a_host, a_device, N * sizeof(int), cudaMemcpyDeviceToHost);

    check_task_1(5, a_host);

    // TODO 6: Free the memory
    free(a_host);
    cudaFree(a_device);
    
    return 0;
}
