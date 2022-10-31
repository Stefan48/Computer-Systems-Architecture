#include <stdio.h>
#include <math.h>
#include "utils/utils.h"

// TODO 6: Write the code to add the two arrays element by element and 
// store the result in another array
__global__ void add_arrays(const float *a, const float *b, float *c, int N)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
	c[i] = a[i] + b[i];
    }
}

int main(void) {
    cudaSetDevice(0);
    int N = 1 << 20;

    float *host_array_a = 0;
    float *host_array_b = 0;
    float *host_array_c = 0;

    float *device_array_a = 0;
    float *device_array_b = 0;
    float *device_array_c = 0;

    int bytes = N * sizeof(float);

    // TODO 1: Allocate the host's arrays
    host_array_a = (float*)malloc(bytes);
    host_array_b = (float*)malloc(bytes);
    host_array_c = (float*)malloc(bytes);

    // TODO 2: Allocate the device's arrays
    cudaMalloc((void**)&device_array_a, bytes);
    cudaMalloc((void**)&device_array_b, bytes);
    cudaMalloc((void**)&device_array_c, bytes);

    // TODO 3: Check for allocation errors
    if (host_array_a == NULL || host_array_b == NULL || host_array_c == NULL
       || device_array_a == NULL || device_array_b == NULL || device_array_c == NULL)
    {
		printf("memory allocation error!\n");
		exit(1);
    }

    // TODO 4: Fill array with values; use fill_array_float to fill
    // host_array_a and fill_array_random to fill host_array_b. Each
    // function has the signature (float *a, int n), where n = number of elements.

    fill_array_float(host_array_a, N);
    fill_array_random(host_array_b, N);

    // TODO 5: Copy the host's arrays to device

    cudaMemcpy(device_array_a, host_array_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_array_b, host_array_b, bytes, cudaMemcpyHostToDevice);

    // TODO 6: Execute the kernel, calculating first the grid size
    // and the amount of threads in each block from the grid
    // Hint: For this exercise the block_size can have any value lower than the
    //      API's maximum value (it's recommended to be close to the maximum
    //      value).

    //int blocks_no = 1 << 5, block_size = 1 << 15;
    //add_arrays<<<blocks_no, block_size>>>(device_array_a, device_array_b, device_array_c, N);

    int blocks_no = 1 << 10, block_size = N / blocks_no;
    // note: thread block size should always be a multiple of 32,
    // because kernels issue instructions in warps (32 threads)

    add_arrays<<<blocks_no, block_size>>>(device_array_a, device_array_b, device_array_c, N);
    cudaDeviceSynchronize();


    // TODO 7: Copy back the results and then uncomment the checking function

    cudaMemcpy(host_array_c, device_array_c, bytes, cudaMemcpyDeviceToHost);
    
    //for (int i = 0; i < N; ++i)
    //	printf("host_array_c[%d]=%f\n", i, host_array_c[i]);

    check_task_2(host_array_a, host_array_b, host_array_c, N);

    // TODO 8: Free the memory
    free(host_array_a);
    free(host_array_b);
    free(host_array_c);
    cudaFree(device_array_a);
    cudaFree(device_array_b);
    cudaFree(device_array_c);
   
    return 0;
}
