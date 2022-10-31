#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>
#include "test_map.hpp"
#include "gpu_hashtable.hpp"

using namespace std;
 
/* computes hash of an unsigned integer */
static __device__ unsigned int getHash(unsigned int x)
{
	x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}
 
static __global__ void insert_batch(struct Slot *slots, unsigned int size, unsigned int *inserted, const int *keys, const int *values, int numKeys)
{
  	/* compute the index this thread should process */
  	unsigned int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  	/* avoid accessing out of bounds elements */
  	if (idx >= numKeys || keys[idx] <= 0 || values[idx] <= 0)
  		return;
  	unsigned int key = (unsigned int)keys[idx];
  	unsigned int value = (unsigned int)values[idx];
  	unsigned int h = getHash(key) % size;
  	unsigned int old_key;
  	/* search for an empty slot to insert the key-value pair */
    while (1)
    {
    	old_key = atomicCAS(&slots[h].key, 0, key);
    	if (old_key == 0 || old_key == key)
    	{
    		if (old_key == 0)
    			atomicAdd(inserted, 1);
    		slots[h].value = value;
    		break;
    	}
    	h = (h + 1) % size;
    }
}

static __global__ void redistribute(struct Slot *slots, unsigned int size, const struct Slot *slots_old, unsigned int size_old)
{
	/* compute the index this thread should process */
  	unsigned int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  	/* avoid accessing out of bounds elements */
  	if (idx >= size_old || slots_old[idx].key == 0)
  		return;
  	unsigned int key = slots_old[idx].key;
  	unsigned int value = slots_old[idx].value;
  	unsigned int h = getHash(key) % size;
  	unsigned int old_key;
    /* search for an empty slot to insert the key-value pair */
    while (1)
    {
		old_key = atomicCAS(&slots[h].key, 0, key);
		if (old_key == 0)
		{
			slots[h].value = value;
			break;
		}
		h = (h + 1) % size;
    }
}

static __global__ void get_batch(int *values, const int *keys, int numKeys, const struct Slot *slots, unsigned int size)
{
	/* compute the index this thread should process */
  	unsigned int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  	/* avoid accessing out of bounds elements */
  	if (idx >= numKeys)
  		return;
  	unsigned int key = keys[idx];
  	unsigned int value = 0;
  	unsigned int h = getHash(key) % size;
    unsigned int iterations = 0;
    /* search for the key */
    while (iterations < size)
    {
    	if (slots[h].key == key)
    	{
    		value = slots[h].value;
    		break;
    	}
    	else if (slots[h].key == 0)
    	{
    		break;
    	}
    	h = (h + 1) % size;
    	iterations++;
    }
    values[idx] = value;
}

/**
 * Function constructor GpuHashTable
 * Performs init
 */
GpuHashTable::GpuHashTable(int size) {
	cudaError_t rt;
	this->size = size;
	this->count = 0;
	rt = glbGpuAllocator->_cudaMalloc((void**)&slots, size * sizeof(struct Slot));
	DIE(rt != cudaSuccess, "_cudaMalloc");
	rt = cudaMemset(slots, 0, size * sizeof(struct Slot));
	DIE(rt != cudaSuccess, "cudaMemset");
}

/**
 * Function desctructor GpuHashTable
 */
GpuHashTable::~GpuHashTable() {
	cudaError_t rt = glbGpuAllocator->_cudaFree(slots);
	DIE(rt != cudaSuccess, "_cudaFree");
}

/**
 * Function reshape
 * Performs resize of the hashtable based on load factor
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	if (numBucketsReshape <= size)
		return;
	cudaError_t rt;
	/* allocate memory corresponding to the new size */
	struct Slot *new_slots;
	rt = glbGpuAllocator->_cudaMalloc((void**)&new_slots, numBucketsReshape * sizeof(struct Slot));
	DIE(rt != cudaSuccess, "_cudaMalloc");
	rt = cudaMemset(new_slots, 0, numBucketsReshape * sizeof(struct Slot));
	DIE(rt != cudaSuccess, "cudaMemset");
	/* launch kernel to redistribute all key-value pairs */
	int numBlocks = size / BLOCK_SIZE;
	if (size % BLOCK_SIZE)
		numBlocks++;
	redistribute<<<numBlocks, BLOCK_SIZE>>>(new_slots, numBucketsReshape, slots, size);
	cudaDeviceSynchronize();
	/* free old memory */
	rt = glbGpuAllocator->_cudaFree(slots);
	DIE(rt != cudaSuccess, "_cudaFree");
	slots = new_slots;
	size = numBucketsReshape;
}

/**
 * Function insertBatch
 * Inserts a batch of key:value, using GPU and wrapper allocators
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	/* reshape hashtable if necessary */
	if ((float)(count + numKeys) / size >= LOAD_MAX)
	{
		unsigned int new_size = (unsigned int)((float)(count + numKeys) / LOAD_AVG);
		reshape(new_size);
	}
	cudaError_t rt;
	/* allocate memory and copy data to device */
	unsigned int *inserted;
	rt = glbGpuAllocator->_cudaMallocManaged((void**)&inserted, sizeof(unsigned int));
	DIE(rt != cudaSuccess, "_cudaMallocManaged");
	*inserted = 0;
	int *keys_device;
	rt = glbGpuAllocator->_cudaMalloc((void**)&keys_device, numKeys * sizeof(int));
	DIE(rt != cudaSuccess, "_cudaMalloc");
	rt = cudaMemcpy(keys_device, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(rt != cudaSuccess, "cudaMemcpy");
	int *values_device;
	rt = glbGpuAllocator->_cudaMalloc((void**)&values_device, numKeys * sizeof(int));
	DIE(rt != cudaSuccess, "_cudaMalloc");
	rt = cudaMemcpy(values_device, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(rt != cudaSuccess, "cudaMemcpy");
	/* launch kernel */
	int numBlocks = numKeys / BLOCK_SIZE;
	if (numKeys % BLOCK_SIZE)
		numBlocks++;
	insert_batch<<<numBlocks, BLOCK_SIZE>>>(slots, size, inserted, keys_device, values_device, numKeys);
	cudaDeviceSynchronize();
	/* update hashtable count */
	count += *inserted;
	/* free memory */
	rt = glbGpuAllocator->_cudaFree(inserted);
	DIE(rt != cudaSuccess, "_cudaFree");
	rt = glbGpuAllocator->_cudaFree(keys_device);
	DIE(rt != cudaSuccess, "_cudaFree");
	rt = glbGpuAllocator->_cudaFree(values_device);
	DIE(rt != cudaSuccess, "_cudaFree");
	return false;
}

/**
 * Function getBatch
 * Gets a batch of key:value, using GPU
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	cudaError_t rt;
	/* allocate memory and copy data to device */
	int *values = (int*)malloc(numKeys * sizeof(int));
	DIE(values == NULL, "malloc");
	int *keys_device;
	rt = glbGpuAllocator->_cudaMalloc((void**)&keys_device, numKeys * sizeof(int));
	DIE(rt != cudaSuccess, "_cudaMalloc");
	rt = cudaMemcpy(keys_device, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	DIE(rt != cudaSuccess, "cudaMemcpy");
	int *values_device;
	rt = glbGpuAllocator->_cudaMalloc((void**)&values_device, numKeys * sizeof(int));
	DIE(rt != cudaSuccess, "_cudaMalloc");
	/* launch kernel */
	int numBlocks = numKeys / BLOCK_SIZE;
	if (numKeys % BLOCK_SIZE)
		numBlocks++;
	get_batch<<<numBlocks, BLOCK_SIZE>>>(values_device, keys_device, numKeys, slots, size);
	cudaDeviceSynchronize();
	/* copy data to host and free memory */
	rt = cudaMemcpy(values, values_device, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	DIE(rt != cudaSuccess, "cudaMemcpy");
	rt = glbGpuAllocator->_cudaFree(keys_device);
	DIE(rt != cudaSuccess, "_cudaFree");
	rt = glbGpuAllocator->_cudaFree(values_device);
	DIE(rt != cudaSuccess, "_cudaFree");
	return values;
}

/**
 * Function load_factor
 * Returns the load factor of the hashtable
 */
float GpuHashTable::load_factor(void) {
	return (float)count / size;
}
