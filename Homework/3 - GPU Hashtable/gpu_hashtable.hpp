#ifndef _HASHCPU_
#define _HASHCPU_

#include <vector>

using namespace std;

#define cudaCheckError() { \
	cudaError_t e=cudaGetLastError(); \
	if(e!=cudaSuccess) { \
		cout << "Cuda failure " << __FILE__ << ", " << __LINE__ << ", " << cudaGetErrorString(e); \
		exit(0); \
	 }\
}

#define BLOCK_SIZE 256

#define LOAD_MAX 0.95f
#define LOAD_AVG 0.55f

/**
 * Struct Slot used to store a key-value pair inside the hashtable
 */
struct Slot
{
	unsigned int key;
	unsigned int value;
};

/**
 * Class GpuHashTable to implement functions
 */
class GpuHashTable
{
private:
	unsigned int size;  /* total capacity */
	unsigned int count; /* used slots */
	struct Slot *slots; /* array of slots stored on the GPU */

public:
	GpuHashTable(int size);
	void reshape(int sizeReshape);
	bool insertBatch(int *keys, int* values, int numKeys);
	int* getBatch(int* key, int numItems);
	float load_factor(void);
	~GpuHashTable();
};

#endif

