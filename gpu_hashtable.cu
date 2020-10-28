#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include "gpu_hashtable.hpp"


/*	HASH function	*/
__device__ int hashFunc(int key) {
	key = (int)((long long)(key * NR_A + NR_C) % NR_B);
	return key;
}


/*
 *	(kernel)INSERT function called by each thread
 *	each thread gets key&value and procceses insertion
 *	location in hash table
 */
__global__ void kernel_insert(int* deviceKeys, int* deviceValues,
							  int* deviceDupes, hashStruct hashTable) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int kickedKey, key, value, b, k, hash;

	/* Check if index out of bounds	*/
	if (i >= hashTable.numKeys) {
		return;
	}

	key = deviceKeys[i];
	value = deviceValues[i];
	hash = (int)(hashFunc(key) % hashTable.sizeHash);

	/*Check coresponding bucket	*/
	for (k = 0; k < BUCKET_SIZE; k++) {
		kickedKey = atomicCAS(&hashTable.hash[k][hash].key, KEY_INVALID, key);
		if (kickedKey == KEY_INVALID || kickedKey == key) {
			hashTable.hash[k][hash].value = value;
			/* Check if updated key	*/
			if (kickedKey == key) {
				atomicAdd(&deviceDupes[0], 1);
			}
			return;
		}
	}
	/*	Check next buckets	*/
	for (b = hash + 1; b < hashTable.sizeHash; b++) {
		for (k = 0; k < BUCKET_SIZE; k++) {
			kickedKey = atomicCAS(&hashTable.hash[k][b].key, KEY_INVALID, key);
			if (kickedKey == KEY_INVALID || kickedKey == key) {
				hashTable.hash[k][b].value = value;
				/* Check if updated key	*/
				if (kickedKey == key) {
					atomicAdd(&deviceDupes[0], 1);
				}
				return;
			}
		}
	}
	/*	Check rear buckets	*/
	for (b = hash - 1; b >= 0; b--) {
		for (k = 0; k < BUCKET_SIZE; k++) {
			kickedKey = atomicCAS(&hashTable.hash[k][b].key, KEY_INVALID, key);
			if (kickedKey == KEY_INVALID || kickedKey == key) {
				hashTable.hash[k][b].value = value;
				/* Check if updated key	*/
				if (kickedKey == key) {
					atomicAdd(&deviceDupes[0], 1);
				}
				return;
			}
		}
	}
}

/*
 *	(kernel)GET function called by each thread
 *	each thread retireves a value for a key from hash
 */
__global__ void kernel_get(int *keys, int *values, hashStruct hashTable) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int key, hash, k, b;

	/* Check if index out of bounds	*/
	if (i >= hashTable.numKeys) {
		return;
	}

	key = keys[i];
	hash = int(hashFunc(key) % hashTable.sizeHash);

	/*	Check coresonding bucket	*/
	for (k = 0; k < BUCKET_SIZE; k++) {
		if (hashTable.hash[k][hash].key == key) {
			values[i] = hashTable.hash[k][hash].value;
			return;
		}
	}
	/*	Check next buckets	*/
	for (b = hash + 1; b < hashTable.sizeHash; b++) {
		for (k = 0; k < BUCKET_SIZE; k++) {
			if (hashTable.hash[k][b].key == key) {
				values[i] = hashTable.hash[k][b].value;
				return;
			} 
		}
	}
	/*	Check rear buckets	*/
	for (b = hash - 1; b >= 0; b--) {
		for (k = 0; k < BUCKET_SIZE; k++) {
			if (hashTable.hash[k][b].key == key) {
				values[i] = hashTable.hash[k][b].value;
				return;
			} 
		}
	}
}

/*
 *	(kernel)REHASH function called by each thread
 *	each thread gets a bucket from an hash index
 *	and reinserts nodes from bucket (where they have
 *	valid key) in new hah table
 */
__global__ void kernel_rehash(hashStruct oldHash, hashStruct newHash) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int b, k, flag, indexOldHash;
	int kickedKey, key, hash, value;

	/* Check if index out of bounds	*/
	if (i >= oldHash.sizeHash) {
		return;
	}

	/*	For each node in bucket at index i -> reinsert in new hash	*/
	for (indexOldHash = 0; indexOldHash <= 1; indexOldHash++) {
		if (oldHash.hash[indexOldHash][i].key == KEY_INVALID)
			continue;
		key = oldHash.hash[indexOldHash][i].key;
		value = oldHash.hash[indexOldHash][i].value;
		hash = (int)(hashFunc(key) % newHash.sizeHash);

		flag = 0;
		/*	Check coresponding bucket	*/
		for (k = 0; k < BUCKET_SIZE; k++) {
			kickedKey = atomicCAS(&newHash.hash[k][hash].key, KEY_INVALID, key);
			if (kickedKey == KEY_INVALID) {
				newHash.hash[k][hash].value = value;
				flag = 1;
				break;
			}
		}
		/*	Check if node has been inserted	*/
		if (flag == 1) {
			continue;
		}

		/*	Check next buckets	*/
		for (b = hash + 1; b < newHash.sizeHash; b++) {
			for (k = 0; k < BUCKET_SIZE; k++) {
				kickedKey = atomicCAS(&newHash.hash[k][b].key, KEY_INVALID, key);
				if (kickedKey == KEY_INVALID) {
					newHash.hash[k][b].value = value;
					flag = 1;
					break;
				}
			}
			/*	Check if node has been inserted	*/
			if (flag == 1) {
				break;
			}
		}
		/*	Check if node has been inserted	*/
		if (flag == 1) {
			continue;
		}

		/*	Check rear buckets	*/
		for (b = hash - 1; b >= 0; b--) {
			for (k = 0; k <BUCKET_SIZE; k++) {
				kickedKey = atomicCAS(&newHash.hash[k][b].key, KEY_INVALID, key);
				if (kickedKey == KEY_INVALID) {
					newHash.hash[k][b].value = value;
					flag = 1;
					break;
				}
			}
			/*	Check if node has been inserted	*/
			if (flag == 1) {
				break;
			}
		}
	}
}

/* 
 *	INIT HASH - init fields of hash structure
 */
GpuHashTable::GpuHashTable(int size) {
	int k;
	int spaceNodes = size * sizeof(hashNode);

	hashTable.sizeHash = size;
	hashTable.numKeys = 0;
	hashTable.error = 0;
	nrNodes = 0;
	for (k = 0; k < BUCKET_SIZE; k++) {
		cudaMalloc((void **)&hashTable.hash[k], spaceNodes);
		if (hashTable.hash[k] == 0) {
			std::cerr << "Memory allocation failed\n";
			hashTable.error = 1;
			return;
		}
		cudaMemset(hashTable.hash[k], KEY_INVALID, spaceNodes);
	}
}

/*
 *	DESTROY HASH and fields associated with hash struct
 */
GpuHashTable::~GpuHashTable() {
	int k;

	for (k = 0; k < BUCKET_SIZE; k++) {
		cudaFree(hashTable.hash[k]);
		hashTable.hash[k] = nullptr;
	}
}

/* 
 *	RESHAPE HASH -> increase size and reinsert nodes
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	int block_size = 1024;
	int k, sizeNew = numBucketsReshape * sizeof(hashNode);
	int blocks_no = (int)(numBucketsReshape / block_size);
	hashStruct newHashmap;
	cudaError_t cudaerr;

	/*	Update size	*/
	newHashmap.sizeHash = numBucketsReshape;

	if (numBucketsReshape % block_size) {
		++blocks_no;
	}

	for (k = 0; k < BUCKET_SIZE; k++) {
		cudaMalloc((void **)&newHashmap.hash[k], sizeNew);
		if (newHashmap.hash[k] == 0) {
			std::cerr << "Memory allocation failed\n";
			hashTable.error = 1;
			return;
		}
		cudaMemset(newHashmap.hash[k], KEY_INVALID, sizeNew);
	}

	/*	Call kernel function	*/
	kernel_rehash<<<blocks_no, block_size>>>(hashTable, newHashmap);

	/*	Sync memory	*/
	cudaerr = cudaDeviceSynchronize();
	if (cudaerr != cudaSuccess) {
		std::cerr<<"kernel launch failed with error \n";
		hashTable.error = 1;
		return;
	}

	/*	Free memory	*/
	for (k = 0; k < BUCKET_SIZE; k++) {
		cudaFree(hashTable.hash[k]);
	}

	/*	Update table	*/
	hashTable = newHashmap;
}

/* INSERT BATCH
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
	int block_size = 1024;
	int blocks_no = (int)(numKeys / block_size);
	int *deviceKeys, *deviceValues, *deviceDupes;
	int num_bytes = numKeys * sizeof(int);
	cudaError_t cudaerr;

	if (numKeys % block_size) 
		++blocks_no;

	cudaMalloc((void **)&deviceKeys, num_bytes);
	cudaMalloc((void **)&deviceValues, num_bytes);
	cudaMalloc((void **)&deviceDupes, 1 * sizeof(int));

	if(deviceKeys == 0 || deviceValues == 0 || deviceDupes == 0) {
		std::cerr << "Memory allocation failed on device\n";
		hashTable.error = 1;
		return false;
	}

	cudaMemcpy(deviceKeys, keys, num_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceValues, values, num_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceDupes, 0, sizeof(int), cudaMemcpyHostToDevice);
	
	if (nrNodes + numKeys > hashTable.sizeHash)
		reshape(int((nrNodes + numKeys) * 5 / 4));

	/*	Check if reshape was completed successfully	*/
	if (hashTable.error == 1)
		return false;
	
	/*	Set data for kernel function	*/
	hashTable.numKeys = numKeys;

	/*	Call kernel function	*/
	kernel_insert<<<blocks_no, block_size>>>(deviceKeys, deviceValues,
											deviceDupes, hashTable);

	/*	Sync memory	*/
	cudaerr = cudaDeviceSynchronize();
	if (cudaerr != cudaSuccess) {
		std::cerr<<"kernel launch failed with error \n";
		hashTable.error = 1;
		return false;
	}

	/*	Free memory	*/
	cudaFree(deviceKeys);
	cudaFree(deviceValues);

	/*	Update number of nodes after insertion	*/
	cudaMemcpy(hashTable.dupes, deviceDupes, sizeof(int),
				cudaMemcpyDeviceToHost);
	if (hashTable.dupes[0]) {
		nrNodes += numKeys;
	} else {
		nrNodes += (numKeys - hashTable.dupes[0]);
	}
	
	return true;
}

/* GET BATCH of values from key associations
 */
int *GpuHashTable::getBatch(int *keys, int numKeys) {
	int block_size = 1024;
	int num_bytes = numKeys * sizeof(int);
	int blocks_no = (int)(numKeys / block_size);
	int *deviceKeys, *resultValues, *hostResultValues;
	cudaError_t cudaerr;

	if (numKeys % block_size) {
		++blocks_no;
	}

	cudaMalloc((void **)&deviceKeys, num_bytes);
	cudaMalloc((void **)&resultValues, num_bytes);
	hostResultValues = (int *)malloc(num_bytes);

	if(deviceKeys == 0 || resultValues == 0 || hostResultValues == 0) {
		std::cerr << "Memory allocation failed on device"<<std::endl;
		hashTable.error = 1;
		return nullptr;
	}

	cudaMemset(resultValues, KEY_INVALID, num_bytes);
	cudaMemset(hostResultValues, KEY_INVALID, num_bytes);

	/*	Set data for kernel function	*/
	cudaMemcpy(deviceKeys, keys, num_bytes, cudaMemcpyHostToDevice);
	hashTable.numKeys = numKeys;

	/*	Call kernel function	*/
	kernel_get<<<blocks_no, block_size>>>(deviceKeys, resultValues, hashTable);

	/*	Sync memory	*/
	cudaerr = cudaDeviceSynchronize();
	if (cudaerr != cudaSuccess) {
		std::cerr<<"kernel launch failed with error \n";
		hashTable.error = 1;
		return nullptr;
	}

	/*	Get results from VRAM	*/
	cudaMemcpy(hostResultValues, resultValues, num_bytes, cudaMemcpyDeviceToHost);
	
	/*	Free memory	*/
	cudaFree(deviceKeys);
	cudaFree(resultValues);
	return hostResultValues;
}

/* GET LOAD FACTOR
 * num elements / hash total slots elements
 */
float GpuHashTable::loadFactor() {
	if (hashTable.sizeHash == 0)
		return 0;
	return (float(nrNodes) / hashTable.sizeHash);
}

/*********************************************************/

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"