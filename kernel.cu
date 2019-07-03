#include <vector>
#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "kernel.h"
#include <stdlib.h>

// fixnum
#include "array/fixnum_array.h"
#include "fixnum/word_fixnum.cu"
#include "fixnum/warp_fixnum.cu"
#include "functions/modexp.cu"
#include "modnum/modnum_monty_cios.cu"
#include "modnum/modnum_monty_redc.cu"

using namespace std;
using namespace cuFIXNUM;

template <typename T>
	__device__ T Kernel<T>::scalarMatrixMultiplicationKernel(
		T* vector,
		T* scalar,
		T* result,
		int size) {

		int ROW = blockIdx.y*blockDim.y + threadIdx.y;
		int COL = blockIdx.x*blockDim.x + threadIdx.x;

		T local_value = 0;

		if (ROW < size && COL < size) {
			// Each thread computes one element of the block sub-matrix
			// XXX TODO: Add mixed addition / checks to not multiply against 0

			// Set scalar multiples
			for (int i = 0; i < size; i++) {
				//local_value += vector[ROW * size + i] * scalar[ROW * size + i];
				local_value += modexp<T>::pow(vector[ROW * size + i] * scalar[ROW * size + i]);
			}
		}
		result[ROW * size + COL] = local_value;
	}

template <typename T>
	__device__ T Kernel<T>::scalarMatrixMultiplication(
		T* vector,
		T* scalar,
		T* result,
		int size) {

		// declare the number of blocks per grid and the number of threads per block
		// use 1 to 512 threads per block
		dim3 threadsPerBlock(size, size);
		dim3 blocksPerGrid(1, 1);
		// Threads overflow safety
		if (size*size > 512) {
			threadsPerBlock.x = 512;
			threadsPerBlock.y = 512;
			blocksPerGrid.x = ceil(int(size) / int(threadsPerBlock.x));
			blocksPerGrid.y = ceil(int(size) / int(threadsPerBlock.y));
		}
		
		// Run processing
		Kernel<T>::scalarMatrixMultiplicationKernel <<<blocksPerGrid, threadsPerBlock>>> (vector, scalar, result, size);
	}

template <typename T>
	T Kernel<T>::setValue(T* value) {
		n = value;
	}