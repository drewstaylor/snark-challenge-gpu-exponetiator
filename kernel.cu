#include <vector>
#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "kernel.h"
#include <stdlib.h>

using namespace std;

__global__ void scalarMatrixMultiplicationKernel(
    float* vector,
    float* scalar,
    float* result,
    int size) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    float local_value = 0;

    if (ROW < size && COL < size) {
        // Each thread computes one element of the block sub-matrix
        // XXX TODO: Add mixed addition / checks to not multiply against 0

        // Set scalar multiples
        for (int i = 0; i < size; i++) {
            local_value += vector[ROW * size + i] * scalar[ROW * size + i];
        }
    }
    result[ROW * size + COL] = local_value;
}


void scalarMatrixMultiplication(
    float* vector,
    float* scalar,
    float* result,
    int size) {

        // declare the number of blocks per grid and the number of threads per block
        // use 1 to 512 threads per block
        dim3 threadsPerBlock(size, size);
        dim3 blocksPerGrid(1, 1);
            // Threads overflow safety
            if (size*size > 512) {
                threadsPerBlock.x = 512;
                threadsPerBlock.y = 512;
                blocksPerGrid.x = ceil(int(size)/int(threadsPerBlock.x));
                blocksPerGrid.y = ceil(int(size)/int(threadsPerBlock.y));
            }

        // Run processing
        scalarMatrixMultiplicationKernel<<<blocksPerGrid,threadsPerBlock>>>(vector, scalar, result, size);
}