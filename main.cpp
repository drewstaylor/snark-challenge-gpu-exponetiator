#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "kernel.h"
#include "dev_array.h"
#include <math.h>

int main()
{
	// Temporary test inputs
	int N = 16;
	int SIZE = N * N;

	// Allocate memory on the host
	std::vector<float> vector_cpu(SIZE);
	std::vector<float> scalar_cpu(SIZE);
	std::vector<float> result_cpu(SIZE);

	// Allocate memory on the device
	dev_array<float> vector_gpu(SIZE);
	dev_array<float> scalar_gpu(SIZE);
	dev_array<float> result_gpu(SIZE);

	vector_gpu.set(&vector_cpu[0], SIZE);
	scalar_gpu.set(&scalar_cpu[0], SIZE);

	// Run worker
	scalarMatrixMultiplication(vector_gpu.getData(), scalar_gpu.getData(), result_gpu.getData(), N);
	// Parallelize worker
	cudaDeviceSynchronize();

	// We out
	std::cout << "Input [Vector]: " << vector_gpu.getData() <<"\n";
	std::cout << "Input [Scalar]: " << scalar_gpu.getData() << "\n";
	std::cout << "Output [Scalar Multiple]: " << result_gpu.getData() << "\n";
	std::cout << "Exiting with code 0";
	return 0;
}