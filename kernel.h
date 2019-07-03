#ifndef KERNEL_CUH_
#define KERNEL_CUH_

template<typename cuda_fixnum>
class Kernel {
	cuda_fixnum n;
	//modexp<cuda_fixnum> pow;

public:
	__device__ static cuda_fixnum scalarMatrixMultiplicationKernel(
		cuda_fixnum* vector,
		cuda_fixnum* scalar,
		cuda_fixnum* result,
		int size);

	__device__ static cuda_fixnum scalarMatrixMultiplication(
		cuda_fixnum* vector, 
		cuda_fixnum* scalar, 
		cuda_fixnum* result, 
		int size);


	cuda_fixnum setValue(cuda_fixnum* value);
};

#endif