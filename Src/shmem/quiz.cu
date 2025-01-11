#include <assert.h>
#include <math.h>
#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#define A(i, j) (a[(i)*n + (j)])
#define B(i, j) (b[(i)*n + (j)])
#define C(i, j) (c[(i)*n + (j)])

#define BLOCKDIM_X              32
#define BLOCKDIM_Y              32
#define TILEDIM                 128
#define TILEDIM_K               32
#define COMPUTE_PER_THREAD      4
#define TILEDIM_M 		TILEDIM 
#define TILEDIM_N 		TILEDIM 
#define 			SHARED_STRIDE ((TILEDIM)*(TILEDIM_K))/((BLOCKDIM_X)*(BLOCKDIM_Y))

using namespace std;

__global__ void 
__launch_bounds__(1024)
matMul(int N, float *C, float *A, float *B) {

	const int b_x = blockIdx.x;
	const int b_y = blockIdx.y;
	
	const int SHARED_STRIDES = (TILEDIM*TILEDIM_K)/(BLOCKDIM_X*BLOCKDIM_Y);

	const int A_tile_offset = b_y*TILEDIM*N;
	const int B_tile_offset = b_x*TILEDIM;

	const int C_tile_offset = b_y*TILEDIM*N + b_x*TILEDIM;

	const int A_tile_off_bndry = b_y*TILEDIM;
	const int B_tile_off_bndry = b_x*TILEDIM;

	A += A_tile_offset;
	B += B_tile_offset;
	C += C_tile_offset; 

	extern __shared__ float shared_memory_start_addr[];
	float* As;
	float* Bs;
	As = shared_memory_start_addr;
	Bs = shared_memory_start_addr + TILEDIM*TILEDIM_K;

	const int shared_column_a = ((threadIdx.x + blockDim.x*threadIdx.y)) % TILEDIM_K;
	const int shared_row_a = ((threadIdx.x + blockDim.x*threadIdx.y)) / TILEDIM_K;
	const int shared_column_b = ((threadIdx.x + blockDim.x*threadIdx.y)) % TILEDIM;
	const int shared_row_b = ((threadIdx.x + blockDim.x*threadIdx.y)) / TILEDIM;
	float bReg;
	float cReg[COMPUTE_PER_THREAD][COMPUTE_PER_THREAD] = {0.0};

	const int x = ((threadIdx.x + blockDim.x*threadIdx.y)) % ( TILEDIM / COMPUTE_PER_THREAD );
	const int y = ((threadIdx.x + blockDim.x*threadIdx.y)) / ( TILEDIM / COMPUTE_PER_THREAD );

	for(unsigned int blockK = 0; blockK < N; blockK += TILEDIM_K)
	{
		for(int shared_strip = 0; shared_strip < SHARED_STRIDES ; shared_strip++) {
			As[(shared_row_a + shared_strip*(TILEDIM/SHARED_STRIDES)) * TILEDIM_K + shared_column_a] = ((shared_row_a + shared_strip*(TILEDIM/SHARED_STRIDES)) +  b_y*TILEDIM < N  && shared_column_a + blockK < N) ?  A[(shared_row_a +  shared_strip*(TILEDIM/SHARED_STRIDES)) * N  + shared_column_a] : 0;
			Bs[(shared_row_b + shared_strip*(TILEDIM_K/SHARED_STRIDES)) * TILEDIM   + (shared_column_b )] = ((shared_row_b + shared_strip*(TILEDIM_K/SHARED_STRIDES)) +  blockK < N  && (shared_column_b ) + b_x*TILEDIM < N) ?  B[(shared_row_b + shared_strip*(TILEDIM_K/SHARED_STRIDES)) * N + (shared_column_b )] : 0;
		}
		__syncthreads();
		A += TILEDIM_K;
		B += TILEDIM_K*N;
		for(unsigned int k = 0; k < TILEDIM_K; k++)
		{
			for(unsigned int sub_b = 0; sub_b < COMPUTE_PER_THREAD; sub_b++)
			{
				bReg = Bs[k * TILEDIM + (x * COMPUTE_PER_THREAD + sub_b  )];
				for(unsigned int sub_a = 0; sub_a < COMPUTE_PER_THREAD; sub_a++)
				{
					cReg[sub_b][sub_a] += As[(y * COMPUTE_PER_THREAD + sub_a) * TILEDIM_K + k] * bReg;
				}
			}
		}
		__syncthreads();
		
	}

	for(unsigned int sub_b = 0; sub_b < COMPUTE_PER_THREAD; sub_b++)
	{
		for(unsigned int sub_a = 0; sub_a < COMPUTE_PER_THREAD; sub_a++)
		{
			if(((y * COMPUTE_PER_THREAD + sub_a + A_tile_off_bndry) < N) &&  (((x * COMPUTE_PER_THREAD) + sub_b + B_tile_off_bndry) < N))
			{
				C[((y * COMPUTE_PER_THREAD) + sub_a)*N + ((x * COMPUTE_PER_THREAD) + sub_b  )  ] = cReg[sub_b][sub_a];
			}
		}
	}
}

double getPerf(int n, int reps, double time)
{
    long long int n2 = n;
    n2 *= n;
    const long long int updates = n2 * (long long)reps;
    const long long int flops = (long long)n * 2L * updates;
    double flop_rate = (double)flops / time;
    return (flop_rate / 1.0e9);
}

void check_cuda_failure(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", msg,
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void printMatrix(float *a, unsigned int m, unsigned int n)
{
    unsigned int i, j;
    cout.precision(4);
    cout.width(8);
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
            cout << A(i, j) << " ";
        cout << endl;
    }
}

void genMatrix(float *a, float *b, unsigned int n)
{
    unsigned int i,j;
    srand(time(NULL));
    float randomFloat = (float)rand() / RAND_MAX;

    memset(a, 0, n*n*sizeof(float));
    // Fill A
    for (i = 0; i < n; i++)
        A(i, i) = 1;
    
    // Fill B
    for (j = 0; j < n*n; j++) {
    	b[j] = (float)rand() / RAND_MAX;
    }	
}


int main(int argc, char **argv)
{
	assert((TILEDIM*TILEDIM)/(COMPUTE_PER_THREAD*COMPUTE_PER_THREAD) == BLOCKDIM_X*BLOCKDIM_Y);
	assert(TILEDIM*TILEDIM_K == SHARED_STRIDE*BLOCKDIM_X*BLOCKDIM_Y);

	// Matrix dimensions
	int n 	= 2048;
	int reps  = 250;

    printf("Running for %d reps ...\n", reps);

	unsigned int n2 = n * n * sizeof(float);
	const int _ntx = BLOCKDIM_X;
	const int _nty = BLOCKDIM_Y;

    	dim3 threads(_ntx, _nty, 1);
    	int numblocksX = n / _ntx;
    	int numblocksY = n / _nty;


    	if (n % _ntx != 0)
    	    numblocksX++;

    	if (n % _nty != 0)
    	    numblocksY++;

    	dim3 grid(numblocksX, numblocksY, 1);
   	grid.x = n / TILEDIM_N;
   	grid.y = n / TILEDIM_M;

   	if (n % TILEDIM_N != 0)
   	   grid.x++;
   	if (n % TILEDIM_M != 0)
   	   grid.y++;


    	float *h_A = (float *)malloc(n2);
    	assert(h_A);
    	float *h_B = (float *)malloc(n2);
    	assert(h_B);

        genMatrix(h_A, h_B, n);

    	float *d_A, *d_B, *d_C;
    	cudaMalloc((void **)&d_A, n2);
    	check_cuda_failure("Error allocating device memory for matrix A");

    	cudaMalloc((void **)&d_B, n2);
    	check_cuda_failure("Error allocating device memory for matrix B");

    	cudaMalloc((void **)&d_C, n2);
    	check_cuda_failure("Error allocating device memory for matrix C");

    	cudaMemset((void **)d_A, -99, n2);
    	check_cuda_failure("Error initializing device memory matrix A");

    	cudaMemset((void **)d_B, -99, n2);
    	check_cuda_failure("Error initializing device memory matrix B");

    	cudaMemset((void **)d_C, 0, n2);
    	check_cuda_failure("Error clearing device memory matrix C");

    	// copy host memory to device
    	cudaMemcpy(d_A, h_A, n2, cudaMemcpyHostToDevice);
    	check_cuda_failure("Error copying matrix A to device");
    	cudaMemcpy(d_B, h_B, n2, cudaMemcpyHostToDevice);
    	check_cuda_failure("Error copying matrix B to device");

    	// allocate host memory for the result
    	float *h_C = (float *)malloc(n2);
    	assert(h_C);

    	cudaFuncSetAttribute(matMul, cudaFuncAttributePreferredSharedMemoryCarveout,
    	    		 cudaSharedmemCarveoutMaxShared);
    	check_cuda_failure("Error seting shared memory to 64K");
    	cudaFuncSetAttribute(matMul, cudaFuncAttributeMaxDynamicSharedMemorySize,
    	    		 1024 * 64);
    	check_cuda_failure("Error seting shared memory to 64K");

	// Start the Timer
    	cudaEvent_t start_event, stop_event;
    	cudaEventCreate(&start_event);
    	cudaEventCreate(&stop_event);

    	cudaEventRecord(start_event, 0);
    	float t_device;

    	for (int r = 0; r < reps; r++)
    		matMul<<<grid, threads, 64 *1024>>>(n, d_C, d_A, d_B);

	// Stop the Timer
    	cudaEventRecord(stop_event, 0);
    	cudaEventSynchronize(stop_event);
    	cudaEventElapsedTime(&t_device, start_event, stop_event);
    	t_device /= 1000.0;
	
    	cudaDeviceSynchronize();

	float GFLOPS = getPerf(n, reps, t_device);

	// Verification of result

	// 1. Copy result from device to host
    	cudaMemcpy(h_C, d_C, n2, cudaMemcpyDeviceToHost);
    	check_cuda_failure("Unable to retrieve result from device");

	// 2. Check against h_B result
	//	2.1 Since h_B is identity, d_C should be equal to h_B
	if (memcmp(h_B, h_C, n2) == 0) {
    		printf("n: %d, tx: %d, ty: %d, repitions: %d, Perf: %0.2f GFLOPS\t PASSED!\n", n, threads.x, threads.y, reps, GFLOPS);
    	} else {
    		printf("n: %d, tx: %d, ty: %d, repitions: %d, Perf: %0.2f GFLOPS\t FAILED!\n", n, threads.x, threads.y, reps, GFLOPS);
    	}

    	free(h_A);
    	free(h_B);
    	free(h_C);

    	assert(cudaSuccess == cudaFree(d_A));
    	assert(cudaSuccess == cudaFree(d_B));
    	assert(cudaSuccess == cudaFree(d_C));

}

