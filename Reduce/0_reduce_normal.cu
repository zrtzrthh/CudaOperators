#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define N 32*1024*1024
#define THREAD_PER_BLOCK 256

__global__ void reduceNomal(float *input, float *output)
{    
    unsigned int global_id = threadIdx.x + blockDim.x*blockIdx.x;

    for(int i = 1; i < blockDim.x; i *= 2)
    {
        if(threadIdx.x % (i*2) == 0)
            input[global_id] += input[global_id + i];
        __syncthreads();
    } 
    if(threadIdx.x == 0)
        output[blockIdx.x] = input[global_id];
}

bool checkResult(float *out, float *res, int n)
{
    for(int i = 0; i < n; i++)
    {
        if(abs(out[i] - res[i]) > 0.5) return false;
    }
    return true;
}

int main()
{
    // cpu alloc
    float *input = (float *)malloc(N*sizeof(float));
    int num_block = ceil((float)N/THREAD_PER_BLOCK);
    float *output = (float *)malloc(num_block*sizeof(float));

    float *cpu_result = (float *)malloc(num_block*sizeof(float));

    // init
    for(int i = 0; i < N; i++)
    {
        input[i] = 1; 
    }

    // cpu result
    for(int i = 0; i < num_block; i++)
    {
        float ans = .0;
        for(int j = 0; j < ((i < num_block - 1)? 
                            THREAD_PER_BLOCK:
                            N - (num_block - 1)*THREAD_PER_BLOCK); j++)
        {
            ans += input[i*THREAD_PER_BLOCK + j];
        }
        
        cpu_result[i] = ans;
    }

    // gpu alloc
    float *d_input;
    cudaMalloc((void **)&d_input, N*sizeof(float));
    cudaMemcpy(d_input, input, N*sizeof(float), cudaMemcpyHostToDevice);
    
    float *d_output = (float *)malloc(num_block*sizeof(float));
    cudaMalloc((void **)&d_output, num_block*sizeof(float));

    dim3 Grid(num_block, 1);
    dim3 Block(THREAD_PER_BLOCK, 1);
    reduceNomal<<<Grid, Block>>>(d_input, d_output);

    cudaMemcpy(output, d_output, num_block*sizeof(float), cudaMemcpyDeviceToHost);
    
    if(checkResult(output, cpu_result, num_block))
        printf("Result is correct!\n");
    else 
        printf("Result is incorret!\n");

}