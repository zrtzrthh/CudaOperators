#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define N 32*1024*1024
#define THREAD_PER_BLOCK 256

__global__ void reduceNomal(float *d_input, float *d_output)
{
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int thread_id = threadIdx.x;
    unsigned int global_id = threadIdx.x + blockDim.x*blockIdx.x;
    sdata[thread_id] = d_input[global_id];
    __syncthreads();

    for(int i = 1; i < blockDim.x; i *= 2)
    {
        if(thread_id % (i*2) == 0)
            sdata[thread_id] += sdata[thread_id + i];
        __syncthreads();
    } 
    if(thread_id == 0)
        d_output[blockIdx.x] = sdata[0];
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
    // cpu malloc
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

    // gpu malloc
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