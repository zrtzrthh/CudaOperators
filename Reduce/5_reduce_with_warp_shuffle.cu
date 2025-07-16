#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define N 10 * 64
#define THREAD_PER_BLOCK 63
#define WARPSIZE 32

__global__ void reduceNomal(float *d_input, float *d_output)
{
    __shared__ float sData[32];
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    unsigned int warp_id = threadIdx.x / WARPSIZE;
    unsigned int lane_id = threadIdx.x % WARPSIZE;

    float val = idx < N ? d_input[idx] : 0.f;

    for (int i = WARPSIZE >> 1; i >0; i >>= 1)
    {
        val += __shfl_down_sync(0Xffffffff, val, i);
    }

    if(lane_id == 0) sData[warp_id] = val;

    __syncthreads();

    if(warp_id == 0)
    {
        unsigned int warp_sum = ceil(blockDim.x / WARPSIZE);
        float val = lane_id < warp_sum ? sData[lane_id] : 0.f;
        for (int i = WARPSIZE >> 1; i >0; i >>= 1)
        {
            val += __shfl_down_sync(0Xffffffff, val, i);
        }

        if(lane_id == 0) atomicAdd(d_output, val);
    }
}

bool checkResult(float *gpu_result, float *cpu_result)
{
    if (abs(gpu_result[0] - cpu_result[0]) > 0.5)
    {
        return false;
    }
    return true;
}

int main()
{
    // cpu malloc
    float *input = (float *)malloc(N * sizeof(float));
    int num_block = ceil((float)N / THREAD_PER_BLOCK);
    float *gpu_result = (float *)malloc(sizeof(float));

    float *cpu_result = (float *)malloc(sizeof(float));
    memset(cpu_result, 0, sizeof(float));

    // init
    for (int i = 0; i < N; i++)
    {
        input[i] = 1;
    }

    // cpu result
    for(int i = 0; i < N; i++)
    {
        cpu_result[0] += input[i];
    }

    // gpu malloc
    float *d_input;
    cudaMalloc((void **)&d_input, N * sizeof(float));
    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    float *d_output = (float *)malloc(sizeof(float));
    cudaMalloc((void **)&d_output, sizeof(float));

    dim3 Grid(num_block, 1);
    dim3 Block(THREAD_PER_BLOCK, 1);
    reduceNomal<<<Grid, Block>>>(d_input, d_output);

    cudaMemcpy(gpu_result, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    if (checkResult(gpu_result, cpu_result))
        printf("Result is correct!\n");
    else
        printf("Result is incorret!\n");

    printf("%f\n", cpu_result[0]);
    printf("%f", gpu_result[0]);
}