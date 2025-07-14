#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define N 32 * 1024 * 1024
#define THREAD_PER_BLOCK 256

// 2dims-tensor SoftMax
template <int Rows, int Cols>
__global__ void softmaxNomal(float *input, float *output)
{
    unsigned int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    if (tidx < Rows)
    {
        float *input_start = input + tidx * Cols;
        float *output_start = output + tidx * Cols;

        float max_val = -1e20f;
        for (int i = 0; i < Cols; i++)
        {
            max_val = max(max_val, input_start[i]);
        }

        float sum_val = 0.f;
        for (int i = 0; i < Cols; i++)
        {
            float e = expf(input_start[i] - max_val);
            sum_val += e;
            output_start[i] = e;
        }

        for (int i = 0; i < Cols; i++)
        {
            output_start[i] /= sum_val;
        }
    }
}

template <int Rows, int Cols>
void CpuCalculate(float *input, float *output)
{
    for (int tidx = 0; tidx < Rows; tidx++)
    {
        float *input_start = input + tidx * Cols;
        float *output_start = output + tidx * Cols;

        float max_val = -1e20f;
        for (int i = 0; i < Cols; i++)
        {
            max_val = max(max_val, input_start[i]);
        }

        float sum_val = 0.f;
        for (int i = 0; i < Cols; i++)
        {
            float e = expf(input_start[i] - max_val);
            sum_val += e;
            output_start[i] = e;
        }

        for (int i = 0; i < Cols; i++)
        {
            output_start[i] /= sum_val;
        }
    }
}

template <int Rows, int Cols>
bool checkResult(float *cpu_result, float *gpu_result)
{
    for (int i = 0; i < Rows; i++)
    {
        for (int j = 0; j < Cols; j++)
        {
            if (abs(cpu_result[i * Cols + j] - gpu_result[i * Cols + j]) > 0.01)
                return false;
        }
    }
    return true;
}

template <int Rows, int Cols>
void printResult(float *gpu_result)
{
    for (int i = 0; i < Rows; i++)
    {
        for (int j = 0; j < Cols; j++)
        {
            printf("%f ", gpu_result[i * Cols + j]);
        }
        printf("\n");
    }
}

int main()
{
    constexpr int Rows = 256;
    constexpr int Cols = 256;
    // cpu alloc
    float *input = (float *)malloc(Rows * Cols * sizeof(float));
    float *gpu_result = (float *)malloc(Rows * Cols * sizeof(float));

    float *cpu_result = (float *)malloc(Rows * Cols * sizeof(float));

    // init
    for (int i = 0; i < Rows; i++)
    {
        for (int j = 0; j < Cols; j++)
            input[i * Cols + j] = j;
    }

    // cpu result
    CpuCalculate<Rows, Cols>(input, cpu_result);

    // gpu alloc
    float *d_input;
    cudaMalloc((void **)&d_input, Rows * Cols * sizeof(float));
    cudaMemcpy(d_input, input, Rows * Cols * sizeof(float), cudaMemcpyHostToDevice);

    float *d_output = (float *)malloc(Rows * Cols * sizeof(float));
    cudaMalloc((void **)&d_output, Rows * Cols * sizeof(float));

    dim3 Grid(1, 1);
    dim3 Block(Rows, 1);
    softmaxNomal<Rows, Cols><<<Grid, Block>>>(d_input, d_output);

    cudaMemcpy(gpu_result, d_output, Rows * Cols * sizeof(float), cudaMemcpyDeviceToHost);

    if (checkResult<Rows, Cols>(cpu_result, gpu_result))
    {
        printf("Result is correct!\n");
        // printResult<Rows, Cols>(gpu_result);
    }
    else
        printf("Result is incorret!\n");
}