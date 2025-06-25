#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define M 64
#define K 64
#define N 64

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32

// grid 2-dim block 2-dim
__global__ void GEMM(float *matrixA, float *matrixB, float *matrixC)
{
    const unsigned int tidx = threadIdx.x + blockDim.x*blockIdx.x;
    const unsigned int tidy = threadIdx.y + blockDim.y*blockIdx.y;

    float *A_start = matrixA + blockIdx.y*blockDim.y*K;
    float *B_start = matrixB + blockIdx.x*blockDim.x;
    __shared__ float sDataA[BLOCK_SIZE_Y][BLOCK_SIZE_X];
    __shared__ float sDataB[BLOCK_SIZE_Y][BLOCK_SIZE_X];

    if(tidy < M && tidx < N)
    {
        float temp = 0;
        for(int s = 0; s < K; s += blockDim.y)
        {
            sDataA[threadIdx.y][threadIdx.x] = A_start[threadIdx.y * K + s + threadIdx.x];
            sDataB[threadIdx.y][threadIdx.x] = B_start[(threadIdx.y + s)*N + threadIdx.x];

            __syncthreads();

            for(int i = 0; i < BLOCK_SIZE_X; i++)
            {
                temp += sDataA[threadIdx.y][i]*sDataB[i][threadIdx.x];
            }

            __syncthreads();
        }
        
        matrixC[tidy*N + tidx] = temp;
    }
}

void CpuCalculate(float *matrixA, float *matrixB, float *matrixC)
{
    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < N; j++)
        {
            float temp = 0;
            for(int k = 0; k < K; k++)
            {
                temp += matrixA[i*K + k]*matrixB[k*N + j];
            }
            matrixC[i*N + j] = temp;
        }
    }
}

bool checkResult(float *CpuResult, float *GpuResult)
{
    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < N; j++)
        {
            if(abs(CpuResult[i*N + j] - GpuResult[i*N + j]) > 0.05)
                return false;
        }
    }
    return true;
}

int main()
{
    // cpu alloc
    float *matrixAHost        = (float *)malloc(M*K*sizeof(float));
    float *matrixBHost        = (float *)malloc(K*N*sizeof(float));
    float *matrixResultDevice = (float *)malloc(M*N*sizeof(float));
    float *matrixResultHost   = (float *)malloc(M*N*sizeof(float));

    // cpu memset
    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < K; j++)
        {
            matrixAHost[i*K + j] = i;
        }
    }
    for(int i = 0; i < K; i++)
    {
        for(int j = 0; j < N; j++)
        {
            matrixBHost[i*N + j] = i;
        }
    }

    // cpu calculate
    CpuCalculate(matrixAHost, matrixBHost, matrixResultHost);

    // gpu alloc
    float *matrixADevice;
    float *matrixBDevice;
    float *matrixCDevice;
    cudaMalloc((void **)&matrixADevice, M*K*sizeof(float));
    cudaMalloc((void **)&matrixBDevice, K*N*sizeof(float));
    cudaMalloc((void **)&matrixCDevice, M*N*sizeof(float));

    // gpu alloc
    cudaMemcpy(matrixADevice, matrixAHost, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(matrixBDevice, matrixBHost, K*N*sizeof(float), cudaMemcpyHostToDevice);

    // gpu calculate
    dim3 Grid(ceil((float)N/BLOCK_SIZE_X), ceil((float)M/BLOCK_SIZE_Y));
    dim3 Block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    GEMM<<<Grid, Block>>>(matrixADevice, matrixBDevice, matrixCDevice);

    cudaMemcpy(matrixResultDevice, matrixCDevice, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    if(checkResult(matrixResultHost, matrixResultDevice))
        printf("the result is right\n");
    else
    {
        for(int i = 0; i < M; i++)
        {
            for(int j = 0; j < N; j++)
            {
                printf("%f ", matrixResultDevice[i*N + j]);
            }
            printf("\n");
        }
    }
}