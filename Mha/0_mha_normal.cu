// file: mha_cublas.cu
#include <iostream>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call)                                        \
    if ((call) != cudaSuccess)                                  \
    {                                                           \
        std::cerr << "CUDA error at " << __LINE__ << std::endl; \
        exit(1);                                                \
    }

#define CHECK_CUBLAS(call)                                        \
    if ((call) != CUBLAS_STATUS_SUCCESS)                          \
    {                                                             \
        std::cerr << "cuBLAS error at " << __LINE__ << std::endl; \
        exit(1);                                                  \
    }

const int BATCH = 1;
const int SEQ_LEN = 4;
const int EMBED_DIM = 8;
const int HEAD_NUM = 2;
const int HEAD_DIM = EMBED_DIM / HEAD_NUM;

__global__ void softmax_kernel(float *scores, int batch, int heads, int seq_len)
{
    int b = blockIdx.x;
    int h = threadIdx.y;

    float *s = scores + (b * heads + h) * seq_len * seq_len;

    for (int i = 0; i < seq_len; ++i)
    {
        float max_val = -1e9f;
        for (int j = 0; j < seq_len; ++j)
            max_val = fmaxf(max_val, s[i * seq_len + j]);

        float sum = 0.f;
        for (int j = 0; j < seq_len; ++j)
        {
            s[i * seq_len + j] = expf((s[i * seq_len + j] - max_val) / sqrtf((float)HEAD_DIM));
            sum += s[i * seq_len + j];
        }

        for (int j = 0; j < seq_len; ++j)
            s[i * seq_len + j] /= sum;
    }
}

void run_mha()
{
    int qkv_size = BATCH * SEQ_LEN * EMBED_DIM;
    int per_head_size = BATCH * SEQ_LEN * HEAD_DIM;

    float *h_Q = new float[qkv_size];
    float *h_K = new float[qkv_size];
    float *h_V = new float[qkv_size];
    for (int i = 0; i < qkv_size; ++i)
    {
        h_Q[i] = 0.01f * i;
        h_K[i] = 0.02f * i;
        h_V[i] = 0.03f * i;
    }

    float *d_Q, *d_K, *d_V;
    CHECK_CUDA(cudaMalloc(&d_Q, qkv_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_K, qkv_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_V, qkv_size * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_Q, h_Q, qkv_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K, qkv_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V, qkv_size * sizeof(float), cudaMemcpyHostToDevice));

    // 每个头的输出
    float *d_scores;   // (B x H x S x S)
    float *d_attn_out; // (B x H x S x D_H)
    float *d_output;   // (B x S x D)
    CHECK_CUDA(cudaMalloc(&d_scores, BATCH * HEAD_NUM * SEQ_LEN * SEQ_LEN * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_attn_out, BATCH * HEAD_NUM * SEQ_LEN * HEAD_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, qkv_size * sizeof(float)));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    for (int h = 0; h < HEAD_NUM; ++h)
    {
        float *Qh = d_Q + h * HEAD_DIM;
        float *Kh = d_K + h * HEAD_DIM;
        float *Vh = d_V + h * HEAD_DIM;
        float *score = d_scores + h * SEQ_LEN * SEQ_LEN;
        float *attn_out = d_attn_out + h * SEQ_LEN * HEAD_DIM;

        const float alpha = 1.0f, beta = 0.0f;

        // Q x K? => score[S x S]
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                 SEQ_LEN, SEQ_LEN, HEAD_DIM, &alpha,
                                 Qh, SEQ_LEN,
                                 Kh, SEQ_LEN,
                                 &beta,
                                 score, SEQ_LEN));

        // softmax
        softmax_kernel<<<1, dim3(1, HEAD_NUM)>>>(d_scores, BATCH, HEAD_NUM, SEQ_LEN);

        // score x V => attn_out[S x D_H]
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 SEQ_LEN, HEAD_DIM, SEQ_LEN, &alpha,
                                 score, SEQ_LEN,
                                 Vh, SEQ_LEN,
                                 &beta,
                                 attn_out, SEQ_LEN));
    }

    // concat heads
    CHECK_CUDA(cudaMemcpy(d_output, d_attn_out, qkv_size * sizeof(float), cudaMemcpyDeviceToDevice));

    float *h_out = new float[qkv_size];
    CHECK_CUDA(cudaMemcpy(h_out, d_output, qkv_size * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "[Attention Output]\n";
    for (int i = 0; i < SEQ_LEN; ++i)
    {
        for (int j = 0; j < EMBED_DIM; ++j)
        {
            std::cout << h_out[i * EMBED_DIM + j] << " ";
        }
        std::cout << "\n";
    }

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_scores);
    cudaFree(d_attn_out);
    cudaFree(d_output);
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_out;
}

int main()
{
    run_mha();
    return 0;
}
