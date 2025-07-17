#pragma once

#include <cuda_runtime.h>

template <typename T>
struct SumOp {
  __device__ __forceinline__ T operator()(const T &a, const T &b) const {
    return a + b;
  }
  static constexpr T identity() { return T(0); }
};

#define WARPSIZE 32

template <typename T, typename ReduceOp>
__device__ T BlockAllReduce(T val)
{
  ReduceOp op;  
  const T identity = ReduceOp::identity();

  uint tid = threadIdx.x;
  uint warpID = tid/WARPSIZE;
  uint laneID = tid%WARPSIZE;
  unit warpSum = blockDim.x / WARPSIZE;

  __shared__ float sum_smem[32];

  for(int i = WARPSIZE >> 2; i > 0; i >>= 1)
  {
    val = op(val, __shfl_down_sync(0xffffffff, val, i));
  }

  if(laneID == 0) sum_smem[warpID] = val;

  __syncthreads();
  
  if(warpID == 0)
  {
    float val = laneID < warpSum ? sum_smem[laneID] : identity;
    for(int i = WARPSIZE >> 2; i > 0; i >>= 1)
    {
      val = op(val,__shfl_down_sync(0xffffffff, val, i));
    }
  }
  return __shfl_sync(0xFFFFFFFF, val, 0);
}