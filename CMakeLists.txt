cmake_minimum_required(VERSION 3.20)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
project(ReduceProject CXX CUDA)

add_subdirectory(Reduce)
add_subdirectory(Gemm)
add_subdirectory(Mha)

add_subdirectory(SoftMax)