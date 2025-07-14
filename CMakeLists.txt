cmake_minimum_required(VERSION 3.20)

project(ReduceProject CXX CUDA)

add_subdirectory(Reduce)
add_subdirectory(Gemm)
add_subdirectory(Mha)

add_subdirectory(SoftMax)