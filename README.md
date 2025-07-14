# CUDA Operator Optimization Project
## 项目概述
本项目是一个专注于CUDA核心算子优化的代码库，包含矩阵乘法(GEMM)、多头注意力机制(MHA)、归约操作(Reduce)和SoftMax等常见深度学习算子的高效实现。
## 构建说明
```shell
  mkdir build && cd build
  cmake ..
  make
```
## 使用说明
* 每个算子目录包含不同优化版本的实现，文件名中的数字表示优化级别。
* 可以自行添加算子，复制相应的算子某一个版本之后，重写__global__函数即可，不用修改CMakeLists.txt，会自动构建你的.cu文件。
