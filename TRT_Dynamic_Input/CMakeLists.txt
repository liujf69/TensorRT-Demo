cmake_minimum_required(VERSION 3.14) 
project(infer_onnx)
set(CMAKE_CXX_STANDARD 14)

# TensorRT
include_directories(/home/liujinfu/Downloads/TensorRT-8.2.5.1/include/)
link_directories(/home/liujinfu/Downloads/TensorRT-8.2.5.1/lib/)

# Cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

add_executable(Infer_Onnx Infer_Onnx.cpp)
target_link_libraries(Infer_Onnx nvinfer cudart nvonnxparser)
add_definitions(-O2 -pthread)