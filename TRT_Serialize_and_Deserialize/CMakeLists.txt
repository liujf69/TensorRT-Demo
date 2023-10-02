cmake_minimum_required(VERSION 3.14) 
project(mlp)
set(CMAKE_CXX_STANDARD 14)

# TensorRT
include_directories(/home/liujinfu/Downloads/TensorRT-8.2.5.1/include/)
link_directories(/home/liujinfu/Downloads/TensorRT-8.2.5.1/lib/)

# Cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

add_executable(Serialize_onnx Serialize_onnx.cpp)
target_link_libraries(Serialize_onnx nvinfer cudart nvonnxparser)
add_definitions(-O2 -pthread)

add_executable(Deserialize Deserialize.cpp)
target_link_libraries(Deserialize nvinfer cudart)
add_definitions(-O2 -pthread)

