cmake_minimum_required(VERSION 3.14) 
project(mlp)
set(CMAKE_CXX_STANDARD 14)

# TensorRT
include_directories(/home/liujinfu/Downloads/TensorRT-8.2.5.1/include/)
link_directories(/home/liujinfu/Downloads/TensorRT-8.2.5.1/lib/)

# Cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

add_executable(vgg_demo VGG_Demo.cpp)
target_link_libraries(vgg_demo nvinfer cudart)

add_definitions(-O2 -pthread)