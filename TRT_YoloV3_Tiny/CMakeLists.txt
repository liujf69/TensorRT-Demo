cmake_minimum_required(VERSION 3.14)
project(yolov3-tiny)
add_definitions(-std=c++11)
enable_language(CUDA)

# TensorRT
include_directories(/home/liujinfu/Downloads/TensorRT-8.2.5.1/include/)
link_directories(/home/liujinfu/Downloads/TensorRT-8.2.5.1/lib/)

# Cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

add_library(yololayer SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu)
target_link_libraries(yololayer nvinfer cudart)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS} ./include)

add_executable(yolov3-tiny_demo ${PROJECT_SOURCE_DIR}/yolov3-tiny_Demo.cpp)
target_link_libraries(yolov3-tiny_demo ${OpenCV_LIBS} nvinfer cudart yololayer)