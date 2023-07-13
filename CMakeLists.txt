cmake_minimum_required(VERSION 3.19)
project(yolov7_and_tensorrt-cpp-inference)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wno-write-strings -Wno-deprecated-declarations")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# TensorRT
set(TensorRT_DIR /home/user/distr/tensorRT/TensorRT/)

find_package(Threads)

# CUDA
find_package(CUDA REQUIRED)
message(STATUS "Find CUDA include at ${CUDA_INCLUDE_DIRS}")
message(STATUS "Find CUDA libraries: ${CUDA_LIBRARIES}")

# OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "Find OpenCV include at ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Find OpenCV libraries: ${OpenCV_LIBRARIES}")

add_executable(yolov7_and_tensorrt-cpp-inference
        tensorrt_cpp_inference/logging.h
        tensorrt_cpp_inference/yolov7.cpp
        tensorrt_cpp_inference/yolov7.h)

target_include_directories(yolov7_and_tensorrt-cpp-inference PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_DIR}/include
        ${TensorRT_DIR}/samples/common)
target_link_libraries(yolov7_and_tensorrt-cpp-inference PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT}
        /home/user/distr/tensorRT/TensorRT/build/libnvonnxparser.so
        /home/user/distr/tensorRT/TensorRT/build/libnvinfer.so
        /home/user/distr/tensorRT/TensorRT/build/libnvinfer_plugin.so)
