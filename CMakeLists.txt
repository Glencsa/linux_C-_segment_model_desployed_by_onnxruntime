cmake_minimum_required(VERSION 3.26)

project(hrnet)


# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ONNXRUNTIME_DIR "/opt/onnxruntime-linux-x64-1.17.3")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib")

include_directories(${ONNXRUNTIME_INCLUDE_DIR})


# find ONNX Runtime 
find_library(ONNXRUNTIME_LIBRARY onnxruntime PATHS ${ONNXRUNTIME_LIB_DIR})


# Find OpenCV package
find_package(OpenCV REQUIRED)

# Add the include directories for OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# 设置项目源文件
add_executable(hrnet main.cpp)


# 链接 ONNX Runtime 库
target_link_libraries(hrnet ${ONNXRUNTIME_LIBRARY})

# Link the OpenCV libraries to your target
target_link_libraries(hrnet ${OpenCV_LIBS})


target_sources(hrnet PRIVATE main.cpp hrnet.cpp)