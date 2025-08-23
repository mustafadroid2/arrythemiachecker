cmake_minimum_required(VERSION 3.6)

set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_ANDROID_STL_TYPE c++_shared)

include_directories(${OpenCV_DIR}/jni/include)

add_library(opencv_jni_shared SHARED dummy.cpp)
