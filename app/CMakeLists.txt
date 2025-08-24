cmake_minimum_required(VERSION 3.22.1)
project(heartrate_opencv)

# Set build type and C++ standard
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force using c++_shared
set(CMAKE_ANDROID_STL_TYPE c++_shared)
set(BUILD_SHARED_LIBS ON)

# Set OpenCV paths
set(OpenCV_SDK_DIR "C:/Users/musta/OneDrive/Pictures/opencv-4.12.0-android-sdk/OpenCV-android-sdk")
set(OpenCV_DIR "${OpenCV_SDK_DIR}/sdk/native/jni")
set(OpenCV_CONFIG_PATH "${OpenCV_DIR}/abi-${ANDROID_ABI}")

# Add paths for finding dependencies
list(APPEND CMAKE_MODULE_PATH "${OpenCV_CONFIG_PATH}")
list(APPEND CMAKE_PREFIX_PATH "${OpenCV_CONFIG_PATH}")

# Debug output
message(STATUS "Looking for OpenCV in: ${OpenCV_CONFIG_PATH}")
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
message(STATUS "ANDROID_ABI: ${ANDROID_ABI}")

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Debug output
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")

# Create native library
add_library(native_opencv SHARED
    src/main/cpp/native_opencv.cpp)

# Include OpenCV headers
target_include_directories(native_opencv PRIVATE
    ${OpenCV_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS})

# Set library properties
set_target_properties(native_opencv PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)

# Link libraries
target_link_libraries(native_opencv
    ${OpenCV_LIBS}
    android
    log)

# Create the jniLibs directory if it doesn't exist
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI})

# Copy OpenCV native libraries
file(GLOB OPENCV_NATIVE_LIBS
    "${OpenCV_DIR}/libs/${ANDROID_ABI}/*.so"
    "${OpenCV_3RDPARTY_DIR}/libs/${ANDROID_ABI}/*.a")
foreach(lib ${OPENCV_NATIVE_LIBS})
    add_custom_command(TARGET native_opencv POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${lib} ${CMAKE_CURRENT_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/
        COMMENT "Copying OpenCV library ${lib} to jniLibs")
endforeach()

# Copy the native library to jniLibs
add_custom_command(TARGET native_opencv POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:native_opencv> ${CMAKE_CURRENT_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/
    COMMENT "Copying native_opencv library to jniLibs directory")
