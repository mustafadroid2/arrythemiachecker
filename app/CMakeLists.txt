cmake_minimum_required(VERSION 3.22.1)
project(heartrate_opencv)

# Set build type and C++ standard
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force using c++_shared
set(CMAKE_ANDROID_STL_TYPE c++_shared)
set(BUILD_SHARED_LIBS ON)

# Create a dummy library just to trigger c++_shared.so generation
add_library(c++_shared SHARED
    src/main/cpp/native_opencv.cpp)

# Set library properties
set_target_properties(c++_shared PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)

# Link only necessary system libraries
target_link_libraries(c++_shared
    android
    log)

# Create the jniLibs directory if it doesn't exist
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI})

# Ensure c++_shared.so is copied to the correct location
add_custom_command(TARGET c++_shared POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:c++_shared>
        ${CMAKE_CURRENT_SOURCE_DIR}/src/main/jniLibs/${ANDROID_ABI}/

    COMMENT "Copying OpenCV library to jniLibs directory")
