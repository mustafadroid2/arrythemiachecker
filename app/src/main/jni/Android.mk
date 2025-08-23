LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE := opencv_java4
LOCAL_SRC_FILES := ../jniLibs/$(TARGET_ARCH_ABI)/libopencv_java4.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := cpufeatures
LOCAL_SRC_FILES := $(ANDROID_NDK)/sources/android/cpufeatures/cpu-features.c
include $(BUILD_STATIC_LIBRARY)
