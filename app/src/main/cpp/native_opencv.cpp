#include <jni.h>

extern "C" {
    JNIEXPORT void JNICALL
    Java_com_gunadarma_heartratearrhythmiachecker_opencv_OpenCVNative_init(JNIEnv *env, jclass clazz) {
        // Dummy implementation
    }

    JNIEXPORT jlong JNICALL
    Java_com_gunadarma_heartratearrhythmiachecker_opencv_OpenCVNative_loadImage(JNIEnv *env, jclass clazz, jstring imagePath) {
        return 0;
    }
}
