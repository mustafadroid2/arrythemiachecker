package com.gunadarma.heartratearrhythmiachecker.service.loader;

import android.content.Context;
import android.util.Log;

public class OpenCVLoader {
    private static final String TAG = "OpenCVLoader";
    private static boolean sInitialized = false;

    public static boolean init(Context context) {
        if (sInitialized) return true;

        try {
            System.loadLibrary("c++_shared");
            Thread.sleep(50);
            loadOpenCV(context);
            sInitialized = true;
            return true;
        } catch (Exception e) {
            Log.e(TAG, "Failed to load OpenCV libraries: " + e.getMessage());
            return false;
        }
    }

    private static void loadOpenCV(Context context) {
        try {
            System.loadLibrary("opencv_java4");
            Log.d(TAG, "Successfully loaded opencv_java4 library");
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "Failed to load opencv_java4: " + e.getMessage());
            String libPath = context.getApplicationInfo().nativeLibraryDir + "/libopencv_java4.so";
            System.load(libPath);
            Log.d(TAG, "Successfully loaded OpenCV from absolute path");
        }
    }

    public static boolean isInitialized() {
        return sInitialized;
    }
}
