package com.gunadarma.heartratearrhythmiachecker.service.loader;

import android.content.Context;
import android.util.Log;

public class OpenCVLoader {
    private static final String TAG = "OpenCVLoader";
    private static boolean sInitialized = false;

    public static boolean init(Context context) {
        if (sInitialized) {
            return true;
        }

        try {
            // Load necessary libraries in the correct order
            System.loadLibrary("opencv_java4");
            // System.loadLibrary("native-lib"); // OpenCV native library for Android
            sInitialized = true;
            Log.d(TAG, "OpenCV libraries loaded successfully");
            return true;
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "Failed to load OpenCV libraries: " + e.getMessage());
            return false;
        }
    }

    public static boolean isInitialized() {
        return sInitialized;
    }
}
