package com.gunadarma.heartratearrhythmiachecker.opencv;

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
            System.loadLibrary("opencv_java4");
            sInitialized = true;
            Log.d(TAG, "OpenCV library loaded successfully");
            return true;
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "Failed to load OpenCV library: " + e.getMessage());
            return false;
        }
    }

    public static boolean isInitialized() {
        return sInitialized;
    }
}
