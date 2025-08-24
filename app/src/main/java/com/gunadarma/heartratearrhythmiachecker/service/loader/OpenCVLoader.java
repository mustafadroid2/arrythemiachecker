package com.gunadarma.heartratearrhythmiachecker.service.loader;

import android.content.Context;
import android.os.Build;
import android.util.Log;

public class OpenCVLoader {
    private static final String TAG = "OpenCVLoader";
    private static boolean sInitialized = false;

    public static boolean init(Context context) {
        if (sInitialized) {
            return true;
        }

        try {
            // Load the C++ shared library first
            System.loadLibrary("c++_shared");
            Thread.sleep(100); // Give a small delay to ensure library is fully loaded

            // Then load OpenCV with full path specification
            String[] abis = Build.SUPPORTED_ABIS;
            String loadError = "";

            // Try to load OpenCV
            try {
                Log.d(TAG, "Trying to load opencv_java4 library");
                System.loadLibrary("opencv_java4");
                Log.d(TAG, "Successfully loaded opencv_java4 library");
            } catch (UnsatisfiedLinkError e) {
                Log.e(TAG, "Failed to load opencv_java4: " + e.getMessage());
                loadError = e.getMessage();
                // Try loading with absolute path
                for (String abi : abis) {
                    try {
                        String libPath = context.getApplicationInfo().nativeLibraryDir + "/libopencv_java4.so";
                        Log.d(TAG, "Trying to load OpenCV from: " + libPath);
                        System.load(libPath);
                        Log.d(TAG, "Successfully loaded OpenCV from: " + libPath);
                        loadError = ""; // Clear error if successful
                        break;
                    } catch (UnsatisfiedLinkError e1) {
                        Log.e(TAG, "Failed to load from " + context.getApplicationInfo().nativeLibraryDir + ": " + e1.getMessage());
                        loadError = e1.getMessage();
                    }
                }
            }

            if (loadError.isEmpty()) {
                sInitialized = true;
                Log.d(TAG, "OpenCV libraries loaded successfully");
                return true;
            } else {
                Log.e(TAG, "Final error loading OpenCV: " + loadError);
                throw new UnsatisfiedLinkError(loadError);
            }
        } catch (Exception e) {
            Log.e(TAG, "Failed to load OpenCV libraries: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }

    public static boolean isInitialized() {
        return sInitialized;
    }
}
