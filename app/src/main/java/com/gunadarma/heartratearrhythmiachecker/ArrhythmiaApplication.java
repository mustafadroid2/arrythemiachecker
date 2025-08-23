package com.gunadarma.heartratearrhythmiachecker;

import android.widget.Toast;
import androidx.multidex.MultiDexApplication;
import com.gunadarma.heartratearrhythmiachecker.opencv.OpenCVLoader;

public class ArrhythmiaApplication extends MultiDexApplication {
    @Override
    public void onCreate() {
        super.onCreate();

        if (!OpenCVLoader.init(this)) {
            Toast.makeText(getApplicationContext(),
                    "Failed to load OpenCV library. Some features may not work.",
                    Toast.LENGTH_LONG).show();
        }
    }
}
