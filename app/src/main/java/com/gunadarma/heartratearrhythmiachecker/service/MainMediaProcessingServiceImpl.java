package com.gunadarma.heartratearrhythmiachecker.service;

import android.util.Log;

import com.gunadarma.heartratearrhythmiachecker.constant.AppConstant;
import com.gunadarma.heartratearrhythmiachecker.model.RPPGData;
import com.gunadarma.heartratearrhythmiachecker.model.RecordEntry;
import com.gunadarma.heartratearrhythmiachecker.service.mediacreator.ImageGeneratorServiceImpl;
import com.gunadarma.heartratearrhythmiachecker.service.mediacreator.VideoGeneratorServiceImpl;
import com.gunadarma.heartratearrhythmiachecker.service.rppg.RPPGHandPalmServiceImpl;
import com.gunadarma.heartratearrhythmiachecker.service.rppg.RPPGHeadFaceServiceImpl;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.VideoWriter;
import org.opencv.videoio.Videoio;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Optional;

public class MainMediaProcessingServiceImpl implements MainMediaProcessingService {
    public enum DetectionMode {
        FACE,
        HAND
    }

    private final android.content.Context context;
    private final CascadeClassifier faceDetector;
    private final DetectionMode currentMode;
    private final ImageGeneratorServiceImpl imageGeneratorService;
    private final VideoGeneratorServiceImpl videoGeneratorService;
    private final RPPGHeadFaceServiceImpl rppgHeadFaceService;
    private final RPPGHandPalmServiceImpl rppgHandPalmService;
    private static final String TAG = "MediaProcessingService";

    public MainMediaProcessingServiceImpl(android.content.Context context) {
        this.context = context;
        this.faceDetector = initializeFaceDetector();
        this.currentMode = DetectionMode.HAND; // Default to hand detection

        this.imageGeneratorService = new ImageGeneratorServiceImpl(context);
        this.videoGeneratorService = new VideoGeneratorServiceImpl(context);
        this.rppgHeadFaceService = new RPPGHeadFaceServiceImpl(context);
        this.rppgHandPalmService = new RPPGHandPalmServiceImpl(context);
    }

    private CascadeClassifier initializeFaceDetector() {
        try {
            // Create a temporary file to store the cascade classifier
            File cascadeDir = context.getDir("cascade", android.content.Context.MODE_PRIVATE);
            cascadeDir.mkdirs();
            File cascadeFile = new File(cascadeDir, "haarcascade_frontalface_alt.xml");

            // Copy the cascade file from assets to the temporary file
            InputStream is = context.getAssets().open("haarcascades/haarcascade_frontalface_alt.xml");
            FileOutputStream os = new FileOutputStream(cascadeFile);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            // Create the cascade classifier
            CascadeClassifier detector = new CascadeClassifier(cascadeFile.getAbsolutePath());

            // Delete the temporary file
            cascadeFile.delete();
            cascadeDir.delete();

            if (detector.empty()) {
                Log.e(TAG, "Failed to load cascade classifier");
                return null;
            }
            return detector;
        } catch (Exception e) {
            Log.e(TAG, "Error initializing face detector", e);
            return null;
        }
    }

    @Override
    public void createHeartBeatsVideo(RecordEntry recordEntry) {
        // Get the proper file path using context
        File videoFile = new File(
          context.getExternalFilesDir(null),
          String.format("%s/%s/%s", AppConstant.DATA_DIR, recordEntry.getId(), AppConstant.ORIGINAL_VIDEO_NAME)
        );
        String videoPath = videoFile.getAbsolutePath();

        // 1. Extract heartbeats based on current detection mode
        List<Long> heartRateData;
        int duration = 0;
        if (currentMode == DetectionMode.HAND) {
            RPPGData rppgData = rppgHandPalmService.getRPPGSignals(videoPath);
            heartRateData = rppgData.getHeartbeats();
            duration = rppgData.getDurationSeconds();
        } else {
            RPPGData rppgData = rppgHeadFaceService.getRPPGSignals(videoPath);
            heartRateData = rppgData.getHeartbeats();
            duration = rppgData.getDurationSeconds();
        }

        // 2. Analyze for arrhythmia
        RecordEntry.Status status = analyzeArrhythmia(heartRateData);

        // 3. Create heartbeats visualization
        imageGeneratorService.createHeartBeatsImage(heartRateData, recordEntry.getId());

        // 4. Update record status
        updateRecordStatus(recordEntry, status, heartRateData, duration);
    }

    private RecordEntry.Status analyzeArrhythmia(List<Long> heartbeats) {
        if (heartbeats == null || heartbeats.size() < 2) {
            return RecordEntry.Status.UNCHECKED;
        }

        List<Long> intervals = new ArrayList<>();
        for (int i = 1; i < heartbeats.size(); i++) {
            intervals.add(heartbeats.get(i) - heartbeats.get(i - 1));
        }

        // Calculate mean RR interval
        double meanRR = intervals.stream().mapToLong(l -> l).average().orElse(0);

        // Calculate standard deviation of RR intervals (SDNN)
        double sdnn = calculateSDNN(intervals, meanRR);

        // Calculate heart rate
        double heartRate = 60000.0 / meanRR; // Convert to BPM

        // Classify based on heart rate and variability
        if (heartRate > 100) {
            return RecordEntry.Status.ARRHYTHMIA_TACHYCARDIA;
        } else if (heartRate < 60) {
            return RecordEntry.Status.ARRHYTHMIA_BRADYCARDIA;
        } else if (sdnn > 100) // High variability might indicate arrhythmia
        {
            return RecordEntry.Status.ARRHYTHMIA_IRREGULAR;
        }

        return RecordEntry.Status.NORMAL;
    }

    private double calculateSDNN(List<Long> intervals, double mean) {
        return Math.sqrt(
          intervals.stream()
            .mapToDouble(interval -> Math.pow(interval - mean, 2))
            .average()
            .orElse(0.0)
        );
    }

    private void updateRecordStatus(RecordEntry recordEntry, RecordEntry.Status status, List<Long> heartbeats, int duration) {
        double averageInterval = 0;
        if (heartbeats.size() > 2) {
            // Calculate average heart rate
            for (int i = 1; i < heartbeats.size(); i++) {
                averageInterval += (heartbeats.get(i) - heartbeats.get(i - 1));
            }
            averageInterval /= (heartbeats.size() - 1);
        }

        int bpm = averageInterval >= 0 ? 0 : (int) (60000.0 / averageInterval);

        // Update database record using context
        recordEntry.setStatus(status);
        recordEntry.setDuration(duration);
        recordEntry.setBeatsPerMinute(bpm);
        recordEntry.setHeartbeats(Optional.of(heartbeats).orElse(List.of()));
        recordEntry.setUpdatedAt(System.currentTimeMillis());
        System.out.println("success updateRecordStatus: " + recordEntry);
    }
}

