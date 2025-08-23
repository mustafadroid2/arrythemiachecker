package com.gunadarma.heartratearrhythmiachecker.service;

import android.hardware.Camera;

import com.gunadarma.heartratearrhythmiachecker.model.RecordEntry;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import java.util.ArrayList;
import java.util.List;

public class MediaProcessingServiceImpl implements MediaProcessingService {

    private final String DEFAULT_DIRECTORY = "data";

    public MediaProcessingServiceImpl() {
        // Constructor without OpenCV initialization
    }

    public void createVideo(Camera camera, String recordId) {
        // TODO: Implement video recording logic
    }

    public void delete(String videoId) {
        // TODO: Implement video deletion logic
    }

    @Override
    public void createHeartBeatsVideo(Long id) {
        String videoPath = String.format("/data/%d/original.mp4", id);
        // 1. Extract heartbeats from video
        List<Long> heartbeatTimestamps = extractHeartbeatTimestamps(videoPath);
        // 2. Analyze arrhythmia
        RecordEntry.Status status = analyzeArrhythmia(heartbeatTimestamps);
        // 3. Create heartbeats image
        createHeartBeatsImage(heartbeatTimestamps, id);
        // 4. Overlay results on video (optional, stub for now)
        overlayResultsOnVideo(videoPath, heartbeatTimestamps, status);
        // 5. Update record status in database (stub for now)
        updateRecordStatus(id, status, heartbeatTimestamps);
    }

    // TODO check
    private List<HeartRateData> extractHeartRateFromVideo(String videoPath) {
        VideoCapture cap = new VideoCapture(videoPath);
        if (!cap.isOpened()) {
            throw new RuntimeException("Failed to open video file: " + videoPath);
        }

        // Define cropping region
        int x = 800, y = 200, w = 100, h = 100;
        Rect roi = new Rect(x, y, w, h);

        // Initialize heart rate data
        int heartbeatCount = 128;
        List<Double> heartbeatValues = new ArrayList<>();
        List<Long> heartbeatTimes = new ArrayList<>();

        for (int i = 0; i < heartbeatCount; i++) {
            heartbeatValues.add(0.0);
            heartbeatTimes.add(System.currentTimeMillis());
        }

        Mat frame = new Mat();
        Mat grayFrame = new Mat();
        Mat croppedFrame = new Mat();

        List<HeartRateData> heartRateDataList = new ArrayList<>();

        while (cap.read(frame)) {
            // Convert to grayscale
            Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);

            // Crop the region of interest
            croppedFrame = new Mat(grayFrame, roi);

            // Calculate average pixel intensity
            double averageIntensity = Core.mean(croppedFrame).val[0];

            // Update heart rate data
            heartbeatValues.remove(0);
            heartbeatValues.add(averageIntensity);
            heartbeatTimes.remove(0);
            heartbeatTimes.add(System.currentTimeMillis());

            // Store heart rate data
            heartRateDataList.add(new HeartRateData(averageIntensity, System.currentTimeMillis()));
        }

        cap.release();
        return heartRateDataList;
    }


    /**
     * Extracts heartbeat timestamps from the video using rPPG or placeholder logic.
     */
    private List<Long> extractHeartbeatTimestamps(String videoPath) {
        // Placeholder: simulate heartbeat detection
        List<Long> heartbeats = new ArrayList<>();
        long start = System.currentTimeMillis();
        for (int i = 0; i < 30; i++) {
            heartbeats.add(start + i * 1000); // 1 beat per second
        }
        return heartbeats;
    }

    /**
     * Analyzes heartbeat intervals to detect arrhythmia.
     */
    private RecordEntry.Status analyzeArrhythmia(List<Long> heartbeats) {
        if (heartbeats == null || heartbeats.size() < 2) return RecordEntry.Status.UNCHECKED;
        List<Long> intervals = new ArrayList<>();
        for (int i = 1; i < heartbeats.size(); i++) {
            intervals.add(heartbeats.get(i) - heartbeats.get(i - 1));
        }
        double avg = intervals.stream().mapToLong(l -> l).average().orElse(0);
        if (avg < 600) return RecordEntry.Status.ARRHYTHMIA_TACHYCARDIA;
        if (avg > 1200) return RecordEntry.Status.ARRHYTHMIA_BRADYCARDIA;
        return RecordEntry.Status.NORMAL;
    }

    /**
     * Overlays results on the video (stub for now).
     */
    void overlayResultsOnVideo(String videoPath, List<Long> heartbeats, RecordEntry.Status status) {
        // TODO: Implement overlay logic if needed
    }

    /**
     * Updates the record status in the database (stub for now).
     */
    void updateRecordStatus(Long id, RecordEntry.Status status, List<Long> heartbeats) {
        // TODO: Implement DB update logic
    }

    private void createHeartBeatsImage(List<Long> heartbeats, Long id) {
        // create image
        // stored in /data/{id}/heartbeats.jpeg
    }

    // Helper class to store heart rate data
    public static class HeartRateData {
        private final double intensity;
        private final long timestamp;

        public HeartRateData(double intensity, long timestamp) {
            this.intensity = intensity;
            this.timestamp = timestamp;
        }

        public double getIntensity() {
            return intensity;
        }

        public long getTimestamp() {
            return timestamp;
        }
    }
}
