package com.gunadarma.heartratearrhythmiachecker.service;

import android.graphics.Bitmap;
import android.hardware.Camera;
import android.util.Log;

import com.gunadarma.heartratearrhythmiachecker.constant.AppConstant;
import com.gunadarma.heartratearrhythmiachecker.model.RecordEntry;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class MediaProcessingServiceImpl implements MediaProcessingService {
    private final android.content.Context context;

    public MediaProcessingServiceImpl(android.content.Context context) {
        this.context = context;
    }

    private final String DEFAULT_DIRECTORY = "data";
    private static final String TAG = "MediaProcessingService";
    private static final double FPS = 30.0; // Assuming 30 fps video
    private static final double SECONDS_PER_WINDOW = 1.5; // Window size for analysis
    private static final int WINDOW_SIZE = (int)(FPS * SECONDS_PER_WINDOW);
    private static final double MIN_HR_BPM = 40.0; // Minimum heart rate in BPM
    private static final double MAX_HR_BPM = 240.0; // Maximum heart rate in BPM

    @Override
    public void createHeartBeatsVideo(Long id) {
        // Get the proper file path using context
        File videoFile = new File(
            context.getExternalFilesDir(null),
            String.format("%s/%s/%s", AppConstant.DATA_DIR, id, AppConstant.ORIGINAL_VIDEO_NAME)
        );
        String videoPath = videoFile.getAbsolutePath();

        // 1. Extract heartbeats using rPPG
        List<HeartRateData> heartRateData = extractHeartRateFromVideo(videoPath);
        List<Long> heartbeatTimestamps = processHeartRateData(heartRateData);

        // 2. Analyze for arrhythmia
        RecordEntry.Status status = analyzeArrhythmia(heartbeatTimestamps);

        // 3. Create visualization
        createHeartBeatsImage(heartbeatTimestamps, id);

        // 4. Update record status
        updateRecordStatus(id, status, heartbeatTimestamps);
    }

    private List<HeartRateData> extractHeartRateFromVideo(String videoPath) {
        Log.i(TAG, "Processing video: " + videoPath);
        VideoCapture cap = new VideoCapture(videoPath);
        if (!cap.isOpened()) {
            Log.e(TAG, "Failed to open video file: " + videoPath);
            return new ArrayList<>();
        }

        // Get video properties
        double fps = cap.get(Videoio.CAP_PROP_FPS);
        int frameCount = (int) cap.get(Videoio.CAP_PROP_FRAME_COUNT);

        // Lists to store RGB signals
        List<Double> redChannel = new ArrayList<>();
        List<Double> greenChannel = new ArrayList<>();
        List<Double> blueChannel = new ArrayList<>();
        List<Long> timestamps = new ArrayList<>();

        Mat frame = new Mat();
        Mat resized = new Mat();
        Mat rgb = new Mat();

        long startTime = System.currentTimeMillis();
        int frameIndex = 0;

        while (cap.read(frame)) {
            if (frame.empty()) continue;

            // Resize for faster processing
            Imgproc.resize(frame, resized, new Size(320, 240));

            // Convert to RGB
            Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_BGR2RGB);

            // Calculate mean color values
            Scalar means = Core.mean(rgb);

            // Store RGB values
            redChannel.add(means.val[0]);
            greenChannel.add(means.val[1]);
            blueChannel.add(means.val[2]);

            // Calculate timestamp for this frame
            long timestamp = startTime + (long)(frameIndex * (1000.0 / fps));
            timestamps.add(timestamp);

            frameIndex++;
        }

        cap.release();

        // Process the signals
        return processRPPGSignals(redChannel, greenChannel, blueChannel, timestamps, fps);
    }

    private List<HeartRateData> processRPPGSignals(List<Double> red, List<Double> green, List<Double> blue,
                                                  List<Long> timestamps, double fps) {
        List<HeartRateData> heartRateDataList = new ArrayList<>();

        // Convert lists to OpenCV Mats for signal processing
        Mat redMat = new Mat(red.size(), 1, CvType.CV_64F);
        Mat greenMat = new Mat(green.size(), 1, CvType.CV_64F);
        Mat blueMat = new Mat(blue.size(), 1, CvType.CV_64F);

        for (int i = 0; i < red.size(); i++) {
            redMat.put(i, 0, red.get(i));
            greenMat.put(i, 0, green.get(i));
            blueMat.put(i, 0, blue.get(i));
        }

        // Detrend and normalize signals
        Mat redDetrended = detrendSignal(redMat);
        Mat greenDetrended = detrendSignal(greenMat);
        Mat blueDetrended = detrendSignal(blueMat);

        // PCA color space transform
        List<Mat> channels = new ArrayList<>();
        channels.add(redDetrended);
        channels.add(greenDetrended);
        channels.add(blueDetrended);
        Mat signals = new Mat();
        Core.vconcat(channels, signals);

        Mat covar = new Mat();
        Mat mean = new Mat();
        Core.calcCovarMatrix(signals, covar, mean, Core.COVAR_NORMAL | Core.COVAR_ROWS);

        Mat eigenvalues = new Mat();
        Mat eigenvectors = new Mat();
        Core.eigen(covar, eigenvalues, eigenvectors);

        // Project signals onto first principal component
        Mat projected = new Mat();
        Core.gemm(signals, eigenvectors.row(0).t(), 1.0, new Mat(), 0.0, projected);

        // Apply bandpass filter (0.7 Hz - 4.0 Hz for heart rate range 42-240 BPM)
        Mat filteredSignal = applyBandpassFilter(projected, fps);

        // Process signal in windows
        int windowSize = (int)(SECONDS_PER_WINDOW * fps);
        int stepSize = windowSize / 2; // 50% overlap

        for (int i = 0; i < filteredSignal.rows() - windowSize; i += stepSize) {
            Mat window = filteredSignal.rowRange(i, i + windowSize);

            // Find peaks in the window
            List<Integer> peakIndices = findPeaks(window);

            if (!peakIndices.isEmpty()) {
                // Calculate heart rate from peak intervals
                double averageInterval = (double)windowSize / peakIndices.size();
                double heartRate = 60.0 * fps / averageInterval;

                // Only add valid heart rates
                if (heartRate >= MIN_HR_BPM && heartRate <= MAX_HR_BPM) {
                    // Convert window peaks to actual timestamps
                    for (Integer peakIndex : peakIndices) {
                        int globalIndex = i + peakIndex;
                        if (globalIndex < timestamps.size()) {
                            heartRateDataList.add(new HeartRateData(heartRate, timestamps.get(globalIndex)));
                        }
                    }
                }
            }
        }

        return heartRateDataList;
    }

    private List<Integer> findPeaks(Mat signal) {
        List<Integer> peaks = new ArrayList<>();
        double[] prev = new double[1];
        double[] curr = new double[1];
        double[] next = new double[1];

        // Calculate the standard deviation for adaptive thresholding
        MatOfDouble stdDev = new MatOfDouble();
        MatOfDouble mean = new MatOfDouble();
        Core.meanStdDev(signal, mean, stdDev);
        double threshold = stdDev.get(0, 0)[0] * 0.6; // Adaptive threshold

        double lastPeakValue = Double.MIN_VALUE;
        int minPeakDistance = (int)(0.3 * FPS); // Minimum 0.3 seconds between peaks
        int lastPeakIndex = -minPeakDistance;

        for (int i = 1; i < signal.rows() - 1; i++) {
            signal.get(i-1, 0, prev);
            signal.get(i, 0, curr);
            signal.get(i+1, 0, next);

            if (curr[0] > prev[0] && curr[0] > next[0] && // Local maximum
                curr[0] > mean.get(0, 0)[0] + threshold && // Above threshold
                i - lastPeakIndex >= minPeakDistance) { // Minimum distance

                if (curr[0] > lastPeakValue || i - lastPeakIndex >= minPeakDistance * 2) {
                    peaks.add(i);
                    lastPeakValue = curr[0];
                    lastPeakIndex = i;
                }
            }
        }

        return peaks;
    }

    private Mat detrendSignal(Mat signal) {
        Mat detrended = new Mat();
        signal.copyTo(detrended);

        // Apply moving average to remove trends
        int kernelSize = (int)(FPS * 2); // 2-second window
        if (kernelSize % 2 == 0) kernelSize++; // Make odd for centered window

        // Create and normalize kernel
        Mat kernel = Mat.ones(kernelSize, 1, CvType.CV_64F);
        Core.multiply(kernel, new Scalar(1.0/kernelSize), kernel);

        // Apply detrending filter
        Mat trend = new Mat();
        Imgproc.filter2D(signal, trend, -1, kernel, new Point(-1, -1), 0, Core.BORDER_REFLECT);
        Core.subtract(signal, trend, detrended);

        // Normalize the signal
        Core.normalize(detrended, detrended, 0, 1, Core.NORM_MINMAX);

        return detrended;
    }

    private Mat applyBandpassFilter(Mat signal, double fps) {
        Mat paddedSignal = new Mat();
        int m = Core.getOptimalDFTSize(signal.rows());
        Core.copyMakeBorder(signal, paddedSignal, 0, m - signal.rows(), 0, 0, Core.BORDER_CONSTANT, Scalar.all(0));

        // Convert to frequency domain
        Mat complexSignal = new Mat();
        Core.dft(paddedSignal, complexSignal, Core.DFT_COMPLEX_OUTPUT);

        // Create bandpass filter
        Mat filter = new Mat(complexSignal.rows(), 2, CvType.CV_64F, new Scalar(0));
        double lowFreq = MIN_HR_BPM / 60.0; // 0.7 Hz
        double highFreq = MAX_HR_BPM / 60.0; // 4.0 Hz

        int lowBin = (int)Math.round(lowFreq * m / fps);
        int highBin = (int)Math.round(highFreq * m / fps);

        // Set passband
        for (int i = lowBin; i <= highBin && i < filter.rows()/2; i++) {
            filter.put(i, 0, 1.0);
            filter.put(i, 1, 0.0);
            if (i != 0) {
                filter.put(filter.rows()-i, 0, 1.0);
                filter.put(filter.rows()-i, 1, 0.0);
            }
        }

        // Apply filter
        Core.mulSpectrums(complexSignal, filter, complexSignal, 0);

        // Convert back to time domain
        Mat filteredSignal = new Mat();
        Core.idft(complexSignal, filteredSignal, Core.DFT_REAL_OUTPUT | Core.DFT_SCALE);

        // Return only the original signal length
        return filteredSignal.rowRange(0, signal.rows());
    }

    private List<Long> processHeartRateData(List<HeartRateData> heartRateData) {
        List<Long> heartbeatTimestamps = new ArrayList<>();

        if (heartRateData.isEmpty()) {
            return heartbeatTimestamps;
        }

        // Convert heart rate data to beat timestamps
        double averageHR = heartRateData.stream()
                .mapToDouble(HeartRateData::getIntensity)
                .average()
                .orElse(60.0);

        // Calculate average interval between beats
        double averageIntervalMs = 60000.0 / averageHR;

        // Generate timestamps
        long startTime = heartRateData.get(0).getTimestamp();
        long currentTime = startTime;

        while (currentTime <= heartRateData.get(heartRateData.size()-1).getTimestamp()) {
            heartbeatTimestamps.add(currentTime);
            currentTime += (long)averageIntervalMs;
        }

        return heartbeatTimestamps;
    }

    private RecordEntry.Status analyzeArrhythmia(List<Long> heartbeats) {
        if (heartbeats == null || heartbeats.size() < 2) {
            return RecordEntry.Status.UNCHECKED;
        }

        List<Long> intervals = new ArrayList<>();
        for (int i = 1; i < heartbeats.size(); i++) {
            intervals.add(heartbeats.get(i) - heartbeats.get(i-1));
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
        } else if (sdnn > 100) { // High variability might indicate arrhythmia
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

    private void createHeartBeatsImage(List<Long> heartbeats, Long id) {
        if (heartbeats.isEmpty()) return;

        // Create an image showing the heart rate timeline
        Mat graph = Mat.zeros(200, 800, CvType.CV_8UC3);

        // Draw background
        graph.setTo(new Scalar(255, 255, 255));

        // Calculate time scale
        long duration = heartbeats.get(heartbeats.size() - 1) - heartbeats.get(0);
        double pixelsPerMs = 800.0 / duration;

        // Draw grid lines
        Scalar gridColor = new Scalar(200, 200, 200);
        for (int i = 0; i < 800; i += 50) {
            Imgproc.line(graph, new Point(i, 0), new Point(i, 200), gridColor, 1);
        }
        for (int i = 0; i < 200; i += 25) {
            Imgproc.line(graph, new Point(0, i), new Point(800, i), gridColor, 1);
        }

        // Draw heartbeats
        Scalar beatColor = new Scalar(255, 0, 0);
        for (int i = 1; i < heartbeats.size(); i++) {
            long t1 = heartbeats.get(i-1) - heartbeats.get(0);
            long t2 = heartbeats.get(i) - heartbeats.get(0);

            int x1 = (int)(t1 * pixelsPerMs);
            int x2 = (int)(t2 * pixelsPerMs);

            // Draw heartbeat spike
            Imgproc.line(graph,
                new Point(x1, 150),
                new Point(x1, 50),
                beatColor, 2);

            // Connect beats with a line
            Imgproc.line(graph,
                new Point(x1, 150),
                new Point(x2, 150),
                beatColor, 1);
        }

        // Save image with proper path
        File imageFile = new File(
            context.getExternalFilesDir(null),
            String.format("%s/%s/heartbeats.jpg", AppConstant.DATA_DIR, id)
        );
        imageFile.getParentFile().mkdirs(); // Ensure directory exists
        org.opencv.imgcodecs.Imgcodecs.imwrite(imageFile.getAbsolutePath(), graph);
    }

    private void updateRecordStatus(Long id, RecordEntry.Status status, List<Long> heartbeats) {
        if (heartbeats.size() < 2) return;

        // Calculate average heart rate
        double averageInterval = 0;
        for (int i = 1; i < heartbeats.size(); i++) {
            averageInterval += (heartbeats.get(i) - heartbeats.get(i-1));
        }
        averageInterval /= (heartbeats.size() - 1);

        int bpm = (int)(60000.0 / averageInterval);

        // Update database record using context
        DataRecordServiceImpl dataRecordService = new DataRecordServiceImpl(context);
        RecordEntry record = dataRecordService.get(String.valueOf(id));
        if (record != null) {
            record.setStatus(status);
            record.setBeatsPerMinute(bpm);
            record.setHeartbeats(heartbeats);
            record.setUpdatedAt(System.currentTimeMillis());
            dataRecordService.saveData(record);
        }
    }

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
