package com.gunadarma.heartratearrhythmiachecker.service;

import android.util.Log;

import com.gunadarma.heartratearrhythmiachecker.constant.AppConstant;
import com.gunadarma.heartratearrhythmiachecker.model.RecordEntry;
import com.gunadarma.heartratearrhythmiachecker.service.mediacreator.ImageGeneratorServiceImpl;
import com.gunadarma.heartratearrhythmiachecker.service.mediacreator.VideoGeneratorServiceImpl;
import com.gunadarma.heartratearrhythmiachecker.service.rppg.RPPGHandPalmServiceImpl;
import com.gunadarma.heartratearrhythmiachecker.service.rppg.RPPGHeadFaceServiceImpl;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
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

public class MainMediaProcessingServiceImplBackup {
    // implements MainMediaProcessingService {
//    public enum DetectionMode {
//        FACE,
//        HAND
//    }
//
//    private final android.content.Context context;
//    private final CascadeClassifier faceDetector;
//    private final DetectionMode currentMode;
//    private final ImageGeneratorServiceImpl imageGeneratorService;
//    private final VideoGeneratorServiceImpl videoGeneratorService;
//    private final RPPGHeadFaceServiceImpl rppgHeadFaceService;
//    private final RPPGHandPalmServiceImpl rppgHandPalmService;
//    private final MediaPipeHandTracker handTracker;
//    private static final String TAG = "MediaProcessingService";
//
//    public MainMediaProcessingServiceImplBackup(android.content.Context context) {
//        this.context = context;
//        this.faceDetector = initializeFaceDetector();
//        this.currentMode = DetectionMode.HAND; // Default to hand detection
//
//        this.handTracker = new MediaPipeHandTracker(context);
//        this.imageGeneratorService = new ImageGeneratorServiceImpl(context);
//        this.videoGeneratorService = new VideoGeneratorServiceImpl(context);
//        this.rppgHeadFaceService = new RPPGHeadFaceServiceImpl(context);
//        this.rppgHandPalmService = new RPPGHandPalmServiceImpl(context);
//    }
//
//    /**
//     * Release all resources including MediaPipe hand tracker
//     */
//    public void release() {
//        try {
//            if (handTracker != null) {
//                handTracker.release();
//            }
//            Log.i(TAG, "MediaProcessingService resources released");
//        } catch (Exception e) {
//            Log.e(TAG, "Error releasing MediaProcessingService resources", e);
//        }
//    }
//
//    private CascadeClassifier initializeFaceDetector() {
//        try {
//            // Create a temporary file to store the cascade classifier
//            File cascadeDir = context.getDir("cascade", android.content.Context.MODE_PRIVATE);
//            cascadeDir.mkdirs();
//            File cascadeFile = new File(cascadeDir, "haarcascade_frontalface_alt.xml");
//
//            // Copy the cascade file from assets to the temporary file
//            InputStream is = context.getAssets().open("haarcascades/haarcascade_frontalface_alt.xml");
//            FileOutputStream os = new FileOutputStream(cascadeFile);
//            byte[] buffer = new byte[4096];
//            int bytesRead;
//            while ((bytesRead = is.read(buffer)) != -1) {
//                os.write(buffer, 0, bytesRead);
//            }
//            is.close();
//            os.close();
//
//            // Create the cascade classifier
//            CascadeClassifier detector = new CascadeClassifier(cascadeFile.getAbsolutePath());
//
//            // Delete the temporary file
//            cascadeFile.delete();
//            cascadeDir.delete();
//
//            if (detector.empty()) {
//                Log.e(TAG, "Failed to load cascade classifier");
//                return null;
//            }
//            return detector;
//        } catch (Exception e) {
//            Log.e(TAG, "Error initializing face detector", e);
//            return null;
//        }
//    }
//
//    private static final double SECONDS_PER_WINDOW = 3.0; // Increased window size for more stable readings
//    private static final int WINDOW_SIZE = (int)(AppConstant.OUTPUT_VIDEO_FPS * SECONDS_PER_WINDOW);
//    private static final double MIN_HR_BPM = 40.0; // Minimum heart rate in BPM
//    private static final double MAX_HR_BPM = 200.0; // Maximum heart rate in BPM
//    private static final double PEAK_THRESHOLD = 0.55; // Threshold for peak detection
//    private static final int MIN_PEAK_DISTANCE = (int)(0.4 * AppConstant.OUTPUT_VIDEO_FPS); // Minimum 0.4 seconds between peaks
//
//    @Override
//    public void createHeartBeatsVideo(RecordEntry recordEntry) {
//        // Get the proper file path using context
//        File videoFile = new File(
//            context.getExternalFilesDir(null),
//            String.format("%s/%s/%s", AppConstant.DATA_DIR, recordEntry.getId(), AppConstant.ORIGINAL_VIDEO_NAME)
//        );
//        String videoPath = videoFile.getAbsolutePath();
//
//        // 1. Extract heartbeats based on current detection mode
//        List<HeartRateData> heartRateData;
//        if (currentMode == DetectionMode.HAND) {
//            heartRateData = processVideoAndExtractHeartRateFromHand(String.valueOf(recordEntry.getId()), videoPath);
//        } else {
//            heartRateData = processVideoAndExtractHeartRate(String.valueOf(recordEntry.getId()), videoPath);
//        }
//        List<Long> heartbeatTimestamps = processHeartRateData(heartRateData);
//        int duration = getVideoDuration(videoPath);
//
//        // 2. Analyze for arrhythmia
//        RecordEntry.Status status = analyzeArrhythmia(heartbeatTimestamps);
//
//        // 3. Create heartbeats visualization
//        imageGeneratorService.createHeartBeatsImage(heartbeatTimestamps, recordEntry.getId());
//
//        // 4. Update record status
//        updateRecordStatus(recordEntry, status, heartbeatTimestamps, duration);
//    }
//
//    private int getVideoDuration(String videoPath) {
//        VideoCapture cap = new VideoCapture(videoPath);
//        if (!cap.isOpened()) {
//            Log.e(TAG, "Failed to open video file: " + videoPath);
//            return 0;
//        }
//        double fps = cap.get(Videoio.CAP_PROP_FPS);
//        int frameCount = (int) cap.get(Videoio.CAP_PROP_FRAME_COUNT);
//        cap.release();
//        return (int)(frameCount / fps);
//    }
//
//    private List<HeartRateData> extractHeartRateFromVideo(String videoPath) {
//        Log.i(TAG, "Processing video: " + videoPath);
//        VideoCapture cap = new VideoCapture(videoPath);
//        if (!cap.isOpened()) {
//            Log.e(TAG, "Failed to open video file: " + videoPath);
//            return new ArrayList<>();
//        }
//
//        // Get video properties
//        double fps = cap.get(Videoio.CAP_PROP_FPS);
//        int frameCount = (int) cap.get(Videoio.CAP_PROP_FRAME_COUNT);
//
//        // Lists to store RGB signals
//        List<Double> redChannel = new ArrayList<>();
//        List<Double> greenChannel = new ArrayList<>();
//        List<Double> blueChannel = new ArrayList<>();
//        List<Long> timestamps = new ArrayList<>();
//
//        Mat frame = new Mat();
//        Mat resized = new Mat();
//        Mat rgb = new Mat();
//
//        long startTime = System.currentTimeMillis();
//        int frameIndex = 0;
//
//        while (cap.read(frame)) {
//            if (frame.empty()) continue;
//
//            // Resize for faster processing (9:16 aspect ratio)
//            Imgproc.resize(frame, resized, new Size(240, 426));
//
//            // Convert to RGB
//            Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_BGR2RGB);
//
//            // Calculate mean color values
//            Scalar means = Core.mean(rgb);
//
//            // Store RGB values
//            redChannel.add(means.val[0]);
//            greenChannel.add(means.val[1]);
//            blueChannel.add(means.val[2]);
//
//            // Calculate timestamp for this frame
//            long timestamp = startTime + (long)(frameIndex * (1000.0 / fps));
//            timestamps.add(timestamp);
//
//            frameIndex++;
//        }
//
//        cap.release();
//
//        // Process the signals
//        return processRPPGSignals(redChannel, greenChannel, blueChannel, timestamps, fps);
//    }
//
//    private List<HeartRateData> processRPPGSignals(List<Double> red, List<Double> green, List<Double> blue,
//                                                  List<Long> timestamps, double fps) {
//        List<HeartRateData> heartRateDataList = new ArrayList<>();
//
//        // Convert lists to OpenCV Mats with double precision
//        Mat redMat = new Mat(red.size(), 1, CvType.CV_64F);
//        Mat greenMat = new Mat(green.size(), 1, CvType.CV_64F);
//        Mat blueMat = new Mat(blue.size(), 1, CvType.CV_64F);
//
//        for (int i = 0; i < red.size(); i++) {
//            redMat.put(i, 0, red.get(i));
//            greenMat.put(i, 0, green.get(i));
//            blueMat.put(i, 0, blue.get(i));
//        }
//
//        // Detrend and normalize signals
//        Mat redDetrended = detrendSignal(redMat);
//        Mat greenDetrended = detrendSignal(greenMat);
//        Mat blueDetrended = detrendSignal(blueMat);
//
//        // PCA color space transform
//        List<Mat> channels = new ArrayList<>();
//        channels.add(redDetrended);
//        channels.add(greenDetrended);
//        channels.add(blueDetrended);
//
//        Mat signals = new Mat();
//        Core.vconcat(channels, signals);
//
//        Mat covar = new Mat();
//        Mat mean = new Mat();
//        Core.calcCovarMatrix(signals, covar, mean, Core.COVAR_NORMAL | Core.COVAR_ROWS);
//
//        Mat eigenvalues = new Mat();
//        Mat eigenvectors = new Mat();
//        Core.eigen(covar, eigenvalues, eigenvectors);
//
//        // Ensure eigenvectors matrix is proper type
//        if (eigenvectors.type() != CvType.CV_64F) {
//            eigenvectors.convertTo(eigenvectors, CvType.CV_64F);
//        }
//
//        // Project signals onto first principal component
//        Mat firstEigenvector = eigenvectors.row(0);
//        Mat projected = new Mat();
//        Core.gemm(signals, firstEigenvector.t(), 1.0, new Mat(), 0.0, projected);
//
//        // Apply bandpass filter
//        Mat filteredSignal = applyBandpassFilter(projected, fps);
//
//        // Process signal in windows
//        int windowSize = (int)(SECONDS_PER_WINDOW * fps);
//        int stepSize = windowSize / 2; // 50% overlap
//
//        for (int i = 0; i < filteredSignal.rows() - windowSize; i += stepSize) {
//            Mat window = filteredSignal.rowRange(i, i + windowSize);
//            List<Integer> peakIndices = findPeaks(window);
//
//            if (!peakIndices.isEmpty()) {
//                double averageInterval = (double)windowSize / peakIndices.size();
//                double heartRate = 60.0 * fps / averageInterval;
//
//                if (heartRate >= MIN_HR_BPM && heartRate <= MAX_HR_BPM) {
//                    for (Integer peakIndex : peakIndices) {
//                        int globalIndex = i + peakIndex;
//                        if (globalIndex < timestamps.size()) {
//                            heartRateDataList.add(new HeartRateData(heartRate, timestamps.get(globalIndex)));
//                        }
//                    }
//                }
//            }
//        }
//
//        return heartRateDataList;
//    }
//
//    private List<Integer> findPeaks(Mat signal) {
//        List<Integer> peaks = new ArrayList<>();
//        double[] prev = new double[1];
//        double[] curr = new double[1];
//        double[] next = new double[1];
//
//        // Calculate the standard deviation for adaptive thresholding
//        MatOfDouble stdDev = new MatOfDouble();
//        MatOfDouble mean = new MatOfDouble();
//        Core.meanStdDev(signal, mean, stdDev);
//        double threshold = stdDev.get(0, 0)[0] * PEAK_THRESHOLD; // Using configurable threshold
//
//        double lastPeakValue = Double.MIN_VALUE;
//        int lastPeakIndex = -MIN_PEAK_DISTANCE;
//
//        for (int i = 1; i < signal.rows() - 1; i++) {
//            signal.get(i-1, 0, prev);
//            signal.get(i, 0, curr);
//            signal.get(i+1, 0, next);
//
//            // More strict peak detection
//            if (curr[0] > prev[0] && curr[0] > next[0] && // Local maximum
//                curr[0] > mean.get(0, 0)[0] + threshold && // Above threshold
//                i - lastPeakIndex >= MIN_PEAK_DISTANCE) { // Minimum distance
//
//                // Only accept peaks that are significantly higher than the last one
//                if (curr[0] > lastPeakValue * 0.8 || i - lastPeakIndex >= MIN_PEAK_DISTANCE * 2) {
//                    peaks.add(i);
//                    lastPeakValue = curr[0];
//                    lastPeakIndex = i;
//                }
//            }
//        }
//
//        return peaks;
//    }
//
//    private Mat detrendSignal(Mat signal) {
//        // Convert signal to double precision
//        Mat signalDouble = new Mat();
//        signal.convertTo(signalDouble, CvType.CV_64F);
//
//        // Create time vector
//        Mat timeVector = new Mat(signal.rows(), 1, CvType.CV_64F);
//        for (int i = 0; i < signal.rows(); i++) {
//            timeVector.put(i, 0, (double)i);
//        }
//
//        // Fit linear trend
//        Mat trend = new Mat();
//        Core.solve(timeVector, signalDouble, trend, Core.DECOMP_SVD);
//
//        // Ensure trend matrix is proper size and type
//        if (trend.rows() != 1) {
//            Core.transpose(trend, trend);
//        }
//
//        // Calculate and remove trend
//        Mat detrended = new Mat();
//        Mat trendLine = timeVector.mul(trend);
//        Core.subtract(signalDouble, trendLine, detrended);
//
//        // Normalize
//        Core.normalize(detrended, detrended, 0, 1, Core.NORM_MINMAX);
//
//        return detrended;
//    }
//
//    private Mat processColorSignals(Mat redMat, Mat greenMat, Mat blueMat, double fps) {
//        // Detrend and normalize signals
//        Mat redDetrended = detrendSignal(redMat);
//        Mat greenDetrended = detrendSignal(greenMat);
//        Mat blueDetrended = detrendSignal(blueMat);
//
//        // Use green channel predominantly as it's most sensitive to blood volume changes
//        Mat weightedSignal = new Mat();
//        Core.addWeighted(greenDetrended, 0.7, redDetrended, 0.2, 0, weightedSignal);
//        Core.addWeighted(weightedSignal, 1.0, blueDetrended, 0.1, 0, weightedSignal);
//
//        // Apply bandpass filter
//        Mat filteredSignal = applyBandpassFilter(weightedSignal, fps);
//
//        // Cleanup
//        redDetrended.release();
//        greenDetrended.release();
//        blueDetrended.release();
//        weightedSignal.release();
//
//        return filteredSignal;
//    }
//
//    private Mat applyBandpassFilter(Mat signal, double fps) {
//        // Convert signal to double precision
//        Mat signalDouble = new Mat();
//        signal.convertTo(signalDouble, CvType.CV_64F);
//
//        // Prepare for FFT
//        int rows = signalDouble.rows();
//        int optimalRows = Core.getOptimalDFTSize(rows);
//        Mat padded = new Mat();
//        Core.copyMakeBorder(signalDouble, padded, 0, optimalRows - rows, 0, 0, Core.BORDER_CONSTANT);
//
//        // Convert to complex format for FFT
//        List<Mat> planes = new ArrayList<>();
//        planes.add(padded);
//        planes.add(Mat.zeros(padded.size(), CvType.CV_64F));
//        Mat complexSignal = new Mat();
//        Core.merge(planes, complexSignal);
//
//        // FFT
//        Core.dft(complexSignal, complexSignal);
//
//        // Create narrower bandpass filter (0.7 Hz - 3.0 Hz) corresponding to 42-180 BPM
//        double lowCut = 0.7 / fps;  // 42 BPM
//        double highCut = 3.0 / fps; // 180 BPM
//
//        // Create filter with same size and type as complex signal
//        Mat filter = Mat.zeros(complexSignal.size(), complexSignal.type());
//        List<Mat> filterPlanes = new ArrayList<>();
//        Mat filterReal = Mat.zeros(complexSignal.size(), CvType.CV_64F);
//        Mat filterImag = Mat.zeros(complexSignal.size(), CvType.CV_64F);
//
//        int halfRows = optimalRows / 2;
//        for (int i = 0; i < optimalRows; i++) {
//            double freq = (i <= halfRows) ? (double)i : (double)(optimalRows - i);
//            freq /= optimalRows;
//
//            // Smooth transition at cutoff frequencies using Hanning window
//            double value = 0.0;
//            if (freq >= lowCut && freq <= highCut) {
//                double normalized = (freq - lowCut) / (highCut - lowCut);
//                value = 0.5 * (1 - Math.cos(2 * Math.PI * normalized));
//            }
//
//            filterReal.put(i, 0, value);
//            filterImag.put(i, 0, 0.0);
//        }
//
//        filterPlanes.add(filterReal);
//        filterPlanes.add(filterImag);
//        Core.merge(filterPlanes, filter);
//
//        // Apply filter in frequency domain
//        Mat filteredSpectrum = new Mat();
//        Core.mulSpectrums(complexSignal, filter, filteredSpectrum, 0);
//
//        // Inverse FFT
//        Core.idft(filteredSpectrum, filteredSpectrum);
//
//        // Extract real part and crop to original size
//        Core.split(filteredSpectrum, planes);
//        Mat filteredSignal = planes.get(0).rowRange(0, rows);
//
//        // Cleanup
//        padded.release();
//        complexSignal.release();
//        filter.release();
//        filteredSpectrum.release();
//        filterReal.release();
//        filterImag.release();
//
//        return filteredSignal;
//    }
//
//    private List<Long> processHeartRateData(List<HeartRateData> heartRateData) {
//        List<Long> heartbeatTimestamps = new ArrayList<>();
//
//        if (heartRateData.isEmpty()) {
//            return heartbeatTimestamps;
//        }
//
//        // Convert heart rate data to beat timestamps
//        double averageHR = heartRateData.stream()
//                .mapToDouble(HeartRateData::getIntensity)
//                .average()
//                .orElse(60.0);
//
//        // Calculate average interval between beats
//        double averageIntervalMs = 60000.0 / averageHR;
//
//        // Generate timestamps
//        long startTime = heartRateData.get(0).getTimestamp();
//        long currentTime = startTime;
//
//        while (currentTime <= heartRateData.get(heartRateData.size()-1).getTimestamp()) {
//            heartbeatTimestamps.add(currentTime);
//            currentTime += (long)averageIntervalMs;
//        }
//
//        return heartbeatTimestamps;
//    }
//
//    private RecordEntry.Status analyzeArrhythmia(List<Long> heartbeats) {
//        if (heartbeats == null || heartbeats.size() < 2) {
//            return RecordEntry.Status.UNCHECKED;
//        }
//
//        List<Long> intervals = new ArrayList<>();
//        for (int i = 1; i < heartbeats.size(); i++) {
//            intervals.add(heartbeats.get(i) - heartbeats.get(i-1));
//        }
//
//        // Calculate mean RR interval
//        double meanRR = intervals.stream().mapToLong(l -> l).average().orElse(0);
//
//        // Calculate standard deviation of RR intervals (SDNN)
//        double sdnn = calculateSDNN(intervals, meanRR);
//
//        // Calculate heart rate
//        double heartRate = 60000.0 / meanRR; // Convert to BPM
//
//        // Classify based on heart rate and variability
//        if (heartRate > 100) {
//            return RecordEntry.Status.ARRHYTHMIA_TACHYCARDIA;
//        } else if (heartRate < 60) {
//            return RecordEntry.Status.ARRHYTHMIA_BRADYCARDIA;
//        } else if (sdnn > 100) // High variability might indicate arrhythmia
//        {
//            return RecordEntry.Status.ARRHYTHMIA_IRREGULAR;
//        }
//
//        return RecordEntry.Status.NORMAL;
//    }
//
//    private double calculateSDNN(List<Long> intervals, double mean) {
//        return Math.sqrt(
//            intervals.stream()
//                .mapToDouble(interval -> Math.pow(interval - mean, 2))
//                .average()
//                .orElse(0.0)
//        );
//    }
//
//    private void updateRecordStatus(RecordEntry recordEntry, RecordEntry.Status status, List<Long> heartbeats, int duration) {
//        double averageInterval = 0;
//        if (heartbeats.size() > 2) {
//            // Calculate average heart rate
//            for (int i = 1; i < heartbeats.size(); i++) {
//                averageInterval += (heartbeats.get(i) - heartbeats.get(i-1));
//            }
//            averageInterval /= (heartbeats.size() - 1);
//        }
//
//        int bpm = averageInterval >= 0 ? 0 : (int)(60000.0 / averageInterval);
//
//        // Update database record using context
//        recordEntry.setStatus(status);
//        recordEntry.setDuration(duration);
//        recordEntry.setBeatsPerMinute(bpm);
//        recordEntry.setHeartbeats(Optional.of(heartbeats).orElse(List.of()));
//        recordEntry.setUpdatedAt(System.currentTimeMillis());
//        System.out.println("asdfasdf");
//    }
//
//    private void createOverlayVideo(String inputPath, String outputPath, List<HeartRateData> heartRateData) {
//        if (faceDetector == null || faceDetector.empty()) {
//            Log.e(TAG, "Face detector not initialized");
//            return;
//        }
//
//        VideoCapture cap = new VideoCapture();
//        VideoWriter writer = null;
//        android.media.MediaMetadataRetriever retriever = new android.media.MediaMetadataRetriever();
//
//        try {
//            File videoFile = new File(inputPath);
//            if (!videoFile.exists()) {
//                Log.e(TAG, "Input video file does not exist: " + inputPath);
//                return;
//            }
//            if (!cap.open(inputPath)) {
//                Log.e(TAG, "Failed to open input video: " + inputPath);
//                return;
//            }
//
//            // Get video properties
//            int frameWidth = (int) cap.get(Videoio.CAP_PROP_FRAME_WIDTH);
//            int frameHeight = (int) cap.get(Videoio.CAP_PROP_FRAME_HEIGHT);
//            double fps = cap.get(Videoio.CAP_PROP_FPS);
//
//            // Get input video orientation
//            retriever.setDataSource(inputPath);
//            String rotationString = retriever.extractMetadata(android.media.MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION);
//            int videoRotation = rotationString != null ? Integer.parseInt(rotationString) : 0;
//            Log.d(TAG, "Video rotation: " + videoRotation);
//
//            // If video is rotated 90 or 270 degrees, swap width and height
//            if (videoRotation == 90 || videoRotation == 270) {
//                int temp = frameWidth;
//                frameWidth = frameHeight;
//                frameHeight = temp;
//            }
//
//            // Create output directory if it doesn't exist
//            new File(outputPath).getParentFile().mkdirs();
//
//            // Create VideoWriter with H264 codec
//            writer = new VideoWriter();
//            int codec = VideoWriter.fourcc('H', '2', '6', '4');  // Using H264 codec
//            Size frameSize = new Size(frameWidth, frameHeight);
//
//            // Ensure output path ends with .mp4
//            if (!outputPath.toLowerCase().endsWith(".mp4")) {
//                outputPath = outputPath + ".mp4";
//            }
//
//            // Configure additional properties for the VideoWriter
//            double frameRate = fps > 0 ? fps : AppConstant.OUTPUT_VIDEO_FPS;
//            boolean isColor = true;
//
//            if (!writer.open(outputPath, codec, frameRate, frameSize, isColor)) {
//                // Try alternative codec parameters
//                codec = VideoWriter.fourcc('X', '2', '6', '4');
//                if (!writer.open(outputPath, codec, frameRate, frameSize, isColor)) {
//                    codec = VideoWriter.fourcc('x', '2', '6', '4');
//                    if (!writer.open(outputPath, codec, frameRate, frameSize, isColor)) {
//                        Log.e(TAG, "Failed to open VideoWriter with any supported codec configuration");
//                        return;
//                    }
//                }
//            }
//
//            // Add missing variable declarations for RGB channels and timestamps
//            List<Double> redChannel = new ArrayList<>();
//            List<Double> greenChannel = new ArrayList<>();
//            List<Double> blueChannel = new ArrayList<>();
//            List<Long> timestamps = new ArrayList<>();
//            List<HeartRateData> heartRateDataList = new ArrayList<>();
//
//            Mat frame = new Mat();
//            int frameCount = 0;
//            long startTime = System.currentTimeMillis();
//            double graphWidth = frameWidth * 0.3;
//            double graphHeight = frameHeight * 0.2;
//            int graphMaxPoints = 50;
//            List<Point> graphPoints = new ArrayList<>();
//
//            while (cap.read(frame)) {
//                if (frame.empty()) {
//                    Log.w(TAG, "Empty frame encountered");
//                    continue;
//                }
//
//                try {
//                    // Convert frame to correct type and color space
//                    Mat processedFrame = new Mat();
//                    frame.copyTo(processedFrame);
//
//                    // Handle rotation first
//                    Mat rotatedFrame = new Mat();
//                    switch (videoRotation) {
//                        case 90:
//                            Core.rotate(processedFrame, rotatedFrame, Core.ROTATE_90_CLOCKWISE);
//                            break;
//                        case 180:
//                            Core.rotate(processedFrame, rotatedFrame, Core.ROTATE_180);
//                            break;
//                        case 270:
//                            Core.rotate(processedFrame, rotatedFrame, Core.ROTATE_90_COUNTERCLOCKWISE);
//                            break;
//                        default:
//                            processedFrame.copyTo(rotatedFrame);
//                            break;
//                    }
//                    processedFrame.release();
//                    processedFrame = rotatedFrame;
//
//                    // Now handle color space conversions
//                    Imgproc.cvtColor(processedFrame, processedFrame, Imgproc.COLOR_BGR2RGB);
//
//                    // Detect face
//                    MatOfRect faces = new MatOfRect();
//                    Mat grayFrame = new Mat();
//                    Imgproc.cvtColor(processedFrame, grayFrame, Imgproc.COLOR_RGB2GRAY);
//                    if (faceDetector != null && !faceDetector.empty()) {
//                        faceDetector.detectMultiScale(grayFrame, faces);
//                    }
//
//                    // Draw face rectangles and forehead ROI
//                    for (Rect face : faces.toArray()) {
//                        Imgproc.rectangle(processedFrame, face, new Scalar(0, 255, 0), 2);
//                        int foreheadHeight = (int)(face.height * 0.3);
//                        Rect foreheadROI = new Rect(
//                            face.x + face.width/4,
//                            face.y,
//                            face.width/2,
//                            foreheadHeight
//                        );
//                        Imgproc.rectangle(processedFrame, foreheadROI, new Scalar(255, 0, 0), 2);
//
//                        // Extract color data from forehead ROI
//                        Mat roi = new Mat(processedFrame, foreheadROI);
//                        Scalar means = Core.mean(roi);
//                        redChannel.add(means.val[0]);
//                        greenChannel.add(means.val[1]);
//                        blueChannel.add(means.val[2]);
//
//                        long timestamp = startTime + (long)(frameCount * (1000.0 / fps));
//                        timestamps.add(timestamp);
//                        roi.release();
//                    }
//
//                    // Update heart rate graph
//                    if (frameCount < heartRateDataList.size()) {
//                        HeartRateData currentData = heartRateDataList.get(frameCount);
//                        double normalizedValue = currentData.getIntensity() / 150.0;
//                        graphPoints.add(new Point(
//                            processedFrame.width() - graphWidth + (graphWidth * graphPoints.size() / graphMaxPoints),
//                            processedFrame.height() - graphHeight + (graphHeight * (1 - normalizedValue))
//                        ));
//
//                        if (graphPoints.size() > graphMaxPoints) {
//                            graphPoints.remove(0);
//                        }
//
//                        // Draw graph background
//                        Imgproc.rectangle(processedFrame,
//                            new Point(processedFrame.width() - graphWidth, processedFrame.height() - graphHeight),
//                            new Point(processedFrame.width(), processedFrame.height()),
//                            new Scalar(255, 255, 255), -1);
//
//                        // Draw graph lines
//                        for (int i = 1; i < graphPoints.size(); i++) {
//                            Imgproc.line(processedFrame, graphPoints.get(i-1), graphPoints.get(i),
//                                new Scalar(0, 0, 255), 2);
//                        }
//                    }
//
//                    // Draw FPS
//                    double currentFps = frameCount / ((System.currentTimeMillis() - startTime) / 1000.0);
//                    Imgproc.putText(processedFrame, String.format(Locale.getDefault(), "FPS: %.1f", currentFps),
//                        new Point(10, processedFrame.height() - 10),
//                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(0, 255, 0), 2);
//
//                    // Convert back to BGR before writing
//                    Imgproc.cvtColor(processedFrame, processedFrame, Imgproc.COLOR_RGB2BGR);
//
//                    // Write the frame
//                    writer.write(processedFrame);
//                    frameCount++;
//
//                    if (frameCount % AppConstant.OUTPUT_VIDEO_FPS == 0) {
//                        Log.d(TAG, "Processed " + frameCount + " frames");
//                    }
//
//                    // Release resources
//                    processedFrame.release();
//                    grayFrame.release();
//                } catch (Exception e) {
//                    Log.e(TAG, "Error processing frame " + frameCount + ": " + e.getMessage());
//                }
//            }
//
//            // Verify frames were written
//            if (frameCount == 0) {
//                Log.e(TAG, "No frames were processed");
//                // Delete empty output file
//                new File(outputPath).delete();
//            } else {
//                Log.i(TAG, "Successfully processed " + frameCount + " frames");
//            }
//
//        } catch (Exception e) {
//            Log.e(TAG, "Error in createOverlayVideo: " + e.getMessage());
//            // Delete potentially corrupt output file
//            new File(outputPath).delete();
//        } finally {
//            try {
//                if (writer != null) {
//                    writer.release();
//                }
//                if (cap != null && cap.isOpened()) {
//                    cap.release();
//                }
//                retriever.release();
//            } catch (Exception e) {
//                Log.e(TAG, "Error releasing resources: " + e.getMessage());
//            }
//        }
//    }
//
//    // by face
//    private List<HeartRateData> processVideoAndExtractHeartRate(String recordId, String videoPath) {
//        Log.i(TAG, "Processing video with face detection: " + videoPath);
//        VideoCapture cap = new VideoCapture(videoPath);
//        if (!cap.isOpened()) {
//            Log.e(TAG, "Failed to open video file: " + videoPath);
//            return new ArrayList<>();
//        }
//
//        // Get video properties and orientation
//        android.media.MediaMetadataRetriever retriever = new android.media.MediaMetadataRetriever();
//        retriever.setDataSource(videoPath);
//        String rotationString = retriever.extractMetadata(android.media.MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION);
//        int videoRotation = rotationString != null ? Integer.parseInt(rotationString) : 0;
//        try {
//            retriever.release();
//        } catch (IOException e) {
//            videoRotation = 0;
//        }
//        Log.d(TAG, "Video rotation: " + videoRotation);
//
//        double fps = cap.get(Videoio.CAP_PROP_FPS);
//        int frameWidth = (int) cap.get(Videoio.CAP_PROP_FRAME_WIDTH);
//        int frameHeight = (int) cap.get(Videoio.CAP_PROP_FRAME_HEIGHT);
//
//        // If video is rotated 90 or 270 degrees, swap width and height for the output
//        Size finalFrameSize;
//        if (videoRotation == 90 || videoRotation == 270) {
//            finalFrameSize = new Size(frameHeight, frameWidth);
//        } else {
//            finalFrameSize = new Size(frameWidth, frameHeight);
//        }
//
//        // Setup video writer with correct orientation
//        VideoWriter writer = new VideoWriter();
//        String outputPath = new File(
//            context.getExternalFilesDir(null),
//            String.format("%s/%s/%s", AppConstant.DATA_DIR, recordId, AppConstant.FINAL_VIDEO_NAME)
//        ).getAbsolutePath();
//
//        // Create output directory
//        new File(outputPath).getParentFile().mkdirs();
//
//        // Configure video writer
//        int codec = VideoWriter.fourcc('H', '2', '6', '4');
//        if (!writer.open(outputPath, codec, fps, finalFrameSize, true)) {
//            codec = VideoWriter.fourcc('X', '2', '6', '4');
//            if (!writer.open(outputPath, codec, fps, finalFrameSize, true)) {
//                codec = VideoWriter.fourcc('x', '2', '6', '4');
//                if (!writer.open(outputPath, codec, fps, finalFrameSize, true)) {
//                    Log.e(TAG, "Failed to open VideoWriter");
//                    return new ArrayList<>();
//                }
//            }
//        }
//
//        // Lists to store RGB signals
//        List<Double> redChannel = new ArrayList<>();
//        List<Double> greenChannel = new ArrayList<>();
//        List<Double> blueChannel = new ArrayList<>();
//        List<Long> timestamps = new ArrayList<>();
//
//        Mat frame = new Mat();
//        long startTime = System.currentTimeMillis();
//        int frameCount = 0;
//        double graphWidth = finalFrameSize.width * 0.3;
//        double graphHeight = finalFrameSize.height * 0.2;
//        int graphMaxPoints = 50;
//        List<Point> graphPoints = new ArrayList<>();
//        List<HeartRateData> heartRateDataList = new ArrayList<>();
//
//        // Heartbeat simulation for ECG
//        double heartbeatPhase = 0.0;
//        double baseHeartRate = 72.0; // Base BPM for simulation
//        List<Double> heartbeatTimes = new ArrayList<>();
//
//        // ECG visualization variables
//        double ecgWidth = finalFrameSize.width * 0.25;
//        double ecgHeight = finalFrameSize.height * 0.15;
//        int ecgMaxPoints = 100;
//        List<Point> ecgPoints = new ArrayList<>();
//
//        int framesWithHandDetection = 0;
//
//        try {
//            while (cap.read(frame)) {
//                if (frame.empty()) {
//                    Log.w(TAG, "Empty frame encountered");
//                    continue;
//                }
//
//                try {
//                    // Convert frame to correct type and color space
//                    Mat processedFrame = new Mat();
//                    frame.copyTo(processedFrame);
//
//                    // Handle rotation first
//                    Mat rotatedFrame = new Mat();
//                    switch (videoRotation) {
//                        case 90:
//                            Core.rotate(processedFrame, rotatedFrame, Core.ROTATE_90_CLOCKWISE);
//                            break;
//                        case 180:
//                            Core.rotate(processedFrame, rotatedFrame, Core.ROTATE_180);
//                            break;
//                        case 270:
//                            Core.rotate(processedFrame, rotatedFrame, Core.ROTATE_90_COUNTERCLOCKWISE);
//                            break;
//                        default:
//                            processedFrame.copyTo(rotatedFrame);
//                            break;
//                    }
//                    processedFrame.release();
//                    processedFrame = rotatedFrame;
//
//                    // Now handle color space conversions
//                    Imgproc.cvtColor(processedFrame, processedFrame, Imgproc.COLOR_BGR2RGB);
//
//                    // Detect face
//                    MatOfRect faces = new MatOfRect();
//                    Mat grayFrame = new Mat();
//                    Imgproc.cvtColor(processedFrame, grayFrame, Imgproc.COLOR_RGB2GRAY);
//                    if (faceDetector != null && !faceDetector.empty()) {
//                        faceDetector.detectMultiScale(grayFrame, faces);
//                    }
//
//                    // Draw face rectangles and forehead ROI
//                    for (Rect face : faces.toArray()) {
//                        Imgproc.rectangle(processedFrame, face, new Scalar(0, 255, 0), 2);
//                        int foreheadHeight = (int)(face.height * 0.3);
//                        Rect foreheadROI = new Rect(
//                            face.x + face.width/4,
//                            face.y,
//                            face.width/2,
//                            foreheadHeight
//                        );
//                        Imgproc.rectangle(processedFrame, foreheadROI, new Scalar(255, 0, 0), 2);
//
//                        // Extract color data from forehead ROI
//                        Mat roi = new Mat(processedFrame, foreheadROI);
//                        Scalar means = Core.mean(roi);
//                        redChannel.add(means.val[0]);
//                        greenChannel.add(means.val[1]);
//                        blueChannel.add(means.val[2]);
//
//                        long timestamp = startTime + (long)(frameCount * (1000.0 / fps));
//                        timestamps.add(timestamp);
//                        roi.release();
//                    }
//
//                    // Update heart rate graph
//                    if (frameCount < heartRateDataList.size()) {
//                        HeartRateData currentData = heartRateDataList.get(frameCount);
//                        double normalizedValue = currentData.getIntensity() / 150.0;
//                        graphPoints.add(new Point(
//                            processedFrame.width() - graphWidth + (graphWidth * graphPoints.size() / graphMaxPoints),
//                            processedFrame.height() - graphHeight + (graphHeight * (1 - normalizedValue))
//                        ));
//
//                        if (graphPoints.size() > graphMaxPoints) {
//                            graphPoints.remove(0);
//                        }
//
//                        // Draw graph background
//                        Imgproc.rectangle(processedFrame,
//                            new Point(processedFrame.width() - graphWidth, processedFrame.height() - graphHeight),
//                            new Point(processedFrame.width(), processedFrame.height()),
//                            new Scalar(255, 255, 255), -1);
//
//                        // Draw graph lines
//                        for (int i = 1; i < graphPoints.size(); i++) {
//                            Imgproc.line(processedFrame, graphPoints.get(i-1), graphPoints.get(i),
//                                new Scalar(0, 0, 255), 2);
//                        }
//                    }
//
//                    // Draw FPS
//                    double currentFps = frameCount / ((System.currentTimeMillis() - startTime) / 1000.0);
//                    Imgproc.putText(processedFrame, String.format(Locale.getDefault(), "FPS: %.1f", currentFps),
//                        new Point(10, processedFrame.height() - 10),
//                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(0, 255, 0), 2);
//
//                    // Convert back to BGR before writing
//                    Imgproc.cvtColor(processedFrame, processedFrame, Imgproc.COLOR_RGB2BGR);
//
//                    // Write the frame
//                    writer.write(processedFrame);
//                    frameCount++;
//
//                    if (frameCount % AppConstant.OUTPUT_VIDEO_FPS == 0) {
//                        Log.d(TAG, "Processed " + frameCount + " frames");
//                    }
//
//                    // Release resources
//                    processedFrame.release();
//                    grayFrame.release();
//                } catch (Exception e) {
//                    Log.e(TAG, "Error processing frame " + frameCount + ": " + e.getMessage());
//                }
//            }
//
//            // Verify frames were written
//            if (frameCount == 0) {
//                Log.e(TAG, "No frames were processed");
//                // Delete empty output file
//                new File(outputPath).delete();
//            } else {
//                Log.i(TAG, "Successfully processed " + frameCount + " frames");
//            }
//
//        } catch (Exception e) {
//            Log.e(TAG, "Error in createOverlayVideo: " + e.getMessage());
//            // Delete potentially corrupt output file
//            new File(outputPath).delete();
//        } finally {
//            try {
//                if (writer != null) {
//                    writer.release();
//                }
//                if (cap != null && cap.isOpened()) {
//                    cap.release();
//                }
//                retriever.release();
//            } catch (Exception e) {
//                Log.e(TAG, "Error releasing resources: " + e.getMessage());
//            }
//        }
//
//        return heartRateDataList;
//    }
//
//    // rPPG using hand detection
//    private List<HeartRateData> processVideoAndExtractHeartRateFromHand(String recordId, String videoPath) {
//        Log.i(TAG, "Processing video with hand detection: " + videoPath);
//        VideoCapture cap = new VideoCapture(videoPath);
//        if (!cap.isOpened()) {
//            Log.e(TAG, "Failed to open video file: " + videoPath);
//            return new ArrayList<>();
//        }
//
//        // Get video properties and orientation
//        android.media.MediaMetadataRetriever retriever = new android.media.MediaMetadataRetriever();
//        try {
//            retriever.setDataSource(videoPath);
//        } catch (Exception e) {
//            Log.e(TAG, "Failed to set data source for retriever", e);
//            cap.release();
//            return new ArrayList<>();
//        }
//
//        String rotationString = retriever.extractMetadata(android.media.MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION);
//        int videoRotation = rotationString != null ? Integer.parseInt(rotationString) : 0;
//        try {
//            retriever.release();
//        } catch (Exception e) {
//            Log.w(TAG, "Error releasing retriever", e);
//        }
//        Log.d(TAG, "Video rotation: " + videoRotation);
//
//        double fps = cap.get(Videoio.CAP_PROP_FPS);
//        int frameWidth = (int) cap.get(Videoio.CAP_PROP_FRAME_WIDTH);
//        int frameHeight = (int) cap.get(Videoio.CAP_PROP_FRAME_HEIGHT);
//        int totalFrames = (int) cap.get(Videoio.CAP_PROP_FRAME_COUNT);
//
//        Log.d(TAG, String.format("Video properties: %dx%d, %.2f fps, %d frames", frameWidth, frameHeight, fps, totalFrames));
//
//        // If video is rotated 90 or 270 degrees, swap width and height for the output
//        Size finalFrameSize;
//        if (videoRotation == 90 || videoRotation == 270) {
//            finalFrameSize = new Size(frameHeight, frameWidth);
//        } else {
//            finalFrameSize = new Size(frameWidth, frameHeight);
//        }
//
//        // Setup video writer with correct orientation
//        VideoWriter writer = new VideoWriter();
//        String outputPath = new File(
//            context.getExternalFilesDir(null),
//            String.format("%s/%s/%s", AppConstant.DATA_DIR, recordId, AppConstant.FINAL_VIDEO_NAME)
//        ).getAbsolutePath();
//
//        // Create output directory
//        new File(outputPath).getParentFile().mkdirs();
//
//        // Configure video writer with more reliable codec settings
//        boolean writerOpened = false;
//        int[] codecs = {
//            VideoWriter.fourcc('m', 'p', '4', 'v'),  // Try MP4V first
//            VideoWriter.fourcc('H', '2', '6', '4'),
//            VideoWriter.fourcc('X', '2', '6', '4'),
//            VideoWriter.fourcc('x', '2', '6', '4')
//        };
//
//        for (int codec : codecs) {
//            if (writer.open(outputPath, codec, fps, finalFrameSize, true)) {
//                writerOpened = true;
//                Log.d(TAG, "Video writer opened successfully with codec: " + codec);
//                break;
//            }
//        }
//
//        if (!writerOpened) {
//            Log.e(TAG, "Failed to open VideoWriter with any codec");
//            cap.release();
//            return new ArrayList<>();
//        }
//
//        // Lists to store RGB signals
//        List<Double> redChannel = new ArrayList<>();
//        List<Double> greenChannel = new ArrayList<>();
//        List<Double> blueChannel = new ArrayList<>();
//        List<Long> timestamps = new ArrayList<>();
//
//        Mat frame = new Mat();
//        long startTime = System.currentTimeMillis();
//        int frameCount = 0;
//        double graphWidth = finalFrameSize.width * 0.3;
//        double graphHeight = finalFrameSize.height * 0.2;
//        int graphMaxPoints = 50;
//        List<Point> graphPoints = new ArrayList<>();
//        List<HeartRateData> heartRateDataList = new ArrayList<>();
//
//        // Heartbeat simulation for ECG
//        double heartbeatPhase = 0.0;
//        double baseHeartRate = 72.0; // Base BPM for simulation
//        List<Double> heartbeatTimes = new ArrayList<>();
//
//        // ECG visualization variables
//        double ecgWidth = finalFrameSize.width * 0.25;
//        double ecgHeight = finalFrameSize.height * 0.15;
//        int ecgMaxPoints = 100;
//        List<Point> ecgPoints = new ArrayList<>();
//
//        int framesWithHandDetection = 0;
//
//        try {
//            while (cap.read(frame)) {
//                if (frame.empty()) {
//                    Log.w(TAG, "Empty frame encountered");
//                    continue;
//                }
//
//                try {
//                    // Convert frame to correct type and color space
//                    Mat processedFrame = new Mat();
//                    frame.copyTo(processedFrame);
//
//                    // Handle rotation first
//                    Mat rotatedFrame = new Mat();
//                    switch (videoRotation) {
//                        case 90:
//                            Core.rotate(processedFrame, rotatedFrame, Core.ROTATE_90_CLOCKWISE);
//                            break;
//                        case 180:
//                            Core.rotate(processedFrame, rotatedFrame, Core.ROTATE_180);
//                            break;
//                        case 270:
//                            Core.rotate(processedFrame, rotatedFrame, Core.ROTATE_90_COUNTERCLOCKWISE);
//                            break;
//                        default:
//                            processedFrame.copyTo(rotatedFrame);
//                            break;
//                    }
//                    processedFrame.release();
//                    processedFrame = rotatedFrame;
//
//                    // Now handle color space conversions
//                    Imgproc.cvtColor(processedFrame, processedFrame, Imgproc.COLOR_BGR2RGB);
//
//                    // Detect face
//                    MatOfRect faces = new MatOfRect();
//                    Mat grayFrame = new Mat();
//                    Imgproc.cvtColor(processedFrame, grayFrame, Imgproc.COLOR_RGB2GRAY);
//                    if (faceDetector != null && !faceDetector.empty()) {
//                        faceDetector.detectMultiScale(grayFrame, faces);
//                    }
//
//                    // Draw face rectangles and forehead ROI
//                    for (Rect face : faces.toArray()) {
//                        Imgproc.rectangle(processedFrame, face, new Scalar(0, 255, 0), 2);
//                        int foreheadHeight = (int)(face.height * 0.3);
//                        Rect foreheadROI = new Rect(
//                            face.x + face.width/4,
//                            face.y,
//                            face.width/2,
//                            foreheadHeight
//                        );
//                        Imgproc.rectangle(processedFrame, foreheadROI, new Scalar(255, 0, 0), 2);
//
//                        // Extract color data from forehead ROI
//                        Mat roi = new Mat(processedFrame, foreheadROI);
//                        Scalar means = Core.mean(roi);
//                        redChannel.add(means.val[0]);
//                        greenChannel.add(means.val[1]);
//                        blueChannel.add(means.val[2]);
//
//                        long timestamp = startTime + (long)(frameCount * (1000.0 / fps));
//                        timestamps.add(timestamp);
//                        roi.release();
//                    }
//
//                    // Extract ROI from hand using MediaPipe-based detection with fallback
//                    MediaPipeHandTracker.HandDetectionResult handResult = null;
//                    try {
//                        handResult = handTracker.detectHand(originalFrame);
//                    } catch (Exception e) {
//                        Log.w(TAG, "MediaPipe hand detection failed, falling back to OpenCV", e);
//                    }
//
//                    // Fallback to OpenCV-based detection if MediaPipe fails
//                    if (handResult == null) {
//                        HandDetectionResult opencvResult = extractROIFromHandWithBounds(originalFrame);
//                        if (opencvResult != null) {
//                            handResult = new MediaPipeHandTracker.HandDetectionResult(
//                                opencvResult.boundingRect,
//                                opencvResult.roi,
//                                new Rect(
//                                    opencvResult.boundingRect.x + opencvResult.boundingRect.width / 4,
//                                    opencvResult.boundingRect.y + opencvResult.boundingRect.height / 4,
//                                    opencvResult.boundingRect.width / 2,
//                                    opencvResult.boundingRect.height / 2
//                                ),
//                                null // No landmarks for OpenCV detection
//                            );
//                        }
//                    }
//
//                    Rect palmROI = null;
//
//                    if (handResult != null) {
//                        framesWithHandDetection++;
//                        try {
//                            // Use the palm ROI provided by MediaPipe or calculated from OpenCV
//                            palmROI = handResult.palmROI;
//
//                            // Calculate mean color values from palm ROI for better signal quality
//                            Mat palmRegion = new Mat(originalFrame, palmROI);
//                            Mat rgb = new Mat();
//                            Imgproc.cvtColor(palmRegion, rgb, Imgproc.COLOR_BGR2RGB);
//
//                            // Apply additional skin color filtering for better signal
//                            Mat skinMask = new Mat();
//                            Core.inRange(rgb,
//                                new Scalar(95, 40, 20),   // Minimum skin color in RGB
//                                new Scalar(255, 220, 180), // Maximum skin color in RGB
//                                skinMask);
//
//                            // Apply mask to get only skin pixels
//                            Mat skinROI = new Mat();
//                            rgb.copyTo(skinROI, skinMask);
//
//                            // Calculate mean of skin pixels only
//                            Scalar means = Core.mean(skinROI, skinMask);
//
//                            // Only store values if we have sufficient skin pixels
//                            if (Core.countNonZero(skinMask) > skinMask.total() * 0.1) {
//                                redChannel.add(means.val[0]);
//                                greenChannel.add(means.val[1]);
//                                blueChannel.add(means.val[2]);
//
//                                // Calculate timestamp for this frame
//                                long timestamp = startTime + (long)(frameCount * (1000.0 / fps));
//                                timestamps.add(timestamp);
//                            }
//
//                            // Cleanup
//                            palmRegion.release();
//                            rgb.release();
//                            skinMask.release();
//                            skinROI.release();
//                        } finally {
//                            if (handResult.roi != null) {
//                                handResult.roi.release();
//                            }
//                        }
//
//                        // Draw hand annotations using MediaPipe tracker or simple rectangle for OpenCV
//                        if (handResult.handLandmarks != null) {
//                            handTracker.drawHandAnnotations(originalFrame, handResult);
//                        } else {
//                            // Draw simple annotations for OpenCV detection
//                            Imgproc.rectangle(originalFrame, handResult.boundingRect, new Scalar(0, 255, 0), 3);
//                            Imgproc.putText(originalFrame, "Hand (OpenCV)",
//                                new Point(handResult.boundingRect.x, handResult.boundingRect.y - 10),
//                                Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(0, 255, 0), 2);
//
//                            if (palmROI != null) {
//                                Imgproc.rectangle(originalFrame, palmROI, new Scalar(0, 0, 255), 2);
//                                Imgproc.putText(originalFrame, "Palm ROI",
//                                    new Point(palmROI.x, palmROI.y - 10),
//                                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 0, 255), 2);
//                            }
//                        }
//                    }
//                }
}