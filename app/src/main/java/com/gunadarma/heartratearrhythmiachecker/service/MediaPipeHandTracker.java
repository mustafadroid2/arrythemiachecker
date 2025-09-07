package com.gunadarma.heartratearrhythmiachecker.service;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import com.google.mediapipe.framework.AndroidAssetUtil;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.Delegate;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker.HandLandmarkerOptions;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

/**
 * MediaPipe-based hand tracker with proper initialization and fallback handling
 */
public class MediaPipeHandTracker {
    private static final String TAG = "MediaPipeHandTracker";

    private HandLandmarker handLandmarker;
    private final Context context;
    private boolean isInitialized = false;
    private HandLandmarkerResult lastResult = null;
    private long lastProcessedTimestamp = 0;
    private LandmarkerListener handLandmarkerHelperListener = null;

    public MediaPipeHandTracker(Context context) {
        this.context = context;
        initializeMediaPipe();
    }

    /**
     * Initialize MediaPipe HandLandmarker with proper configuration
     */
    private void initializeMediaPipe() {
        if (isInitialized) return;

        final Float MIN_HAND_DETECTION_CONFIDENCE = 0.5f;
        final Float MIN_HAND_TRACKING_CONFIDENCE = 0.5f;
        final Float MIN_HAND_PRESENCE_CONFIDENCE = 0.5f;
        final Integer MAX_NUM_HANDS = 1;
        final RunningMode RUNNING_MODE = RunningMode.VIDEO;

        try {
            // Initialize MediaPipe assets
            // AndroidAssetUtil.initializeNativeAssetManager(context);

            var baseOpetionBuilder = BaseOptions.builder();
            baseOpetionBuilder.setDelegate(Delegate.GPU);
            baseOpetionBuilder.setModelAssetPath("mediapipe/hand_landmarker.task");

            var baseOptions = baseOpetionBuilder.build();
            var optionBuilder = HandLandmarker.HandLandmarkerOptions.builder()
              .setBaseOptions(baseOptions)
              .setMinHandDetectionConfidence(MIN_HAND_DETECTION_CONFIDENCE)
              .setMinTrackingConfidence(MIN_HAND_TRACKING_CONFIDENCE)
              .setMinHandPresenceConfidence(MIN_HAND_PRESENCE_CONFIDENCE)
              .setNumHands(MAX_NUM_HANDS)
              .setRunningMode(RUNNING_MODE);

            // The ResultListener and ErrorListener only use for LIVE_STREAM mode.
            if (RUNNING_MODE == RunningMode.LIVE_STREAM) {
                optionBuilder
                  .setResultListener(this::returnLivestreamResult)
                  .setErrorListener(this::returnLivestreamError);
            }

            this.handLandmarker = HandLandmarker.createFromOptions(context, optionBuilder.build());
            this.isInitialized = true;
            Log.i(TAG, "MediaPipe HandLandmarker initialized successfully");
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize MediaPipe HandLandmarker", e);
            this.isInitialized = false;
        }
    }

    // Return the landmark result to this HandLandmarkerHelper's caller
    private void returnLivestreamResult(HandLandmarkerResult result, MPImage input) {
        // Cache the result for use in detectHand method
        this.lastResult = result;

        long finishTimeMs = android.os.SystemClock.uptimeMillis();
        long inferenceTime = finishTimeMs - result.timestampMs();

        if (handLandmarkerHelperListener != null) {
            handLandmarkerHelperListener.onResults(
              new ResultBundle(
                java.util.Collections.singletonList(result),
                inferenceTime,
                input.getHeight(),
                input.getWidth()
              )
            );
        }
    }

    // Return errors thrown during detection to this HandLandmarkerHelper's caller
    private void returnLivestreamError(RuntimeException error) {
        if (handLandmarkerHelperListener != null) {
            String message = (error.getMessage() != null) ? error.getMessage() : "An unknown error has occurred";
            handLandmarkerHelperListener.onError(message, 500);
        }
    }



    /**
     * Detect hand in the given frame and return hand detection result
     * Uses LIVE_STREAM mode with async processing and cached results
     */
    public HandDetectionResult detectHand(Mat frame) {
        if (!isInitialized || handLandmarker == null) {
            Log.w(TAG, "MediaPipe not initialized");
            return null;
        }

        try {
            // Convert OpenCV Mat to Bitmap
            Bitmap bitmap = Bitmap.createBitmap(frame.cols(), frame.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(frame, bitmap);

            // Create MPImage for MediaPipe processing
            MPImage image = new BitmapImageBuilder(bitmap).build();
            long timestamp = System.currentTimeMillis();

            // Only process if enough time has passed to avoid overwhelming the detector
            if (timestamp - lastProcessedTimestamp > 33) { // ~30 FPS max
                handLandmarker.detectAsync(image, timestamp);
                lastProcessedTimestamp = timestamp;
            }

            // Return result from the last successful detection (from the listener)
            if (lastResult != null && lastResult.landmarks() != null && !lastResult.landmarks().isEmpty()) {
                // Get the first hand landmarks
                var handLandmarks = lastResult.landmarks().get(0);

                // Calculate bounding box from landmarks
                float minX = Float.MAX_VALUE, minY = Float.MAX_VALUE;
                float maxX = Float.MIN_VALUE, maxY = Float.MIN_VALUE;

                for (var landmark : handLandmarks) {
                    float x = landmark.x() * frame.cols();
                    float y = landmark.y() * frame.rows();

                    minX = Math.min(minX, x);
                    maxX = Math.max(maxX, x);
                    minY = Math.min(minY, y);
                    maxY = Math.max(maxY, y);
                }

                // Add padding to bounding box
                int padding = 20;
                int rectX = Math.max(0, (int)(minX - padding));
                int rectY = Math.max(0, (int)(minY - padding));
                int width = Math.min((int)(maxX - minX + 2 * padding), frame.cols() - rectX);
                int height = Math.min((int)(maxY - minY + 2 * padding), frame.rows() - rectY);

                Rect boundingRect = new Rect(rectX, rectY, width, height);

                // Extract ROI
                Mat roi = new Mat(frame, boundingRect);

                // Calculate palm ROI
                Rect palmROI = calculatePalmROI(handLandmarks, frame, boundingRect);

                Log.d(TAG, "Hand detected using MediaPipe");
                return new HandDetectionResult(boundingRect, roi, palmROI, handLandmarks);
            }

        } catch (Exception e) {
            Log.e(TAG, "Error in hand detection", e);
        }

        return null;
    }

    /**
     * Calculate palm region of interest based on hand landmarks
     */
    private Rect calculatePalmROI(java.util.List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark> handLandmarks,
                                   Mat frame, Rect handBounds) {
        try {
            // Palm landmarks: wrist (0), thumb base (1), index base (5), middle base (9), ring base (13), pinky base (17)
            int[] palmLandmarkIndices = {0, 1, 5, 9, 13, 17};

            float minX = Float.MAX_VALUE, minY = Float.MAX_VALUE;
            float maxX = Float.MIN_VALUE, maxY = Float.MIN_VALUE;

            for (int index : palmLandmarkIndices) {
                if (index < handLandmarks.size()) {
                    var landmark = handLandmarks.get(index);
                    float x = landmark.x() * frame.cols();
                    float y = landmark.y() * frame.rows();

                    minX = Math.min(minX, x);
                    maxX = Math.max(maxX, x);
                    minY = Math.min(minY, y);
                    maxY = Math.max(maxY, y);
                }
            }

            // Create palm ROI with padding
            int palmPadding = 10;
            int palmX = Math.max(handBounds.x, (int)(minX - palmPadding));
            int palmY = Math.max(handBounds.y, (int)(minY - palmPadding));
            int palmWidth = Math.min((int)(maxX - minX + 2 * palmPadding),
                                   handBounds.x + handBounds.width - palmX);
            int palmHeight = Math.min((int)(maxY - minY + 2 * palmPadding),
                                    handBounds.y + handBounds.height - palmY);

            return new Rect(palmX, palmY, palmWidth, palmHeight);

        } catch (Exception e) {
            Log.w(TAG, "Could not calculate palm ROI, using fallback", e);
            // Fallback: use center 60% of hand bounding box
            int palmWidth = (int)(handBounds.width * 0.6);
            int palmHeight = (int)(handBounds.height * 0.6);
            int palmX = handBounds.x + (handBounds.width - palmWidth) / 2;
            int palmY = handBounds.y + (handBounds.height - palmHeight) / 2;
            return new Rect(palmX, palmY, palmWidth, palmHeight);
        }
    }

    /**
     * Draw hand landmarks and bounding boxes on the frame
     */
    public void drawHandAnnotations(Mat frame, HandDetectionResult result) {
        if (result == null) return;

        try {
            // Draw hand bounding box (green)
            Imgproc.rectangle(frame, result.boundingRect, new Scalar(0, 255, 0), 3);
            Imgproc.putText(frame, "Hand (MediaPipe)",
                new Point(result.boundingRect.x, result.boundingRect.y - 10),
                Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(0, 255, 0), 2);

            // Draw palm ROI (red)
            if (result.palmROI != null) {
                Imgproc.rectangle(frame, result.palmROI, new Scalar(0, 0, 255), 2);
                Imgproc.putText(frame, "Palm ROI",
                    new Point(result.palmROI.x, result.palmROI.y - 10),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 0, 255), 2);
            }

            // Draw hand landmarks if available
            if (result.handLandmarks != null) {
                drawHandLandmarks(frame, result.handLandmarks);
            }

        } catch (Exception e) {
            Log.w(TAG, "Error drawing hand annotations", e);
        }
    }

    /**
     * Draw individual hand landmarks
     */
    private void drawHandLandmarks(Mat frame, java.util.List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark> landmarks) {
        try {
            // Draw landmarks as small circles
            for (int i = 0; i < landmarks.size(); i++) {
                var landmark = landmarks.get(i);
                int x = (int)(landmark.x() * frame.cols());
                int y = (int)(landmark.y() * frame.rows());

                // Different colors for different types of landmarks
                Scalar color;
                if (i == 0) {
                    color = new Scalar(255, 0, 0); // Wrist - red
                } else if (i <= 4) {
                    color = new Scalar(255, 255, 0); // Thumb - yellow
                } else if (i <= 8) {
                    color = new Scalar(0, 255, 255); // Index - cyan
                } else if (i <= 12) {
                    color = new Scalar(255, 0, 255); // Middle - magenta
                } else if (i <= 16) {
                    color = new Scalar(0, 255, 0); // Ring - green
                } else {
                    color = new Scalar(0, 0, 255); // Pinky - blue
                }

                Imgproc.circle(frame, new Point(x, y), 3, color, -1);
            }
        } catch (Exception e) {
            Log.w(TAG, "Error drawing landmarks", e);
        }
    }

    /**
     * Release MediaPipe resources
     */
    public void release() {
        try {
            if (handLandmarker != null) {
                handLandmarker.close();
                handLandmarker = null;
            }
            isInitialized = false;
            Log.i(TAG, "MediaPipe HandLandmarker released");
        } catch (Exception e) {
            Log.e(TAG, "Error releasing MediaPipe resources", e);
        }
    }

    /**
     * Hand detection result with MediaPipe landmarks
     */
    public static class HandDetectionResult {
        public final Rect boundingRect;
        public final Mat roi;
        public final Rect palmROI;
        public final java.util.List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark> handLandmarks;

        public HandDetectionResult(Rect boundingRect, Mat roi, Rect palmROI,
                                 java.util.List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark> handLandmarks) {
            this.boundingRect = boundingRect;
            this.roi = roi;
            this.palmROI = palmROI;
            this.handLandmarks = handLandmarks;
        }
    }


    public class ResultBundle {
        public final java.util.List<HandLandmarkerResult> results;
        public final long inferenceTime;
        public final int inputImageHeight;
        public final int inputImageWidth;

        public ResultBundle(java.util.List<HandLandmarkerResult> results, long inferenceTime, int inputImageHeight, int inputImageWidth) {
            this.results = results;
            this.inferenceTime = inferenceTime;
            this.inputImageHeight = inputImageHeight;
            this.inputImageWidth = inputImageWidth;
        }
    }

    public interface LandmarkerListener {
        int OTHER_ERROR = -1; // Define a default error code

        void onError(String error, int errorCode);
        void onResults(ResultBundle resultBundle);
    }
}
