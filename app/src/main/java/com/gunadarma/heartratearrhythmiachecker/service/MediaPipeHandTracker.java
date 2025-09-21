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
 * Enhanced MediaPipe-based hand tracker with improved performance and reliability
 */
public class MediaPipeHandTracker {
    private static final String TAG = "MediaPipeHandTracker";

    private final Context context;
    private HandLandmarker handLandmarker;
    private boolean isInitialized = false;
    private long lastProcessedTimestamp = 0;

    // Optimized configuration for better performance
    private final Float MIN_HAND_DETECTION_CONFIDENCE = 0.7f; // Increased for better accuracy
    private final Float MIN_HAND_TRACKING_CONFIDENCE = 0.6f;  // Increased for stability
    private final Float MIN_HAND_PRESENCE_CONFIDENCE = 0.6f;  // Increased for reliability
    private final Integer MAX_NUM_HANDS = 1;
    private final RunningMode RUNNING_MODE = RunningMode.IMAGE; // Changed to IMAGE mode for sync processing

    // Performance tracking
    private int consecutiveFailures = 0;
    private static final int MAX_CONSECUTIVE_FAILURES = 5;

    public MediaPipeHandTracker(Context context) {
        this.context = context;
        initializeMediaPipe();
    }

    /**
     * Initialize MediaPipe HandLandmarker with enhanced configuration for better performance
     */
    private void initializeMediaPipe() {
        if (isInitialized) return;

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

            this.handLandmarker = HandLandmarker.createFromOptions(context, optionBuilder.build());
            this.isInitialized = true;
            Log.i(TAG, "MediaPipe HandLandmarker initialized successfully");
        } catch (Exception e) {
            Log.e(TAG, "Failed to initialize MediaPipe HandLandmarker", e);
            this.isInitialized = false;
        }
    }

    /**
     * Fallback initialization with CPU delegate and lower thresholds
     */
    private void tryFallbackInitialization() {
        try {
            Log.i(TAG, "Attempting fallback initialization with CPU delegate");

            var baseOptions = BaseOptions.builder()
                .setDelegate(Delegate.CPU)
                .setModelAssetPath("hand_landmarker.task")
                .build();

            var optionBuilder = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(baseOptions)
                .setMinHandDetectionConfidence(0.5f) // Lower threshold for fallback
                .setMinTrackingConfidence(0.5f)
                .setMinHandPresenceConfidence(0.5f)
                .setNumHands(MAX_NUM_HANDS)
                .setRunningMode(RUNNING_MODE);

            this.handLandmarker = HandLandmarker.createFromOptions(context, optionBuilder.build());
            this.isInitialized = true;
            this.consecutiveFailures = 0;
            Log.i(TAG, "Fallback MediaPipe HandLandmarker initialized successfully");

        } catch (Exception e) {
            Log.e(TAG, "Fallback initialization also failed", e);
            this.isInitialized = false;
        }
    }

    /**
     * Enhanced hand detection with improved error handling and performance
     */
    public HandDetectionResult detectHand(Mat frame) {
        if (!isInitialized || handLandmarker == null) {
            // Try to reinitialize if we've had too many failures
            if (consecutiveFailures > MAX_CONSECUTIVE_FAILURES) {
                Log.w(TAG, "Too many failures, attempting reinitialization");
                initializeMediaPipe();
            }
            return null;
        }

        try {
            // Convert OpenCV Mat to Bitmap with error checking
            if (frame.empty() || frame.cols() == 0 || frame.rows() == 0) {
                Log.w(TAG, "Invalid frame dimensions");
                return null;
            }

            Bitmap bitmap = Bitmap.createBitmap(frame.cols(), frame.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(frame, bitmap);

            // Create MPImage for MediaPipe processing
            MPImage image = new BitmapImageBuilder(bitmap).build();
            long timestamp = System.currentTimeMillis();

            // Process synchronously for more reliable results
            HandLandmarkerResult result = handLandmarker.detect(image);

            if (result != null && result.landmarks() != null && !result.landmarks().isEmpty()) {
                // Reset failure counter on success
                consecutiveFailures = 0;

                // Get the first (and best) hand landmarks
                var handLandmarks = result.landmarks().get(0);

                // Validate landmarks quality
                if (handLandmarks.size() < 21) {
                    Log.w(TAG, "Incomplete hand landmarks detected: " + handLandmarks.size());
                    return null;
                }

                // Calculate enhanced bounding box from landmarks
                Rect boundingRect = calculateEnhancedBoundingBox(handLandmarks, frame);
                if (boundingRect == null) {
                    return null;
                }

                // Extract ROI safely
                Mat roi = extractSafeROI(frame, boundingRect);
                if (roi == null) {
                    return null;
                }

                // Calculate optimized palm ROI
                Rect palmROI = calculateOptimizedPalmROI(handLandmarks, frame, boundingRect);

                Log.d(TAG, String.format("Hand detected: bounds(%d,%d,%d,%d), palm(%s)",
                    boundingRect.x, boundingRect.y, boundingRect.width, boundingRect.height,
                    palmROI != null ? String.format("%d,%d,%d,%d", palmROI.x, palmROI.y, palmROI.width, palmROI.height) : "null"));

                return new HandDetectionResult(boundingRect, roi, palmROI, handLandmarks);
            } else {
                consecutiveFailures++;
                if (consecutiveFailures % 10 == 0) {
                    Log.w(TAG, "No hand detected for " + consecutiveFailures + " consecutive frames");
                }
            }

        } catch (Exception e) {
            consecutiveFailures++;
            Log.e(TAG, "Error in enhanced hand detection (failures: " + consecutiveFailures + ")", e);

            // Reinitialize if too many failures
            if (consecutiveFailures > MAX_CONSECUTIVE_FAILURES) {
                Log.w(TAG, "Reinitializing due to consecutive failures");
                isInitialized = false;
                initializeMediaPipe();
            }
        }

        return null;
    }

    /**
     * Calculate enhanced bounding box with better landmark analysis
     */
    private Rect calculateEnhancedBoundingBox(java.util.List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark> handLandmarks, Mat frame) {
        try {
            float minX = Float.MAX_VALUE, minY = Float.MAX_VALUE;
            float maxX = Float.MIN_VALUE, maxY = Float.MIN_VALUE;

            // Analyze all landmarks for robust bounding box
            for (var landmark : handLandmarks) {
                float x = landmark.x() * frame.cols();
                float y = landmark.y() * frame.rows();

                minX = Math.min(minX, x);
                maxX = Math.max(maxX, x);
                minY = Math.min(minY, y);
                maxY = Math.max(maxY, y);
            }

            // Add adaptive padding based on hand size
            float handWidth = maxX - minX;
            float handHeight = maxY - minY;
            int paddingX = (int)(handWidth * 0.15); // 15% padding
            int paddingY = (int)(handHeight * 0.15);

            int rectX = Math.max(0, (int)(minX - paddingX));
            int rectY = Math.max(0, (int)(minY - paddingY));
            int width = Math.min((int)(maxX - minX + 2 * paddingX), frame.cols() - rectX);
            int height = Math.min((int)(maxY - minY + 2 * paddingY), frame.rows() - rectY);

            // Validate bounding box
            if (width < 50 || height < 50 || width > frame.cols() * 0.8 || height > frame.rows() * 0.8) {
                Log.w(TAG, "Invalid bounding box dimensions: " + width + "x" + height);
                return null;
            }

            return new Rect(rectX, rectY, width, height);

        } catch (Exception e) {
            Log.e(TAG, "Error calculating enhanced bounding box", e);
            return null;
        }
    }

    /**
     * Extract ROI safely with bounds checking
     */
    private Mat extractSafeROI(Mat frame, Rect boundingRect) {
        try {
            // Ensure bounding rect is within frame bounds
            int x = Math.max(0, Math.min(boundingRect.x, frame.cols() - 1));
            int y = Math.max(0, Math.min(boundingRect.y, frame.rows() - 1));
            int width = Math.min(boundingRect.width, frame.cols() - x);
            int height = Math.min(boundingRect.height, frame.rows() - y);

            if (width <= 0 || height <= 0) {
                Log.w(TAG, "Invalid ROI dimensions after bounds checking");
                return null;
            }

            Rect safeRect = new Rect(x, y, width, height);
            return new Mat(frame, safeRect);

        } catch (Exception e) {
            Log.e(TAG, "Error extracting safe ROI", e);
            return null;
        }
    }

    /**
     * Calculate optimized palm ROI using key landmarks
     */
    private Rect calculateOptimizedPalmROI(java.util.List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark> handLandmarks,
                                          Mat frame, Rect handBounds) {
        try {
            // Use specific palm landmarks for better accuracy
            // Wrist (0), Thumb CMC (1), Index MCP (5), Middle MCP (9), Ring MCP (13), Pinky MCP (17)
            int[] palmIndices = {0, 1, 5, 9, 13, 17};

            float palmCenterX = 0, palmCenterY = 0;
            int validLandmarks = 0;

            for (int index : palmIndices) {
                if (index < handLandmarks.size()) {
                    var landmark = handLandmarks.get(index);
                    palmCenterX += landmark.x() * frame.cols();
                    palmCenterY += landmark.y() * frame.rows();
                    validLandmarks++;
                }
            }

            if (validLandmarks == 0) {
                return createFallbackPalmROI(handLandmarks, frame, handBounds);
            }

            palmCenterX /= validLandmarks;
            palmCenterY /= validLandmarks;

            // Calculate palm size based on hand dimensions
            int palmSize = Math.min(handBounds.width, handBounds.height) / 2;
            palmSize = Math.max(60, Math.min(palmSize, 120)); // Clamp between 60-120 pixels

            int palmX = Math.max(handBounds.x, (int)(palmCenterX - palmSize / 2));
            int palmY = Math.max(handBounds.y, (int)(palmCenterY - palmSize / 2));
            int palmWidth = Math.min(palmSize, handBounds.x + handBounds.width - palmX);
            int palmHeight = Math.min(palmSize, handBounds.y + handBounds.height - palmY);

            // Validate palm ROI
            if (palmWidth < 40 || palmHeight < 40) {
                return createFallbackPalmROI(handLandmarks, frame, handBounds);
            }

            return new Rect(palmX, palmY, palmWidth, palmHeight);

        } catch (Exception e) {
            Log.w(TAG, "Error calculating optimized palm ROI, using fallback", e);
            return createFallbackPalmROI(handLandmarks, frame, handBounds);
        }
    }

    /**
     * Create a reliable fallback palm ROI
     */
    private Rect createFallbackPalmROI(java.util.List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark> handLandmarks,
                                      Mat frame, Rect handBounds) {
        try {
            // Use center of hand bounds as fallback
            int centerX = handBounds.x + handBounds.width / 2;
            int centerY = handBounds.y + handBounds.height / 2;

            // Create square ROI around center
            int roiSize = Math.min(handBounds.width, handBounds.height) / 3;
            roiSize = Math.max(50, Math.min(roiSize, 100)); // Clamp size

            int palmX = Math.max(handBounds.x, centerX - roiSize / 2);
            int palmY = Math.max(handBounds.y, centerY - roiSize / 2);
            int palmWidth = Math.min(roiSize, handBounds.x + handBounds.width - palmX);
            int palmHeight = Math.min(roiSize, handBounds.y + handBounds.height - palmY);

            return new Rect(palmX, palmY, palmWidth, palmHeight);

        } catch (Exception e) {
            Log.e(TAG, "Failed to create fallback palm ROI", e);
            return null;
        }
    }

    /**
     * Draw enhanced hand annotations with performance indicators
     */
    public void drawHandAnnotations(Mat frame, HandDetectionResult result) {
        if (result == null) return;

        try {
            // Draw hand bounding box with status color
            Scalar handColor = consecutiveFailures == 0 ? new Scalar(0, 255, 0) : new Scalar(0, 165, 255);
            Imgproc.rectangle(frame, result.boundingRect, handColor, 3);

            String handLabel = String.format("Hand (MP) - F:%d", consecutiveFailures);
            Imgproc.putText(frame, handLabel,
                new Point(result.boundingRect.x, result.boundingRect.y - 10),
                Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, handColor, 2);

            // Draw palm ROI with enhanced styling
            if (result.palmROI != null) {
                Imgproc.rectangle(frame, result.palmROI, new Scalar(0, 0, 255), 2);

                String palmLabel = String.format("Palm %dx%d", result.palmROI.width, result.palmROI.height);
                Imgproc.putText(frame, palmLabel,
                    new Point(result.palmROI.x, result.palmROI.y - 10),
                    Imgproc.FONT_HERSHEY_SIMPLEX, 0.4, new Scalar(0, 0, 255), 1);
            }

            // Draw key landmarks only (to reduce visual clutter)
            if (result.handLandmarks != null) {
                drawKeyLandmarks(frame, result.handLandmarks);
            }

        } catch (Exception e) {
            Log.w(TAG, "Error drawing enhanced hand annotations", e);
        }
    }

    /**
     * Draw only key landmarks for better performance and visibility
     */
    private void drawKeyLandmarks(Mat frame, java.util.List<com.google.mediapipe.tasks.components.containers.NormalizedLandmark> landmarks) {
        try {
            // Draw only key landmarks: wrist, fingertips, and palm base points
            int[] keyIndices = {0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17}; // Wrist, fingertips, palm bases

            for (int i : keyIndices) {
                if (i < landmarks.size()) {
                    var landmark = landmarks.get(i);
                    int x = (int)(landmark.x() * frame.cols());
                    int y = (int)(landmark.y() * frame.rows());

                    Scalar color = (i == 0) ? new Scalar(255, 0, 0) : new Scalar(0, 255, 255);
                    int radius = (i == 0) ? 5 : 3;

                    Imgproc.circle(frame, new Point(x, y), radius, color, -1);
                }
            }
        } catch (Exception e) {
            Log.w(TAG, "Error drawing key landmarks", e);
        }
    }

    /**
     * Release MediaPipe resources with proper cleanup
     */
    public void release() {
        try {
            if (handLandmarker != null) {
                handLandmarker.close();
                handLandmarker = null;
            }
            isInitialized = false;
            consecutiveFailures = 0;
            Log.i(TAG, "Enhanced MediaPipe HandLandmarker released");
        } catch (Exception e) {
            Log.e(TAG, "Error releasing MediaPipe resources", e);
        }
    }

    /**
     * Get current performance status
     */
    public boolean isPerformingWell() {
        return consecutiveFailures < MAX_CONSECUTIVE_FAILURES / 2;
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
}
