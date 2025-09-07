package com.gunadarma.heartratearrhythmiachecker.service;

import android.content.Context;
import android.util.Log;

import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

/**
 * Face tracker using OpenCV for face detection and forehead ROI extraction
 * This implementation provides a fallback when MediaPipe is not available
 */
public class MediaPipeFaceTracker {
    private static final String TAG = "MediaPipeFaceTracker";

    private final Context context;
    private CascadeClassifier faceDetector;
    private boolean isInitialized = false;

    public MediaPipeFaceTracker(Context context) {
        this.context = context;
        initializeFaceDetector();
    }

    private void initializeFaceDetector() {
        try {
            // Create temporary file for cascade classifier
            File cascadeDir = context.getDir("cascade", Context.MODE_PRIVATE);
            cascadeDir.mkdirs();
            File cascadeFile = new File(cascadeDir, "haarcascade_frontalface_alt.xml");

            // Copy cascade file from assets
            InputStream is = context.getAssets().open("haarcascades/haarcascade_frontalface_alt.xml");
            FileOutputStream os = new FileOutputStream(cascadeFile);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            // Create classifier
            faceDetector = new CascadeClassifier(cascadeFile.getAbsolutePath());

            // Cleanup
            cascadeFile.delete();
            cascadeDir.delete();

            if (faceDetector.empty()) {
                Log.e(TAG, "Failed to load cascade classifier");
                isInitialized = false;
                return;
            }

            isInitialized = true;
            Log.i(TAG, "OpenCV Face detector initialized successfully");
        } catch (Exception e) {
            Log.e(TAG, "Error initializing face detector", e);
            isInitialized = false;
        }
    }

    /**
     * Detect face in the given frame and return face detection result
     */
    public FaceDetectionResult detectFace(Mat frame) {
        if (!isInitialized || faceDetector == null || faceDetector.empty()) {
            Log.w(TAG, "Face detector not initialized");
            return null;
        }

        try {
            MatOfRect faces = new MatOfRect();
            Mat grayFrame = new Mat();
            Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);

            // Detect faces
            faceDetector.detectMultiScale(grayFrame, faces, 1.1, 3, 0, new Size(30, 30));

            Rect[] faceArray = faces.toArray();
            if (faceArray.length > 0) {
                // Use the largest face
                Rect face = getLargestFace(faceArray);

                // Add padding to face bounding box
                int padding = 20;
                int x = Math.max(0, face.x - padding);
                int y = Math.max(0, face.y - padding);
                int width = Math.min(face.width + 2 * padding, frame.cols() - x);
                int height = Math.min(face.height + 2 * padding, frame.rows() - y);

                Rect boundingRect = new Rect(x, y, width, height);

                // Extract ROI
                Mat roi = new Mat(frame, boundingRect);

                // Calculate forehead ROI (upper 30% of face, centered)
                Rect foreheadROI = calculateForeheadROI(face);

                grayFrame.release();
                return new FaceDetectionResult(boundingRect, roi, foreheadROI, null);
            }

            grayFrame.release();
        } catch (Exception e) {
            Log.e(TAG, "Error in face detection", e);
        }

        return null;
    }

    private Rect getLargestFace(Rect[] faces) {
        Rect largest = faces[0];
        int maxArea = largest.width * largest.height;

        for (int i = 1; i < faces.length; i++) {
            int area = faces[i].width * faces[i].height;
            if (area > maxArea) {
                maxArea = area;
                largest = faces[i];
            }
        }

        return largest;
    }

    /**
     * Calculate forehead region of interest for rPPG signal extraction
     */
    private Rect calculateForeheadROI(Rect face) {
        // Forehead is typically in the upper 30% of the face
        int foreheadHeight = (int)(face.height * 0.3);
        int foreheadWidth = (int)(face.width * 0.6);
        int foreheadX = face.x + (face.width - foreheadWidth) / 2;
        int foreheadY = face.y;

        return new Rect(foreheadX, foreheadY, foreheadWidth, foreheadHeight);
    }

    /**
     * Draw face landmarks and bounding boxes on the frame
     */
    public void drawFaceAnnotations(Mat frame, FaceDetectionResult result) {
        if (result == null) return;

        // Draw face bounding box
        Imgproc.rectangle(frame, result.boundingRect, new Scalar(0, 255, 0), 2);

        // Draw forehead ROI
        Imgproc.rectangle(frame, result.foreheadROI, new Scalar(0, 0, 255), 2);

        // Add labels
        Imgproc.putText(frame, "Face",
            new Point(result.boundingRect.x, result.boundingRect.y - 10),
            Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(0, 255, 0), 2);

        Imgproc.putText(frame, "Forehead ROI",
            new Point(result.foreheadROI.x, result.foreheadROI.y - 10),
            Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 0, 255), 2);
    }

    /**
     * Release face detector resources
     */
    public void release() {
        try {
            isInitialized = false;
            Log.i(TAG, "Face tracker resources released");
        } catch (Exception e) {
            Log.e(TAG, "Error releasing face tracker resources", e);
        }
    }

    /**
     * Face detection result containing bounding box, ROI, and forehead ROI
     */
    public static class FaceDetectionResult {
        public final Rect boundingRect;
        public final Mat roi;
        public final Rect foreheadROI;
        public final Object faceLandmarks; // Placeholder for future MediaPipe integration

        public FaceDetectionResult(Rect boundingRect, Mat roi, Rect foreheadROI, Object faceLandmarks) {
            this.boundingRect = boundingRect;
            this.roi = roi;
            this.foreheadROI = foreheadROI;
            this.faceLandmarks = faceLandmarks;
        }
    }
}
