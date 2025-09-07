package com.gunadarma.heartratearrhythmiachecker.service.rppg;

import android.content.Context;
import android.media.MediaMetadataRetriever;
import android.util.Log;

import com.gunadarma.heartratearrhythmiachecker.constant.AppConstant;
import com.gunadarma.heartratearrhythmiachecker.model.RPPGData;
import com.gunadarma.heartratearrhythmiachecker.service.MediaPipeHandTracker;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.VideoWriter;
import org.opencv.videoio.Videoio;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

public class RPPGHandPalmServiceImpl implements RPPGService {
  private static final String TAG = "RPPGHandPalmService";
  private final MediaPipeHandTracker mpHandTracker;
  private final Context context;

  // rPPG processing constants for palm detection
  private static final double MIN_HR_BPM = 50.0;
  private static final double MAX_HR_BPM = 180.0;
  private static final double SKIN_CONFIDENCE_THRESHOLD = 0.5;
  private static final double PEAK_THRESHOLD = 0.5;
  private static final double BANDPASS_LOW_FREQ = 0.83; // 50 BPM in Hz
  private static final double BANDPASS_HIGH_FREQ = 0.83; // 50 BPM in Hz

  public RPPGHandPalmServiceImpl(Context context) {
    this.context = context;
    this.mpHandTracker = new MediaPipeHandTracker(context);
  }

  @Override
  public RPPGData getRPPGSignals(String videoPath) {
    Log.i(TAG, "Starting rPPG signal extraction from hand palm: " + videoPath);

    VideoCapture cap = new VideoCapture(videoPath);
    if (!cap.isOpened()) {
      Log.e(TAG, "Failed to open video file: " + videoPath);
      return RPPGData.empty();
    }

    try {
      // Get video properties
      double fps = AppConstant.OUTPUT_VIDEO_FPS;
      int totalFrames = (int) cap.get(Videoio.CAP_PROP_FRAME_COUNT);
      int videoDurationSeconds = (int) (totalFrames / fps);

      // Detect video rotation to maintain original aspect ratio
      int rotationDegrees = getVideoRotation(videoPath);
      boolean needsRotation = rotationDegrees == 90 || rotationDegrees == 270;

      Log.d(TAG, String.format("Video properties: %.2f fps, %d frames, %d seconds, rotation: %d°%s",
                               fps, totalFrames, videoDurationSeconds, rotationDegrees,
                               needsRotation ? " (needs correction)" : ""));

      // Extract RGB signals from palm region
      List<RPPGData.Signal> signals = extractPalmSignals(videoPath, cap, fps, rotationDegrees);

      if (signals.isEmpty()) {
        Log.w(TAG, "No valid palm signals extracted");
        return RPPGData.builder()
            .heartbeats(new ArrayList<>())
            .minBpm(0)
            .maxBpm(0)
            .averageBpm(0)
            .durationSeconds(videoDurationSeconds)
            .signals(new ArrayList<>())
            .build();
      }

      // Process signals to extract heart rate data
      RPPGProcessingResult result = processRPPGSignals(signals, fps);

      // Create and return RPPGData using builder pattern
      RPPGData rppgData = RPPGData.builder()
          .heartbeats(result.heartbeats)
          .minBpm(result.minBpm)
          .maxBpm(result.maxBpm)
          .averageBpm(result.averageBpm)
          .durationSeconds(videoDurationSeconds)
          .signals(signals)
          .build();

      Log.i(TAG, String.format("Palm rPPG extraction completed: %.1f BPM average, %d heartbeats detected",
                               result.averageBpm, result.heartbeats.size()));

      return rppgData;

    } finally {
      cap.release();
      if (mpHandTracker != null) {
        mpHandTracker.release();
      }
    }
  }

  private List<RPPGData.Signal> extractPalmSignals(String videoPath, VideoCapture cap, double fps, int rotationDegrees) {
    List<RPPGData.Signal> signals = new ArrayList<>();
    Mat frame = new Mat();
    long startTime = System.currentTimeMillis();
    int frameCount = 0;
    int validFrames = 0;
    int palmFrames = 0;

    // Data capture from hand palm
    // --------------------------------
    // - Overlay video spec
    //   -> Hand palm detection marked with green box
    //   -> ROI of hand palm between index and thumb finger red box
    //   -> ECG waveform at bottom
    //   -> FPS and frame count
    //   -> Heart rate BPM
    // - List of Heartbeat timestamps in ms
    // - Average, min, max BPM

    // Initialize video writer after reading first frame to get proper dimensions
    VideoWriter writer = null;
    String outputPath = videoPath.replace("original.mp4", "final.mp4");

    // Ensure output directory exists
    File outputFile = new File(outputPath);
    File outputDir = outputFile.getParentFile();
    if (outputDir != null && !outputDir.exists()) {
      boolean created = outputDir.mkdirs();
      Log.d(TAG, "Created output directory: " + outputDir.getAbsolutePath() + " - " + created);
    }

    // Try multiple codec options for better compatibility
    int[] codecOptions = {
      VideoWriter.fourcc('X', 'V', 'I', 'D'), // XVID - widely supported
      VideoWriter.fourcc('M', 'J', 'P', 'G'), // MJPEG - very compatible
      VideoWriter.fourcc('H', '2', '6', '4'), // H264 - modern standard
      VideoWriter.fourcc('m', 'p', '4', 'v'), // MP4V - original attempt
      0 // Default codec
    };

    // For ECG waveform visualization
    List<Double> ecgWaveform = new ArrayList<>();
    double currentBPM = 0.0;
    double lastSignalValue = 0.0;
    
    // ENSURE ONLY USE MEDIAPIPE FOR HAND DETECTION FOR CONSISTENCY
    while (cap.read(frame)) {
      if (frame.empty()) continue;

      // Apply rotation correction to maintain original aspect ratio
      if (rotationDegrees != 0) {
        frame = rotateFrame(frame, rotationDegrees);
      }

      // Initialize video writer with first frame dimensions (after rotation)
      if (writer == null) {
        Size frameSize = new Size(frame.cols(), frame.rows());

        // Try different codecs until one works
        boolean writerOpened = false;
        for (int fourcc : codecOptions) {
          try {
            writer = new VideoWriter(outputPath, fourcc, fps, frameSize, true);
            if (writer.isOpened()) {
              writerOpened = true;
              Log.i(TAG, "Video writer initialized successfully with fourcc: " + fourcc +
                         ", path: " + outputPath + ", size: " + frameSize.width + "x" + frameSize.height);
              break;
            } else {
              if (writer != null) {
                writer.release();
                writer = null;
              }
            }
          } catch (Exception e) {
            Log.w(TAG, "Failed to create writer with fourcc " + fourcc, e);
            if (writer != null) {
              writer.release();
              writer = null;
            }
          }
        }

        if (!writerOpened) {
          Log.e(TAG, "Failed to open video writer with any codec. Continuing without video output.");
          Log.e(TAG, "Output path: " + outputPath);
          Log.e(TAG, "Frame size: " + frameSize.width + "x" + frameSize.height);
          Log.e(TAG, "FPS: " + fps);
          // Continue processing without video output
        }
      }

      // Clone frame for overlay drawing
      Mat overlayFrame = frame.clone();

      try {
        // Try MediaPipe hand detection first
        MediaPipeHandTracker.HandDetectionResult handResult = null;
        if (mpHandTracker != null) {
          handResult = mpHandTracker.detectHand(frame);
        }

        if (handResult != null && handResult.palmROI != null && isValidPalmROI(handResult.palmROI, frame)) {
          palmFrames++;
          
          // Draw hand detection overlays using MediaPipe's method
          mpHandTracker.drawHandAnnotations(overlayFrame, handResult);
          
          Mat palmRegion = new Mat(frame, handResult.palmROI);
          Mat rgb = new Mat();
          Imgproc.cvtColor(palmRegion, rgb, Imgproc.COLOR_BGR2RGB);

          Mat skinMask = createPalmSkinMask(rgb);
          Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(3, 3));
          Imgproc.morphologyEx(skinMask, skinMask, Imgproc.MORPH_OPEN, kernel);
          Imgproc.morphologyEx(skinMask, skinMask, Imgproc.MORPH_CLOSE, kernel);

          Scalar means = Core.mean(rgb, skinMask);
          double skinRatio = (double) Core.countNonZero(skinMask) / skinMask.total();
          if (skinRatio > SKIN_CONFIDENCE_THRESHOLD) {
            long timestamp = startTime + (long) (frameCount * (1000.0 / fps));
            signals.add(new RPPGData.Signal(means.val[0], means.val[1], means.val[2], timestamp));
            validFrames++;
            
            // Calculate instantaneous signal value for ECG waveform
            lastSignalValue = (means.val[0] + means.val[1] + means.val[2]) / 3.0;
            ecgWaveform.add(lastSignalValue);
            
            // Calculate current BPM from recent signals (last 5 seconds)
            if (signals.size() > fps * 5) {
              currentBPM = calculateInstantaneousBPM(signals, (int)(fps * 5));
            }
          }

          palmRegion.release();
          rgb.release();
          skinMask.release();
          kernel.release();
        } else {
          // No hand detected - add zero to maintain timeline
          ecgWaveform.add(0.0);
        }
        
        // Draw all overlays on frame
        drawOverlays(overlayFrame, frameCount, fps, validFrames, palmFrames, currentBPM, ecgWaveform);
        
        // Write frame to output video only if writer is available
        if (writer != null && writer.isOpened()) {
          writer.write(overlayFrame);
        }

      } catch (Exception e) {
        Log.w(TAG, "Error processing frame " + frameCount, e);
        // Still write the frame even if processing failed
        drawOverlays(overlayFrame, frameCount, fps, validFrames, palmFrames, currentBPM, ecgWaveform);
        if (writer != null && writer.isOpened()) {
          writer.write(overlayFrame);
        }
      }

      overlayFrame.release();
      frameCount++;
      
      if (frameCount % 100 == 0) {
        Log.d(TAG, String.format("Processed %d frames, %d valid, %d with palm detection",
                                 frameCount, validFrames, palmFrames));
      }
    }

    if (writer != null) {
      writer.release();
      Log.i(TAG, "Overlay video saved to: " + outputPath);
    }
    
    Log.i(TAG, String.format("Extracted signals from %d/%d frames (%d palm frames)",
                             validFrames, frameCount, palmFrames));
    return signals;
  }

  private boolean isValidPalmROI(Rect palmROI, Mat frame) {
    // Check if ROI is within frame bounds
    if (palmROI.x < 0 || palmROI.y < 0 ||
        palmROI.x + palmROI.width > frame.cols() ||
        palmROI.y + palmROI.height > frame.rows()) {
      return false;
    }

    // Check minimum size requirements
    int minSize = Math.min(frame.cols(), frame.rows()) / 10;
    return palmROI.width >= minSize && palmROI.height >= minSize;
  }

  private Mat createPalmSkinMask(Mat rgb) {
    Mat skinMask = new Mat();

    // Convert to YCrCb color space for better skin detection
    Mat ycrcb = new Mat();
    Imgproc.cvtColor(rgb, ycrcb, Imgproc.COLOR_RGB2YCrCb);

    // Define skin color range in YCrCb (more robust for different lighting)
    Mat mask1 = new Mat();
    Core.inRange(ycrcb,
      new Scalar(0, 133, 77),   // Lower bound
      new Scalar(255, 173, 127), // Upper bound
      mask1);

    // Additional HSV-based skin detection for robustness
    Mat hsv = new Mat();
    Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV);

    Mat mask2 = new Mat();
    Core.inRange(hsv,
      new Scalar(0, 20, 70),   // Lower bound in HSV
      new Scalar(20, 255, 255), // Upper bound in HSV
      mask2);

    // Combine both masks
    Core.bitwise_and(mask1, mask2, skinMask);

    // Cleanup
    ycrcb.release();
    hsv.release();
    mask1.release();
    mask2.release();

    return skinMask;
  }

  private RPPGProcessingResult processRPPGSignals(List<RPPGData.Signal> signals, double fps) {
    if (signals.size() < fps * 8) { // Need at least 8 seconds of data for palm
      Log.w(TAG, "Insufficient signal data for processing");
      return new RPPGProcessingResult(new ArrayList<>(), 0, 0, 0);
    }

    // Convert to OpenCV Mats
    Mat redChannel = new Mat(signals.size(), 1, CvType.CV_64F);
    Mat greenChannel = new Mat(signals.size(), 1, CvType.CV_64F);
    Mat blueChannel = new Mat(signals.size(), 1, CvType.CV_64F);

    for (int i = 0; i < signals.size(); i++) {
      RPPGData.Signal signal = signals.get(i);
      redChannel.put(i, 0, signal.getRedChannel());
      greenChannel.put(i, 0, signal.getGreenChannel());
      blueChannel.put(i, 0, signal.getBlueChannel());
    }

    // Preprocess signals (detrend and normalize)
    Mat redProcessed = preprocessSignal(redChannel);
    Mat greenProcessed = preprocessSignal(greenChannel);
    Mat blueProcessed = preprocessSignal(blueChannel);

    // Apply CHROM algorithm for better palm rPPG signal extraction
    Mat chromSignal = applyCHROMAlgorithm(redProcessed, greenProcessed, blueProcessed);

    // Apply bandpass filter
    Mat filteredSignal = applyBandpassFilter(chromSignal, fps);

    // Extract heartbeats using peak detection with palm-specific parameters
    List<Long> heartbeats = extractHeartbeats(filteredSignal, signals, fps);

    // Calculate statistics
    double averageBpm = calculateAverageBPM(heartbeats);
    double minBpm = calculateMinBPM(heartbeats);
    double maxBpm = calculateMaxBPM(heartbeats);

    // Cleanup
    redChannel.release();
    greenChannel.release();
    blueChannel.release();
    redProcessed.release();
    greenProcessed.release();
    blueProcessed.release();
    chromSignal.release();
    filteredSignal.release();

    return new RPPGProcessingResult(heartbeats, minBpm, maxBpm, averageBpm);
  }

  private Mat applyCHROMAlgorithm(Mat red, Mat green, Mat blue) {
    // CHROM algorithm is more robust for palm-based rPPG
    // X_s = 3*R - 2*G
    // Y_s = 1.5*R + G - 1.5*B

    Mat Xs = new Mat();
    Mat Ys = new Mat();
    Mat temp1 = new Mat();
    Mat temp2 = new Mat();
    Mat temp3 = new Mat();

    // Calculate X_s = 3*R - 2*G
    Core.multiply(red, new Scalar(3.0), temp1);
    Core.multiply(green, new Scalar(2.0), temp2);
    Core.subtract(temp1, temp2, Xs);

    // Calculate Y_s = 1.5*R + G - 1.5*B
    Core.multiply(red, new Scalar(1.5), temp1);
    Core.multiply(blue, new Scalar(1.5), temp2);
    Core.add(temp1, green, temp3);
    Core.subtract(temp3, temp2, Ys);

    // Calculate alpha = std(X_s) / std(Y_s)
    MatOfDouble meanXs = new MatOfDouble();
    MatOfDouble stdXs = new MatOfDouble();
    Core.meanStdDev(Xs, meanXs, stdXs);

    MatOfDouble meanYs = new MatOfDouble();
    MatOfDouble stdYs = new MatOfDouble();
    Core.meanStdDev(Ys, meanYs, stdYs);

    double alpha = stdXs.get(0, 0)[0] / Math.max(stdYs.get(0, 0)[0], 1e-6);

    // S = X_s - alpha * Y_s
    Mat alphaMat = new Mat();
    Core.multiply(Ys, new Scalar(alpha), alphaMat);
    Mat chromSignal = new Mat();
    Core.subtract(Xs, alphaMat, chromSignal);

    // Cleanup
    Xs.release();
    Ys.release();
    temp1.release();
    temp2.release();
    temp3.release();
    alphaMat.release();

    return chromSignal;
  }

  private Mat preprocessSignal(Mat signal) {
    // Detrend signal by removing linear trend
    Mat detrended = detrendSignal(signal);

    // Apply smoothing filter to reduce noise
    Mat smoothed = new Mat();
    Imgproc.GaussianBlur(detrended, smoothed, new Size(5, 1), 1.0);

    // Normalize to [0, 1]
    Mat normalized = new Mat();
    Core.normalize(smoothed, normalized, 0, 1, Core.NORM_MINMAX);

    detrended.release();
    smoothed.release();
    return normalized;
  }

  private Mat detrendSignal(Mat signal) {
    Mat signalDouble = new Mat();
    signal.convertTo(signalDouble, CvType.CV_64F);

    // Create time vector
    Mat timeVector = new Mat(signal.rows(), 1, CvType.CV_64F);
    for (int i = 0; i < signal.rows(); i++) {
      timeVector.put(i, 0, (double)i);
    }

    // Fit linear trend using least squares
    Mat A = new Mat(signal.rows(), 2, CvType.CV_64F);
    for (int i = 0; i < signal.rows(); i++) {
      A.put(i, 0, 1.0); // Constant term
      A.put(i, 1, (double)i); // Linear term
    }

    Mat coefficients = new Mat();
    Core.solve(A, signalDouble, coefficients, Core.DECOMP_SVD);

    // Calculate trend line
    Mat trend = new Mat();
    Core.gemm(A, coefficients, 1.0, new Mat(), 0.0, trend);

    // Remove trend
    Mat detrended = new Mat();
    Core.subtract(signalDouble, trend, detrended);

    // Cleanup
    timeVector.release();
    A.release();
    coefficients.release();
    trend.release();

    return detrended;
  }

  private Mat applyBandpassFilter(Mat signal, double fps) {
    // Convert to frequency domain
    Mat signalFloat = new Mat();
    signal.convertTo(signalFloat, CvType.CV_32F);

    // Prepare for DFT
    int optimalRows = Core.getOptimalDFTSize(signalFloat.rows());
    Mat padded = new Mat();
    Core.copyMakeBorder(signalFloat, padded, 0, optimalRows - signalFloat.rows(), 0, 0, Core.BORDER_CONSTANT);

    // Convert to complex
    List<Mat> planes = new ArrayList<>();
    planes.add(padded);
    planes.add(Mat.zeros(padded.size(), CvType.CV_32F));
    Mat complexSignal = new Mat();
    Core.merge(planes, complexSignal);

    // Apply DFT
    Core.dft(complexSignal, complexSignal);

    // Create bandpass filter
    Mat filter = createBandpassFilter(optimalRows, fps);

    // Apply filter
    Mat filteredSpectrum = new Mat();
    Core.mulSpectrums(complexSignal, filter, filteredSpectrum, 0);

    // Inverse DFT
    Core.idft(filteredSpectrum, filteredSpectrum);

    // Extract real part
    Core.split(filteredSpectrum, planes);
    Mat result = planes.get(0).rowRange(0, signalFloat.rows());

    // Cleanup
    signalFloat.release();
    padded.release();
    complexSignal.release();
    filter.release();
    filteredSpectrum.release();

    return result;
  }

  private Mat createBandpassFilter(int size, double fps) {
    Mat filter = Mat.zeros(new Size(1, size), CvType.CV_32FC2);

    double lowCutNorm = BANDPASS_LOW_FREQ / fps;
    double highCutNorm = BANDPASS_HIGH_FREQ / fps;

    for (int i = 0; i < size; i++) {
      double freq = (double)i / size;
      if (freq > 0.5) freq = 1.0 - freq;

      float[] value = new float[2];
      if (freq >= lowCutNorm && freq <= highCutNorm) {
        value[0] = 1.0f; // Real part
        value[1] = 0.0f; // Imaginary part
      } else {
        value[0] = 0.0f;
        value[1] = 0.0f;
      }
      filter.put(i, 0, value);
    }

    return filter;
  }

  private List<Long> extractHeartbeats(Mat signal, List<RPPGData.Signal> originalSignals, double fps) {
    List<Long> heartbeats = new ArrayList<>();

    // Calculate signal statistics for peak detection
    MatOfDouble mean = new MatOfDouble();
    MatOfDouble stdDev = new MatOfDouble();
    Core.meanStdDev(signal, mean, stdDev);

    double threshold = mean.get(0, 0)[0] + stdDev.get(0, 0)[0] * PEAK_THRESHOLD;
    int minPeakDistance = (int)(0.35 * fps); // Minimum 0.35 seconds between peaks for palm

    // Find peaks
    List<Integer> peakIndices = new ArrayList<>();
    double[] prev = new double[1];
    double[] curr = new double[1];
    double[] next = new double[1];

    for (int i = 1; i < signal.rows() - 1; i++) {
      signal.get(i-1, 0, prev);
      signal.get(i, 0, curr);
      signal.get(i+1, 0, next);

      if (curr[0] > prev[0] && curr[0] > next[0] && curr[0] > threshold) {
        // Check minimum distance from last peak
        boolean validPeak = true;
        for (int peakIdx : peakIndices) {
          if (i - peakIdx < minPeakDistance) {
            validPeak = false;
            break;
          }
        }

        if (validPeak) {
          peakIndices.add(i);
        }
      }
    }

    // Convert peak indices to timestamps
    for (int peakIdx : peakIndices) {
      if (peakIdx < originalSignals.size()) {
        heartbeats.add(originalSignals.get(peakIdx).getTimestamp());
      }
    }

    return heartbeats;
  }

  private double calculateAverageBPM(List<Long> heartbeats) {
    if (heartbeats.size() < 2) return 0;

    List<Long> intervals = new ArrayList<>();
    for (int i = 1; i < heartbeats.size(); i++) {
      intervals.add(heartbeats.get(i) - heartbeats.get(i-1));
    }

    double avgInterval = intervals.stream().mapToLong(l -> l).average().orElse(0);
    return avgInterval > 0 ? 60000.0 / avgInterval : 0;
  }

  private double calculateMinBPM(List<Long> heartbeats) {
    if (heartbeats.size() < 2) return 0;

    List<Long> intervals = new ArrayList<>();
    for (int i = 1; i < heartbeats.size(); i++) {
      intervals.add(heartbeats.get(i) - heartbeats.get(i-1));
    }

    long maxInterval = Collections.max(intervals);
    return maxInterval > 0 ? 60000.0 / maxInterval : 0;
  }

  private double calculateMaxBPM(List<Long> heartbeats) {
    if (heartbeats.size() < 2) return 0;

    List<Long> intervals = new ArrayList<>();
    for (int i = 1; i < heartbeats.size(); i++) {
      intervals.add(heartbeats.get(i) - heartbeats.get(i-1));
    }

    long minInterval = Collections.min(intervals);
    return minInterval > 0 ? 60000.0 / minInterval : 0;
  }

  /**
   * Calculate instantaneous BPM from recent signals
   */
  private double calculateInstantaneousBPM(List<RPPGData.Signal> signals, int windowSize) {
    if (signals.size() < 2) return 0;
    
    int startIdx = Math.max(0, signals.size() - windowSize);
    List<RPPGData.Signal> recentSignals = signals.subList(startIdx, signals.size());
    
    if (recentSignals.size() < 2) return 0;
    
    // Simple peak counting for instantaneous BPM
    double sum = 0;
    for (RPPGData.Signal signal : recentSignals) {
      sum += (signal.getRedChannel() + signal.getGreenChannel() + signal.getBlueChannel()) / 3.0;
    }
    double mean = sum / recentSignals.size();
    
    int peaks = 0;
    boolean wasAboveMean = false;
    for (RPPGData.Signal signal : recentSignals) {
      double value = (signal.getRedChannel() + signal.getGreenChannel() + signal.getBlueChannel()) / 3.0;
      boolean isAboveMean = value > mean;
      
      if (isAboveMean && !wasAboveMean) {
        peaks++;
      }
      wasAboveMean = isAboveMean;
    }
    
    // Convert peaks to BPM
    double timeSpanSeconds = (recentSignals.get(recentSignals.size() - 1).getTimestamp() - 
                             recentSignals.get(0).getTimestamp()) / 1000.0;
    
    return timeSpanSeconds > 0 ? (peaks * 60.0) / timeSpanSeconds : 0;
  }

  /**
   * Draw all overlays on the frame including FPS, frame count, BPM, and ECG waveform
   */
  private void drawOverlays(Mat frame, int frameCount, double fps, int validFrames, 
                           int palmFrames, double currentBPM, List<Double> ecgWaveform) {
    try {
      // Text properties
      int fontFace = Imgproc.FONT_HERSHEY_SIMPLEX;
      double fontScale = 0.8;
      Scalar textColor = new Scalar(255, 255, 255); // White text
      Scalar bgColor = new Scalar(0, 0, 0); // Black background
      int thickness = 2;
      
      int frameHeight = frame.rows();
      int frameWidth = frame.cols();
      
      // Draw semi-transparent background for text
      Mat overlay = frame.clone();
      Imgproc.rectangle(overlay, 
        new org.opencv.core.Point(10, 10), 
        new org.opencv.core.Point(400, 150), 
        bgColor, -1);
      Core.addWeighted(frame, 0.7, overlay, 0.3, 0, frame);
      overlay.release();
      
      // Display FPS and frame information
      String fpsText = String.format(Locale.US, "FPS: %.1f", fps);
      String frameText = String.format(Locale.US, "Frame: %d", frameCount);
      String validText = String.format(Locale.US, "Valid: %d", validFrames);
      String palmText = String.format(Locale.US, "Palm: %d", palmFrames);
      String bpmText = String.format(Locale.US, "BPM: %.1f", currentBPM);
      
      Imgproc.putText(frame, fpsText, new org.opencv.core.Point(20, 40), 
                     fontFace, fontScale, textColor, thickness);
      Imgproc.putText(frame, frameText, new org.opencv.core.Point(20, 70), 
                     fontFace, fontScale, textColor, thickness);
      Imgproc.putText(frame, validText, new org.opencv.core.Point(20, 100), 
                     fontFace, fontScale, textColor, thickness);
      Imgproc.putText(frame, palmText, new org.opencv.core.Point(20, 130), 
                     fontFace, fontScale, textColor, thickness);
      
      // Display BPM in larger text at top right
      Imgproc.putText(frame, bpmText, new org.opencv.core.Point(frameWidth - 200, 50), 
                     fontFace, 1.2, new Scalar(0, 255, 0), 3); // Green BPM text
      
      // Draw ECG waveform at bottom of frame
      drawECGWaveform(frame, ecgWaveform, frameWidth, frameHeight);
      
    } catch (Exception e) {
      Log.w(TAG, "Error drawing overlays", e);
    }
  }

  /**
   * Draw ECG-style waveform at the bottom of the frame
   */
  private void drawECGWaveform(Mat frame, List<Double> waveform, int frameWidth, int frameHeight) {
    if (waveform.size() < 2) return;
    
    try {
      int waveformHeight = 100; // Height of waveform area
      int waveformY = frameHeight - waveformHeight - 20; // Y position of waveform baseline
      int maxPoints = Math.min(waveform.size(), frameWidth - 40); // Maximum points to display
      
      // Draw waveform background
      Imgproc.rectangle(frame, 
        new org.opencv.core.Point(20, waveformY - 50), 
        new org.opencv.core.Point(frameWidth - 20, frameHeight - 20),
        new Scalar(0, 0, 0, 128), -1); // Semi-transparent black
      
      // Draw grid lines
      Scalar gridColor = new Scalar(64, 64, 64); // Dark gray
      for (int i = 0; i < 5; i++) {
        int y = waveformY - 40 + (i * 20);
        Imgproc.line(frame, 
          new org.opencv.core.Point(20, y), 
          new org.opencv.core.Point(frameWidth - 20, y),
          gridColor, 1);
      }
      
      // Normalize waveform data
      double minVal = waveform.stream().mapToDouble(d -> d).min().orElse(0);
      double maxVal = waveform.stream().mapToDouble(d -> d).max().orElse(1);
      double range = Math.max(maxVal - minVal, 1e-6);
      
      // Draw waveform
      Scalar waveformColor = new Scalar(0, 255, 0); // Green waveform
      org.opencv.core.Point prevPoint = null;
      
      int startIdx = Math.max(0, waveform.size() - maxPoints);
      for (int i = startIdx; i < waveform.size(); i++) {
        double normalizedValue = (waveform.get(i) - minVal) / range;
        int x = 20 + (int)((double)(i - startIdx) * (frameWidth - 40) / maxPoints);
        int y = waveformY - (int)(normalizedValue * 80) + 40; // Scale to fit in waveform area
        
        org.opencv.core.Point currentPoint = new org.opencv.core.Point(x, y);
        
        if (prevPoint != null) {
          Imgproc.line(frame, prevPoint, currentPoint, waveformColor, 2);
        }
        prevPoint = currentPoint;
      }
      
      // Draw waveform label
      Imgproc.putText(frame, "rPPG Signal", 
        new org.opencv.core.Point(30, waveformY - 55), 
        Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(255, 255, 255), 2);
      
    } catch (Exception e) {
      Log.w(TAG, "Error drawing ECG waveform", e);
    }
  }

  /**
   * Result container for rPPG processing
   */
  private static class RPPGProcessingResult {
    final List<Long> heartbeats;
    final double minBpm;
    final double maxBpm;
    final double averageBpm;

    RPPGProcessingResult(List<Long> heartbeats, double minBpm, double maxBpm, double averageBpm) {
      this.heartbeats = heartbeats;
      this.minBpm = minBpm;
      this.maxBpm = maxBpm;
      this.averageBpm = averageBpm;
    }
  }

  /**
   * Get the rotation of the video in degrees using MediaMetadataRetriever
   */
  private int getVideoRotation(String videoPath) {
    MediaMetadataRetriever retriever = new MediaMetadataRetriever();
    try {
      retriever.setDataSource(videoPath);
      String rotation = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION);
      return rotation != null ? Integer.parseInt(rotation) : 0;
    } catch (Exception e) {
      Log.w(TAG, "Failed to get video rotation", e);
      return 0;
    } finally {
      try {
        retriever.release();
      } catch (Exception e) {
        Log.w(TAG, "Failed to release MediaMetadataRetriever", e);
      }
    }
  }

  /**
   * Rotate the frame to correct the orientation
   */
  private Mat rotateFrame(Mat frame, int rotationDegrees) {
    Mat rotated = new Mat();
    switch (rotationDegrees) {
      case 90:
        Core.transpose(frame, rotated);
        Core.flip(rotated, rotated, 1);
        break;
      case 180:
        Core.flip(frame, rotated, -1);
        break;
      case 270:
        Core.transpose(frame, rotated);
        Core.flip(rotated, rotated, 0);
        break;
      default:
        return frame; // No rotation needed
    }
    return rotated;
  }
}
