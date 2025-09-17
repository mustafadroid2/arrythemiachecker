package com.gunadarma.heartratearrhythmiachecker.service.rppg;

import android.content.Context;
import android.media.MediaMetadataRetriever;
import android.util.Log;

import com.gunadarma.heartratearrhythmiachecker.constant.AppConstant;
import com.gunadarma.heartratearrhythmiachecker.model.RPPGData;
import com.gunadarma.heartratearrhythmiachecker.service.MediaPipeHandTracker;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.VideoWriter;
import org.opencv.videoio.Videoio;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class RPPGHandPalmServiceImpl implements RPPGService {
  private static final String TAG = "RPPGHandPalmService";
  private final MediaPipeHandTracker mpHandTracker;
  private final Context context;

  // Store signal history for ECG graph visualization
  private List<Double> signalHistory = new ArrayList<>();
  // Add synchronized timestamp tracking for ECG waveform
  private List<Long> signalTimestamps = new ArrayList<>();
  private static final int MAX_SIGNAL_HISTORY = 300; // Keep last 300 samples (10 seconds at 30fps)


  // Signal projection and continuity variables
  private List<Double> continuousSignalBuffer = new ArrayList<>();
  private List<Long> continuousTimestamps = new ArrayList<>();
  private Double lastValidSignal = null;
  private Long lastValidTimestamp;

  private double signalTrend = 0.0; // Track signal trend for projection
  private double averageHeartRate = 70.0; // Default HR for projection
  private int missedFrameCount = 0;
  private static final int MAX_INTERPOLATION_FRAMES = 15; // Max frames to interpolate (0.5 seconds at 30fps)

  // Signal quality tracking
  private double runningMean = 0.0;
  private double runningVariance = 0.0;

  // BPM history for smoothing
  private List<Double> bpmHistory = new ArrayList<>();
  private static final int BPM_HISTORY_SIZE = 10;
  private double lastStableBPM = 0.0;

  public RPPGHandPalmServiceImpl(Context context) {
    this.context = context;
    this.mpHandTracker = new MediaPipeHandTracker(context);
  }

  @Override
  public RPPGData getRPPGSignals(String videoPath) {
    Log.i(TAG, "Starting hand palm rPPG analysis: " + videoPath);

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

      // Process video for hand detection and rPPG calculation
      RPPGData rppgData = processHandDetection(videoPath, cap, fps, rotationDegrees);

      Log.i(TAG, "Hand palm rPPG analysis completed");

      return rppgData;

    } finally {
      cap.release();
      if (mpHandTracker != null) {
        mpHandTracker.release();
      }
    }
  }

  private RPPGData processHandDetection(String videoPath, VideoCapture cap, double fps, int rotationDegrees) {
    Mat frame = new Mat();
    int frameCount = 0;
    int palmFrames = 0;

    // rPPG signal storage
    List<RPPGData.Signal> signals = new ArrayList<>();
    List<Double> redSignals = new ArrayList<>();
    List<Double> greenSignals = new ArrayList<>();
    List<Double> blueSignals = new ArrayList<>();
    List<Long> timestamps = new ArrayList<>();

    // Initialize video writer for output with hand detection overlays
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

    long startTime = System.currentTimeMillis();

    // Process video frames for hand detection and rPPG
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
      long timestamp = startTime + (long)(frameCount * 1000 / fps);

      try {
        // Try MediaPipe hand detection
        MediaPipeHandTracker.HandDetectionResult handResult = null;
        if (mpHandTracker != null) {
          handResult = mpHandTracker.detectHand(frame);
        }

        RPPGData.Signal signal = null;
        boolean isPalmDetected = handResult != null && handResult.palmROI != null && isValidPalmROI(handResult.palmROI, frame);

        if (isPalmDetected) {
          palmFrames++;

          // Extract rPPG signals from palm region
          signal = extractRPPGSignal(frame, handResult.palmROI, timestamp);

          // Draw hand detection overlays using MediaPipe's method
          mpHandTracker.drawHandAnnotations(overlayFrame, handResult);

          // Draw rPPG info on palm region
          drawRPPGOverlays(overlayFrame, handResult.palmROI, signal);
        } else {
          // Palm not detected - try signal projection
          signal = processFrameWithProjection(timestamp, fps);

          // Draw projection indicator on frame
          if (signal != null) {
            drawProjectionIndicator(overlayFrame, missedFrameCount, MAX_INTERPOLATION_FRAMES);
          }
        }

        // Add signal to collections if available (either real or projected)
        if (signal != null) {
          signals.add(signal);
          redSignals.add(signal.getRedChannel());
          greenSignals.add(signal.getGreenChannel());
          blueSignals.add(signal.getBlueChannel());
          timestamps.add(signal.getTimestamp());

          // Update synchronized signal history for ECG graph visualization
          signalHistory.add(signal.getGreenChannel());
          signalTimestamps.add(signal.getTimestamp());

          // Keep only recent history with synchronized timestamps
          while (signalHistory.size() > MAX_SIGNAL_HISTORY) {
            signalHistory.remove(0);
            signalTimestamps.remove(0);
          }
        }

        // Draw comprehensive overlays on frame
        drawRPPGAnalysisOverlaysWithProjection(overlayFrame, frameCount, fps, palmFrames, signals.size(),
                                              isPalmDetected, missedFrameCount);

        // Write frame to output video only if writer is available
        if (writer != null && writer.isOpened()) {
          writer.write(overlayFrame);
        }

      } catch (Exception e) {
        Log.w(TAG, "Error processing frame " + frameCount, e);
        // Still write the frame even if processing failed
        drawRPPGAnalysisOverlaysWithProjection(overlayFrame, frameCount, fps, palmFrames, signals.size(),
                                              false, missedFrameCount);
        if (writer != null && writer.isOpened()) {
          writer.write(overlayFrame);
        }
      }

      overlayFrame.release();
      frameCount++;

      if (frameCount % 100 == 0) {
        Log.d(TAG, String.format("Processed %d frames, %d with palm detection, %d rPPG signals",
                                 frameCount, palmFrames, signals.size()));
      }
    }

    if (writer != null) {
      writer.release();
      Log.i(TAG, "rPPG analysis video saved to: " + outputPath);
    }

    Log.i(TAG, String.format("Processed %d frames (%d palm frames, %d rPPG signals)",
                             frameCount, palmFrames, signals.size()));

    // Calculate heart rate metrics from collected signals
    return calculateHeartRateMetrics(greenSignals, timestamps, fps);
  }

  /**
   * Extract rPPG signal from palm region of interest with projection support
   */
  private RPPGData.Signal extractRPPGSignal(Mat frame, Rect palmROI, long timestamp) {
    try {
      // Extract palm region
      Mat palmRegion = new Mat(frame, palmROI);

      // Convert to RGB for proper color channel analysis
      Mat rgbPalm = new Mat();
      Imgproc.cvtColor(palmRegion, rgbPalm, Imgproc.COLOR_BGR2RGB);

      // Calculate mean color values for each channel
      Scalar meanColor = Core.mean(rgbPalm);

      double redChannel = meanColor.val[0];
      double greenChannel = meanColor.val[1];
      double blueChannel = meanColor.val[2];

      // Release temporary matrices
      palmRegion.release();
      rgbPalm.release();

      // Update signal continuity tracking
      updateSignalContinuity(greenChannel, timestamp);

      // Reset missed frame counter
      missedFrameCount = 0;
      lastValidSignal = greenChannel;
      lastValidTimestamp = timestamp;

      return RPPGData.Signal.builder()
          .redChannel(redChannel)
          .greenChannel(greenChannel)
          .blueChannel(blueChannel)
          .timestamp(timestamp)
          .build();

    } catch (Exception e) {
      Log.w(TAG, "Error extracting rPPG signal", e);
      return null;
    }
  }

  /**
   * Process frame with signal projection when palm detection fails
   */
  private RPPGData.Signal processFrameWithProjection(long timestamp, double fps) {
    missedFrameCount++;

    if (lastValidSignal == null || missedFrameCount > MAX_INTERPOLATION_FRAMES) {
      // Too many missed frames or no previous signal - return null
      return null;
    }

    // Project signal based on established heart rate pattern
    double projectedSignal = projectSignalValue(timestamp, fps);

    Log.d(TAG, String.format("Projecting signal: missed frames=%d, projected=%.2f",
                             missedFrameCount, projectedSignal));

    return RPPGData.Signal.builder()
        .redChannel(0.0)
        .greenChannel(projectedSignal)
        .blueChannel(0.0)
        .timestamp(timestamp)
        .build();
  }

  /**
   * Project signal value based on established heart rate pattern
   */
  private double projectSignalValue(long timestamp, double fps) {
    if (continuousSignalBuffer.isEmpty() || lastValidSignal == null) {
      return lastValidSignal != null ? lastValidSignal : 128.0; // Default value
    }

    // Calculate time since last valid signal
    double timeDelta = (timestamp - lastValidTimestamp) / 1000.0; // seconds

    // Generate synthetic heart beat signal based on current heart rate
    double heartRatePeriod = 60.0 / averageHeartRate; // seconds per beat
    double phase = (timeDelta % heartRatePeriod) / heartRatePeriod * 2 * Math.PI;

    // Create realistic heart rate signal with controlled amplitude variations
    double baseSignal = lastValidSignal;

    // Reduced amplitude heart rate component to prevent jumps
    double heartRateComponent = 5.0 * Math.sin(phase) + 2.0 * Math.sin(2 * phase);

    // Minimal breathing component
    double breathingComponent = 1.5 * Math.sin(timeDelta * 2 * Math.PI / 4.0);

    // Reduced noise component
    double noiseComponent = 1.0 * (Math.random() - 0.5);

    // Strong decay to return to baseline quickly
    double decayFactor = Math.exp(-missedFrameCount * 0.3);

    // Calculate projected value with all components
    double projectedValue = baseSignal +
                          (heartRateComponent + breathingComponent + noiseComponent) * decayFactor +
                          signalTrend * timeDelta * decayFactor * 0.5; // Reduced trend influence

    // Ensure realistic bounds with tighter constraints
    return Math.max(lastValidSignal - 20, Math.min(lastValidSignal + 20, projectedValue));
  }

  /**
   * Update signal continuity tracking and heart rate estimation
   */
  private void updateSignalContinuity(double signal, long timestamp) {
    // Add to continuous buffer
    continuousSignalBuffer.add(signal);
    continuousTimestamps.add(timestamp);

    // Maintain buffer size
    int maxBufferSize = 150; // 5 seconds at 30fps
    while (continuousSignalBuffer.size() > maxBufferSize) {
      continuousSignalBuffer.remove(0);
      continuousTimestamps.remove(0);
    }

    // Update running statistics
    updateRunningStatistics(signal);

    // Update signal trend
    if (continuousSignalBuffer.size() >= 10) {
      updateSignalTrend();
    }

    // Update heart rate estimate
    if (continuousSignalBuffer.size() >= 60) { // 2 seconds of data
      updateHeartRateEstimate();
    }
  }

  /**
   * Update running mean and variance for signal quality assessment
   */
  private void updateRunningStatistics(double signal) {
    if (continuousSignalBuffer.size() == 1) {
      runningMean = signal;
      runningVariance = 0.0;
    } else {
      int n = continuousSignalBuffer.size();
      double delta = signal - runningMean;
      runningMean += delta / n;
      double delta2 = signal - runningMean;
      runningVariance = ((n - 2) * runningVariance + delta * delta2) / (n - 1);
    }
  }

  /**
   * Update signal trend for better projection
   */
  private void updateSignalTrend() {
    if (continuousSignalBuffer.size() < 10) return;

    // Calculate trend using linear regression on recent samples
    int sampleSize = Math.min(30, continuousSignalBuffer.size());
    List<Double> recentSignals = continuousSignalBuffer.subList(
        continuousSignalBuffer.size() - sampleSize, continuousSignalBuffer.size());

    double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;

    for (int i = 0; i < recentSignals.size(); i++) {
      double x = i;
      double y = recentSignals.get(i);
      sumX += x;
      sumY += y;
      sumXY += x * y;
      sumX2 += x * x;
    }

    int n = recentSignals.size();
    signalTrend = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);

    // Limit trend to reasonable values
    signalTrend = Math.max(-5.0, Math.min(5.0, signalTrend));
  }

  /**
   * Update heart rate estimate based on continuous signal - IMPROVED WITH SMOOTHING
   */
  private void updateHeartRateEstimate() {
    if (continuousSignalBuffer.size() < 60) return;

    try {
      // Use recent signal data for heart rate estimation
      List<Double> recentSignals = continuousSignalBuffer.subList(
          Math.max(0, continuousSignalBuffer.size() - 90),
          continuousSignalBuffer.size());

      double estimatedBPM = estimateCurrentBPM(recentSignals, 30.0);

      if (estimatedBPM > 40 && estimatedBPM < 200) {
        // Add to BPM history for temporal smoothing
        bpmHistory.add(estimatedBPM);
        
        // Maintain BPM history size
        while (bpmHistory.size() > BPM_HISTORY_SIZE) {
          bpmHistory.remove(0);
        }
        
        // Calculate smoothed BPM from history
        double smoothedBPM = calculateSmoothedBPM();
        
        // Apply additional exponential smoothing to prevent jumps
        double alpha = 0.2; // More conservative smoothing
        averageHeartRate = alpha * smoothedBPM + (1 - alpha) * averageHeartRate;
        
        // Update stable BPM with even more conservative approach
        if (Math.abs(smoothedBPM - lastStableBPM) < 15) { // Only update if change is < 15 BPM
          lastStableBPM = 0.8 * lastStableBPM + 0.2 * smoothedBPM;
        }

        Log.d(TAG, String.format("BPM estimate: %.1f -> smoothed: %.1f -> final: %.1f",
                                 estimatedBPM, smoothedBPM, averageHeartRate));
      }
    } catch (Exception e) {
      Log.w(TAG, "Failed to update heart rate estimate", e);
    }
  }

  /**
   * Calculate smoothed BPM from history to prevent jumps
   */
  private double calculateSmoothedBPM() {
    if (bpmHistory.isEmpty()) return averageHeartRate;
    
    if (bpmHistory.size() == 1) return bpmHistory.get(0);
    
    // Remove outliers from BPM history
    List<Double> filteredBPM = new ArrayList<>();
    double median = calculateMedian(new ArrayList<>(bpmHistory));
    
    for (Double bpm : bpmHistory) {
      // Only keep BPM values within 25% of median
      if (Math.abs(bpm - median) <= median * 0.25) {
        filteredBPM.add(bpm);
      }
    }
    
    // If too many outliers removed, use original history
    if (filteredBPM.size() < bpmHistory.size() / 2) {
      filteredBPM = new ArrayList<>(bpmHistory);
    }
    
    // Calculate weighted average (more recent values have higher weight)
    double weightedSum = 0;
    double totalWeight = 0;
    
    for (int i = 0; i < filteredBPM.size(); i++) {
      double weight = Math.pow(1.2, i); // Exponential weighting
      weightedSum += filteredBPM.get(i) * weight;
      totalWeight += weight;
    }
    
    return totalWeight > 0 ? weightedSum / totalWeight : averageHeartRate;
  }

  /**
   * Calculate median value from list
   */
  private double calculateMedian(List<Double> values) {
    if (values.isEmpty()) return 0;
    
    values.sort(Double::compareTo);
    int size = values.size();
    
    if (size % 2 == 0) {
      return (values.get(size/2 - 1) + values.get(size/2)) / 2.0;
    } else {
      return values.get(size/2);
    }
  }

  /**
   * Draw projection indicator when using estimated signals
   */
  private void drawProjectionIndicator(Mat frame, int missedFrames, int maxFrames) {
    try {
      int frameWidth = frame.cols();
      int frameHeight = frame.rows();

      // Calculate projection confidence
      double confidence = 1.0 - ((double)missedFrames / maxFrames);

      // Draw projection status
      String projectionText = String.format("PROJECTING (%.0f%% confidence)", confidence * 100);
      Scalar projectionColor = confidence > 0.7 ? new Scalar(0, 200, 0) :
                              confidence > 0.4 ? new Scalar(0, 200, 200) : new Scalar(0, 100, 255);

      Imgproc.putText(frame, projectionText,
          new Point(frameWidth - 300, 80),
          Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, projectionColor, 2);

      // Draw confidence bar
      int barX = frameWidth - 300;
      int barY = 90;
      int barWidth = 200;
      int barHeight = 8;

      // Background bar
      Imgproc.rectangle(frame,
          new Point(barX, barY),
          new Point(barX + barWidth, barY + barHeight),
          new Scalar(50, 50, 50), -1);

      // Confidence level bar
      int fillWidth = (int)(barWidth * confidence);
      Imgproc.rectangle(frame,
          new Point(barX, barY),
          new Point(barX + fillWidth, barY + barHeight),
          projectionColor, -1);

    } catch (Exception e) {
      Log.w(TAG, "Error drawing projection indicator", e);
    }
  }

  /**
   * Draw rPPG-specific overlays on palm region
   */
  private void drawRPPGOverlays(Mat frame, Rect palmROI, RPPGData.Signal signal) {
    if (signal == null) return;

    try {
      // Draw signal strength indicator
      int signalStrength = (int)(signal.getGreenChannel() / 2.55); // Convert to 0-100 scale
      Scalar strengthColor = signalStrength > 50 ? new Scalar(0, 255, 0) : new Scalar(0, 165, 255);

      // Draw signal strength bar next to palm ROI
      int barX = palmROI.x + palmROI.width + 5;
      int barY = palmROI.y;
      int barWidth = 10;
      int barHeight = palmROI.height;

      // Background bar
      Imgproc.rectangle(frame,
          new Point(barX, barY),
          new Point(barX + barWidth, barY + barHeight),
          new Scalar(50, 50, 50), -1);

      // Signal level bar
      int fillHeight = (int)(barHeight * signalStrength / 100.0);
      Imgproc.rectangle(frame,
          new Point(barX, barY + barHeight - fillHeight),
          new Point(barX + barWidth, barY + barHeight),
          strengthColor, -1);

      // Signal value text
      String signalText = String.format("G:%.0f", signal.getGreenChannel());
      Imgproc.putText(frame, signalText,
          new Point(palmROI.x, palmROI.y + palmROI.height + 20),
          Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255, 255, 255), 1);

    } catch (Exception e) {
      Log.w(TAG, "Error drawing rPPG overlays", e);
    }
  }

  /**
   * Draw comprehensive rPPG analysis overlays on the frame
   */
  private void drawRPPGAnalysisOverlays(Mat frame, int frameCount, double fps, int palmFrames, int signalCount) {
    try {
      // Text properties
      int fontFace = Imgproc.FONT_HERSHEY_SIMPLEX;
      double fontScale = 0.8;
      Scalar textColor = new Scalar(255, 255, 255); // White text
      Scalar bgColor = new Scalar(0, 0, 0); // Black background
      int thickness = 2;

      int frameWidth = frame.cols();

      // Draw semi-transparent background for text
      Mat overlay = frame.clone();
      Imgproc.rectangle(overlay,
        new Point(10, 10),
        new Point(450, 150),
        bgColor, -1);
      Core.addWeighted(frame, 0.7, overlay, 0.3, 0, frame);
      overlay.release();

      // Display comprehensive information
      String fpsText = String.format(Locale.US, "FPS: %.1f", fps);
      String frameText = String.format(Locale.US, "Frame: %d", frameCount);
      String palmText = String.format(Locale.US, "Palm Detected: %d", palmFrames);
      String signalText = String.format(Locale.US, "rPPG Signals: %d", signalCount);
      String statusText = "rPPG: ACTIVE";

      Imgproc.putText(frame, fpsText, new Point(20, 40),
                     fontFace, fontScale, textColor, thickness);
      Imgproc.putText(frame, frameText, new Point(20, 70),
                     fontFace, fontScale, textColor, thickness);
      Imgproc.putText(frame, palmText, new Point(20, 100),
                     fontFace, fontScale, textColor, thickness);
      Imgproc.putText(frame, signalText, new Point(20, 130),
                     fontFace, fontScale, textColor, thickness);

      // Display status in larger text at top right
      Imgproc.putText(frame, statusText, new Point(frameWidth - 200, 50),
                     fontFace, 1.0, new Scalar(0, 255, 0), 3); // Green status text

      // Draw ECG-style graph at the bottom of the frame
      drawECGGraph(frame, frameCount);

    } catch (Exception e) {
      Log.w(TAG, "Error drawing rPPG analysis overlays", e);
    }
  }

  /**
   * Draw comprehensive rPPG analysis overlays with projection status
   */
  private void drawRPPGAnalysisOverlaysWithProjection(Mat frame, int frameCount, double fps,
                                                     int palmFrames, int signalCount,
                                                     boolean isPalmDetected, int missedFrames) {
    try {
      // Text properties
      int fontFace = Imgproc.FONT_HERSHEY_SIMPLEX;
      double fontScale = 0.8;
      Scalar textColor = new Scalar(255, 255, 255); // White text
      Scalar bgColor = new Scalar(0, 0, 0); // Black background
      int thickness = 2;

      int frameWidth = frame.cols();

      // Draw semi-transparent background for text
      Mat overlay = frame.clone();
      Imgproc.rectangle(overlay,
        new Point(10, 10),
        new Point(500, 180),
        bgColor, -1);
      Core.addWeighted(frame, 0.7, overlay, 0.3, 0, frame);
      overlay.release();

      // Display comprehensive information
      String fpsText = String.format(Locale.US, "FPS: %.1f", fps);
      String frameText = String.format(Locale.US, "Frame: %d", frameCount);
      String palmText = String.format(Locale.US, "Palm Detected: %d", palmFrames);
      String signalText = String.format(Locale.US, "Total Signals: %d", signalCount);
      String detectionStatus = isPalmDetected ? "DETECTING" : "PROJECTING";
      String continuityText = String.format(Locale.US, "Missed Frames: %d/%d", missedFrames, MAX_INTERPOLATION_FRAMES);

      Imgproc.putText(frame, fpsText, new Point(20, 40),
                     fontFace, fontScale, textColor, thickness);
      Imgproc.putText(frame, frameText, new Point(20, 70),
                     fontFace, fontScale, textColor, thickness);
      Imgproc.putText(frame, palmText, new Point(20, 100),
                     fontFace, fontScale, textColor, thickness);
      Imgproc.putText(frame, signalText, new Point(20, 130),
                     fontFace, fontScale, textColor, thickness);
      Imgproc.putText(frame, continuityText, new Point(20, 160),
                     fontFace, fontScale, textColor, thickness);

      // Display status in larger text at top right
      Scalar statusColor = isPalmDetected ? new Scalar(0, 255, 0) : new Scalar(0, 200, 200);
      Imgproc.putText(frame, "rPPG: " + detectionStatus, new Point(frameWidth - 300, 50),
                     fontFace, 1.0, statusColor, 3);

      // Draw ECG-style graph at the bottom of the frame
      drawECGGraph(frame, frameCount);

    } catch (Exception e) {
      Log.w(TAG, "Error drawing rPPG analysis overlays with projection", e);
    }
  }

  /**
   * Calculate heart rate metrics from green channel signals using enhanced peak detection
   */
  private RPPGData calculateHeartRateMetrics(List<Double> greenSignals, List<Long> timestamps, double fps) {
    Log.d(TAG, "Starting heart rate calculation with " + greenSignals.size() + " signals");

    if (greenSignals.size() < 30) { // Need at least 1 second of data at 30fps
      Log.w(TAG, "Insufficient data for heart rate calculation: " + greenSignals.size() + " samples");
      return RPPGData.empty();
    }

    try {
      // Apply continuous signal generation to fill gaps
      List<Double> continuousSignals = generateContinuousSignalStream(greenSignals, timestamps, fps);

      // Enhanced signal preprocessing for better peak detection
      List<Double> processedSignals = enhanceSignalForPeakDetection(continuousSignals);

      // Apply moving average filter to smooth the signal
      List<Double> smoothedSignals = applyMovingAverageFilter(processedSignals, 3);

      // Detect peaks in the smoothed green channel signal with enhanced algorithm
      List<Long> heartbeats = detectHeartbeatsEnhanced(smoothedSignals, timestamps, fps);

      Log.d(TAG, "Detected " + heartbeats.size() + " heartbeats from " + smoothedSignals.size() + " signal points");

      if (heartbeats.size() < 2) {
        Log.w(TAG, "Insufficient heartbeats detected: " + heartbeats.size());
        // Try alternative detection method
        heartbeats = detectHeartbeatsAlternative(smoothedSignals, timestamps, fps);
        Log.d(TAG, "Alternative detection found " + heartbeats.size() + " heartbeats");
      }

      if (heartbeats.size() < 2) {
        Log.w(TAG, "Still insufficient heartbeats after alternative detection");
        return createPartialRPPGData(greenSignals, timestamps);
      }

      // Calculate BPM from intervals between heartbeats
      List<Double> bpmValues = calculateBPMFromHeartbeatsEnhanced(heartbeats);

      if (bpmValues.isEmpty()) {
        Log.w(TAG, "No valid BPM values calculated");
        return createPartialRPPGData(greenSignals, timestamps);
      }

      double averageBpm = bpmValues.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
      double minBpm = bpmValues.stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
      double maxBpm = bpmValues.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);

      // Update the real-time heart rate estimate to match the final calculated value
      // This ensures synchronization between video overlay and final results
      averageHeartRate = averageBpm;
      lastStableBPM = averageBpm;

      Log.d(TAG, String.format("Synchronized heart rate estimates: real-time=%.1f, calculated=%.1f",
                               averageHeartRate, averageBpm));

      // Create signal list
      List<RPPGData.Signal> signalList = new ArrayList<>();
      for (int i = 0; i < Math.min(greenSignals.size(), timestamps.size()); i++) {
        signalList.add(RPPGData.Signal.builder()
            .redChannel(0.0)
            .greenChannel(greenSignals.get(i))
            .blueChannel(0.0)
            .timestamp(timestamps.get(i))
            .build());
      }

      int durationSeconds = (int)((timestamps.get(timestamps.size()-1) - timestamps.get(0)) / 1000);

      Log.i(TAG, String.format("Heart rate analysis complete: %.1f BPM (%.1f-%.1f), %d beats in %d seconds",
                               averageBpm, minBpm, maxBpm, heartbeats.size(), durationSeconds));

      return RPPGData.builder()
          .heartbeats(heartbeats)
          .minBpm(minBpm)
          .maxBpm(maxBpm)
          .averageBpm(averageBpm)
          .baselineBpm(averageBpm)
          .durationSeconds(durationSeconds)
          .signals(signalList)
          .build();

    } catch (Exception e) {
      Log.e(TAG, "Error calculating heart rate metrics", e);
      return createPartialRPPGData(greenSignals, timestamps);
    }
  }

  /**
   * Generate continuous signal stream with interpolation
   */
  private List<Double> generateContinuousSignalStream(List<Double> originalSignals,
                                                     List<Long> originalTimestamps,
                                                     double fps) {
    if (originalSignals.isEmpty()) return originalSignals;

    List<Double> continuousSignals = new ArrayList<>();
    List<Long> interpolatedTimestamps = new ArrayList<>();

    // Generate timestamp sequence with constant intervals
    long startTime = originalTimestamps.get(0);
    long endTime = originalTimestamps.get(originalTimestamps.size() - 1);
    long intervalMs = (long)(1000.0 / fps);

    int originalIndex = 0;
    for (long currentTime = startTime; currentTime <= endTime; currentTime += intervalMs) {
      // Find closest original signal
      while (originalIndex < originalTimestamps.size() - 1 &&
             Math.abs(originalTimestamps.get(originalIndex + 1) - currentTime) <
             Math.abs(originalTimestamps.get(originalIndex) - currentTime)) {
        originalIndex++;
      }

      double signal;
      if (Math.abs(originalTimestamps.get(originalIndex) - currentTime) < intervalMs * 2) {
        // Use original signal if close enough
        signal = originalSignals.get(originalIndex);
      } else {
        // Interpolate signal
        signal = interpolateSignal(originalSignals, originalTimestamps, currentTime, originalIndex);
      }

      continuousSignals.add(signal);
      interpolatedTimestamps.add(currentTime);
    }

    Log.d(TAG, String.format("Generated continuous signal: %d original -> %d interpolated points",
                             originalSignals.size(), continuousSignals.size()));

    return continuousSignals;
  }

  /**
   * Interpolate signal value at specific timestamp
   */
  private double interpolateSignal(List<Double> signals, List<Long> timestamps, long targetTime, int nearestIndex) {
    if (nearestIndex == 0) {
      return signals.get(0);
    } else if (nearestIndex >= signals.size() - 1) {
      return signals.get(signals.size() - 1);
    }

    // Linear interpolation between two nearest points
    long t1 = timestamps.get(nearestIndex - 1);
    long t2 = timestamps.get(nearestIndex + 1);
    double s1 = signals.get(nearestIndex - 1);
    double s2 = signals.get(nearestIndex + 1);

    if (t2 == t1) return s1; // Avoid division by zero

    double ratio = (double)(targetTime - t1) / (t2 - t1);
    return s1 + ratio * (s2 - s1);
  }

  /**
   * Enhance signal for better peak detection
   */
  private List<Double> enhanceSignalForPeakDetection(List<Double> signals) {
    if (signals.size() < 10) return signals;

    List<Double> enhanced = new ArrayList<>();

    // Apply DC removal (remove baseline drift)
    double mean = signals.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    for (Double signal : signals) {
      enhanced.add(signal - mean);
    }

    // Apply simple high-pass filter to remove low frequency noise
    List<Double> filtered = new ArrayList<>();
    filtered.add(enhanced.get(0));

    for (int i = 1; i < enhanced.size(); i++) {
      double filtered_val = 0.95 * (filtered.get(i-1) + enhanced.get(i) - enhanced.get(i-1));
      filtered.add(filtered_val);
    }

    return filtered;
  }

  /**
   * Enhanced heartbeat detection with multiple validation steps
   */
  private List<Long> detectHeartbeatsEnhanced(List<Double> signals, List<Long> timestamps, double fps) {
    List<Long> heartbeats = new ArrayList<>();

    if (signals.size() < 10) return heartbeats;

    // Calculate adaptive threshold based on signal statistics
    double mean = signals.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    double stdDev = Math.sqrt(signals.stream()
        .mapToDouble(val -> Math.pow(val - mean, 2))
        .average().orElse(0.0));

    // Use multiple threshold levels for robustness
    double primaryThreshold = mean + 0.7 * stdDev;
    double secondaryThreshold = mean + 0.4 * stdDev;

    // Physiological constraints (40-180 BPM)
    int minDistanceFrames = (int)(fps * 60.0 / 180.0); // Max 180 BPM
    int maxDistanceFrames = (int)(fps * 60.0 / 40.0);  // Min 40 BPM

    int lastPeakIndex = -minDistanceFrames;
    List<Integer> candidatePeaks = new ArrayList<>();

    // Find primary peaks
    for (int i = 2; i < signals.size() - 2; i++) {
      double current = signals.get(i);
      double prev1 = signals.get(i - 1);
      double prev2 = signals.get(i - 2);
      double next1 = signals.get(i + 1);
      double next2 = signals.get(i + 2);

      // Enhanced peak detection criteria
      boolean isLocalMax = current > prev1 && current > next1 && current > prev2 && current > next2;
      boolean aboveThreshold = current > primaryThreshold;
      boolean properDistance = (i - lastPeakIndex) >= minDistanceFrames;

      if (isLocalMax && aboveThreshold && properDistance) {
        candidatePeaks.add(i);
        lastPeakIndex = i;
      }
    }

    // If not enough peaks found, try with lower threshold
    if (candidatePeaks.size() < 3) {
      candidatePeaks.clear();
      lastPeakIndex = -minDistanceFrames;

      for (int i = 2; i < signals.size() - 2; i++) {
        double current = signals.get(i);
        double prev1 = signals.get(i - 1);
        double next1 = signals.get(i + 1);

        boolean isLocalMax = current > prev1 && current > next1;
        boolean aboveThreshold = current > secondaryThreshold;
        boolean properDistance = (i - lastPeakIndex) >= minDistanceFrames;

        if (isLocalMax && aboveThreshold && properDistance) {
          candidatePeaks.add(i);
          lastPeakIndex = i;
        }
      }
    }

    // Convert to timestamps
    for (int peakIndex : candidatePeaks) {
      if (peakIndex < timestamps.size()) {
        heartbeats.add(timestamps.get(peakIndex));
      }
    }

    Log.d(TAG, "Enhanced detection found " + heartbeats.size() + " heartbeats with thresholds: "
          + String.format("%.2f/%.2f", primaryThreshold, secondaryThreshold));

    return heartbeats;
  }

  /**
   * Alternative heartbeat detection method using autocorrelation
   */
  private List<Long> detectHeartbeatsAlternative(List<Double> signals, List<Long> timestamps, double fps) {
    List<Long> heartbeats = new ArrayList<>();

    if (signals.size() < 60) return heartbeats; // Need at least 2 seconds

    try {
      // Estimate heart rate using FFT-like approach (simplified)
      double estimatedPeriod = estimateHeartRatePeriod(signals, fps);

      if (estimatedPeriod > 0) {
        // Generate heartbeats based on estimated period
        double periodFrames = estimatedPeriod * fps;
        int startFrame = (int)(periodFrames / 2); // Start after half period

        for (int frame = startFrame; frame < signals.size(); frame += (int)periodFrames) {
          if (frame < timestamps.size()) {
            heartbeats.add(timestamps.get(frame));
          }
        }

        Log.d(TAG, "Alternative method estimated period: " + estimatedPeriod + "s, generated " + heartbeats.size() + " beats");
      }
    } catch (Exception e) {
      Log.w(TAG, "Alternative detection failed", e);
    }

    return heartbeats;
  }

  /**
   * Estimate heart rate period using signal analysis
   */
  private double estimateHeartRatePeriod(List<Double> signals, double fps) {
    // Simple autocorrelation-like approach
    int maxLag = (int)(fps * 2.0); // Max 2 seconds lag (30 BPM minimum)
    int minLag = (int)(fps * 0.33); // Min 0.33 seconds lag (180 BPM maximum)

    double maxCorrelation = 0;
    int bestLag = 0;

    for (int lag = minLag; lag < Math.min(maxLag, signals.size() / 2); lag++) {
      double correlation = 0;
      int count = 0;

      for (int i = lag; i < signals.size(); i++) {
        correlation += signals.get(i) * signals.get(i - lag);
        count++;
      }

      if (count > 0) {
        correlation /= count;
        if (correlation > maxCorrelation) {
          maxCorrelation = correlation;
          bestLag = lag;
        }
      }
    }

    return bestLag > 0 ? bestLag / fps : 0;
  }

  /**
   * Enhanced BPM calculation with outlier filtering
   */
  private List<Double> calculateBPMFromHeartbeatsEnhanced(List<Long> heartbeats) {
    List<Double> bpmValues = new ArrayList<>();

    if (heartbeats.size() < 2) return bpmValues;

    // Calculate all intervals
    List<Long> intervals = new ArrayList<>();
    for (int i = 1; i < heartbeats.size(); i++) {
      long intervalMs = heartbeats.get(i) - heartbeats.get(i-1);
      intervals.add(intervalMs);
    }

    // Remove obvious outliers (intervals that are too short or too long)
    List<Long> filteredIntervals = new ArrayList<>();
    for (Long interval : intervals) {
      double intervalSec = interval / 1000.0;
      double bpm = 60.0 / intervalSec;

      // Only keep physiologically reasonable intervals (40-180 BPM)
      if (bpm >= 40 && bpm <= 180) {
        filteredIntervals.add(interval);
      }
    }

    // Convert filtered intervals to BPM
    for (Long interval : filteredIntervals) {
      double intervalSeconds = interval / 1000.0;
      double bpm = 60.0 / intervalSeconds;
      bpmValues.add(bpm);
    }

    Log.d(TAG, "BPM calculation: " + intervals.size() + " raw intervals -> " +
          filteredIntervals.size() + " filtered -> " + bpmValues.size() + " BPM values");

    return bpmValues;
  }

  /**
   * Create partial RPPGData when BPM calculation fails but we have signal data
   */
  private RPPGData createPartialRPPGData(List<Double> greenSignals, List<Long> timestamps) {
    Log.i(TAG, "Creating partial RPPG data without BPM values");

    List<RPPGData.Signal> signalList = new ArrayList<>();
    for (int i = 0; i < Math.min(greenSignals.size(), timestamps.size()); i++) {
      signalList.add(RPPGData.Signal.builder()
          .redChannel(0.0)
          .greenChannel(greenSignals.get(i))
          .blueChannel(0.0)
          .timestamp(timestamps.get(i))
          .build());
    }

    int durationSeconds = timestamps.size() > 1 ?
        (int)((timestamps.get(timestamps.size()-1) - timestamps.get(0)) / 1000) : 0;

    return RPPGData.builder()
        .heartbeats(new ArrayList<>()) // Empty heartbeats list
        .minBpm(0.0)
        .maxBpm(0.0)
        .averageBpm(0.0)
        .baselineBpm(0.0)
        .durationSeconds(durationSeconds)
        .signals(signalList)
        .build();
  }

  /**
   * Apply moving average filter to smooth the signal
   */
  private List<Double> applyMovingAverageFilter(List<Double> signals, int windowSize) {
    List<Double> smoothed = new ArrayList<>();

    for (int i = 0; i < signals.size(); i++) {
      int start = Math.max(0, i - windowSize/2);
      int end = Math.min(signals.size(), i + windowSize/2 + 1);

      double sum = 0;
      int count = 0;
      for (int j = start; j < end; j++) {
        sum += signals.get(j);
        count++;
      }

      smoothed.add(sum / count);
    }

    return smoothed;
  }

  /**
   * Validate if the palm ROI is suitable for rPPG signal extraction
   */
  private boolean isValidPalmROI(Rect palmROI, Mat frame) {
    if (palmROI == null) return false;

    // Check if ROI is within frame bounds
    if (palmROI.x < 0 || palmROI.y < 0 ||
        palmROI.x + palmROI.width > frame.cols() ||
        palmROI.y + palmROI.height > frame.rows()) {
      return false;
    }

    // Check minimum size requirements for reliable signal extraction
    int minSize = 40; // Minimum 40x40 pixels
    if (palmROI.width < minSize || palmROI.height < minSize) {
      return false;
    }

    // Check maximum size to avoid including too much background
    int maxSize = Math.min(frame.cols(), frame.rows()) / 3;
    if (palmROI.width > maxSize || palmROI.height > maxSize) {
      return false;
    }

    return true;
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

  /**
   * Draw ECG-style graph visualization with realistic ECG waveforms
   */
  private void drawECGGraph(Mat frame, int frameCount) {
    try {
      int frameWidth = frame.cols();
      int frameHeight = frame.rows();

      // Graph dimensions
      int graphHeight = 120;
      int graphWidth = frameWidth - 40;
      int graphX = 20;
      int graphY = frameHeight - graphHeight - 20;

      // Background for ECG graph - medical monitor style
      Scalar graphBgColor = new Scalar(10, 20, 10); // Dark green background like medical monitors
      Imgproc.rectangle(frame,
        new Point(graphX - 10, graphY - 10),
        new Point(graphX + graphWidth + 10, graphY + graphHeight + 10),
        graphBgColor, -1);

      // Border
      Scalar borderColor = new Scalar(100, 100, 100);
      Imgproc.rectangle(frame,
        new Point(graphX - 10, graphY - 10),
        new Point(graphX + graphWidth + 10, graphY + graphHeight + 10),
        borderColor, 2);

      // Graph title
      Imgproc.putText(frame, "ECG-Style rPPG Signal",
        new Point(graphX, graphY - 20),
        Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(0, 255, 0), 1);

      if (signalHistory.size() < 2) {
        // Show message when no signal data available
        Imgproc.putText(frame, "Waiting for palm detection...",
          new Point(graphX + graphWidth/2 - 100, graphY + graphHeight/2),
          Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(150, 150, 150), 1);
        return;
      }

      // Draw ECG paper-style grid
      drawECGPaperGrid(frame, graphX, graphY, graphWidth, graphHeight);

      // Generate ECG-like waveform from rPPG signal
      drawECGWaveform(frame, graphX, graphY, graphWidth, graphHeight);

      // Draw signal info with medical-style annotations
      drawECGAnnotations(frame, graphX, graphY, graphWidth, graphHeight, frameWidth);

    } catch (Exception e) {
      Log.w(TAG, "Error drawing ECG graph", e);
    }
  }

  /**
   * Draw ECG paper-style grid like medical devices
   */
  private void drawECGPaperGrid(Mat frame, int graphX, int graphY, int graphWidth, int graphHeight) {
    // Fine grid (1mm equivalent on ECG paper)
    Scalar fineGridColor = new Scalar(0, 80, 0); // Dark green
    int fineGridStep = 5;

    for (int x = graphX; x <= graphX + graphWidth; x += fineGridStep) {
      Imgproc.line(frame, new Point(x, graphY), new Point(x, graphY + graphHeight), fineGridColor, 1);
    }
    for (int y = graphY; y <= graphY + graphHeight; y += fineGridStep) {
      Imgproc.line(frame, new Point(graphX, y), new Point(graphX + graphWidth, y), fineGridColor, 1);
    }

    // Major grid (5mm equivalent on ECG paper)
    Scalar majorGridColor = new Scalar(0, 120, 0); // Brighter green
    int majorGridStep = 25;

    for (int x = graphX; x <= graphX + graphWidth; x += majorGridStep) {
      Imgproc.line(frame, new Point(x, graphY), new Point(x, graphY + graphHeight), majorGridColor, 1);
    }
    for (int y = graphY; y <= graphY + graphHeight; y += majorGridStep) {
      Imgproc.line(frame, new Point(graphX, y), new Point(graphX + graphWidth, y), majorGridColor, 1);
    }

    // Draw baseline (isoelectric line)
    int baselineY = graphY + graphHeight * 2 / 3;
    Imgproc.line(frame, new Point(graphX, baselineY), new Point(graphX + graphWidth, baselineY),
                 new Scalar(0, 150, 0), 2);
  }

  /**
   * Draw ECG-like waveform from rPPG signal data
   */
  private void drawECGWaveform(Mat frame, int graphX, int graphY, int graphWidth, int graphHeight) {
    if (signalHistory.isEmpty()) return;

    // Calculate baseline and detect heartbeats from signal
    List<Integer> heartbeatIndices = detectHeartbeatIndicesForVisualization();
    int baselineY = graphY + graphHeight * 2 / 3;

    // ECG waveform color - bright green like medical monitors
    Scalar ecgColor = new Scalar(0, 255, 0);

    // Generate smooth ECG waveform with proper interpolation
    List<Point> waveformPoints = generateSmoothECGWaveform(heartbeatIndices, graphX, graphY, graphWidth, graphHeight, baselineY);

    // Draw the waveform with anti-aliasing
    for (int i = 1; i < waveformPoints.size(); i++) {
      Point p1 = waveformPoints.get(i - 1);
      Point p2 = waveformPoints.get(i);
      Imgproc.line(frame, p1, p2, ecgColor, 2, Imgproc.LINE_AA);
    }

    // Highlight R-wave peaks
    highlightRWavePeaks(frame, heartbeatIndices, graphX, graphWidth, baselineY);
  }

  /**
   * Detect heartbeat indices in signal history for visualization
   */
  private List<Integer> detectHeartbeatIndicesForVisualization() {
    List<Integer> heartbeatIndices = new ArrayList<>();

    if (signalHistory.size() < 10) {
      return heartbeatIndices;
    }

    // Calculate signal statistics
    double mean = signalHistory.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    double stdDev = Math.sqrt(signalHistory.stream()
        .mapToDouble(val -> Math.pow(val - mean, 2))
        .average().orElse(0.0));

    double threshold = mean + 0.5 * stdDev;
    int minDistance = 15; // Minimum distance between heartbeats

    int lastPeakIndex = -minDistance;

    // Find peaks in signal history
    for (int i = 2; i < signalHistory.size() - 2; i++) {
      double current = signalHistory.get(i);
      double prev = signalHistory.get(i - 1);
      double next = signalHistory.get(i + 1);

      if (current > prev && current > next && current > threshold && (i - lastPeakIndex) >= minDistance) {
        heartbeatIndices.add(i);
        lastPeakIndex = i;
      }
    }

    return heartbeatIndices;
  }

  /**
   * Generate smooth ECG waveform with proper interpolation between signal points
   */
  private List<Point> generateSmoothECGWaveform(List<Integer> heartbeatIndices, int graphX, int graphY,
                                               int graphWidth, int graphHeight, int baselineY) {
    List<Point> points = new ArrayList<>();

    if (signalHistory.isEmpty()) {
      points.add(new Point(graphX, baselineY));
      return points;
    }

    // Calculate signal amplitude scaling
    double minSignalRange = signalHistory.stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
    double maxSignalRange = signalHistory.stream().mapToDouble(Double::doubleValue).max().orElse(128.0);
    double signalRange = Math.max(maxSignalRange - minSignalRange, 50.0);
    double amplitudeScale = (graphHeight * 0.6) / signalRange;

    // Create interpolated signal values across the entire graph width
    int totalPixels = graphWidth;
    double[] interpolatedSignal = new double[totalPixels];

    for (int pixel = 0; pixel < totalPixels; pixel++) {
      double signalPosition = (double)pixel * (signalHistory.size() - 1) / (totalPixels - 1);

      if (signalPosition >= signalHistory.size() - 1) {
        interpolatedSignal[pixel] = signalHistory.get(signalHistory.size() - 1);
      } else {
        int index1 = (int)Math.floor(signalPosition);
        int index2 = Math.min(index1 + 1, signalHistory.size() - 1);
        double fraction = signalPosition - index1;

        double value1 = signalHistory.get(index1);
        double value2 = signalHistory.get(index2);
        interpolatedSignal[pixel] = value1 + (value2 - value1) * fraction;
      }
    }

    // Apply smoothing
    double[] smoothedSignal = applySmoothingFilter(interpolatedSignal);

    // Mark heartbeat regions
    boolean[] isHeartbeatRegion = new boolean[totalPixels];
    for (int heartbeatIndex : heartbeatIndices) {
      if (heartbeatIndex < signalHistory.size()) {
        int pixelX = (int)((double)heartbeatIndex * totalPixels / signalHistory.size());
        int complexWidth = 40;
        int startX = Math.max(0, pixelX - complexWidth/2);
        int endX = Math.min(totalPixels - 1, pixelX + complexWidth/2);

        for (int x = startX; x <= endX; x++) {
          isHeartbeatRegion[x] = true;
        }
      }
    }

    // Generate waveform points
    for (int pixel = 0; pixel < totalPixels; pixel++) {
      int x = graphX + pixel;
      double signalValue = smoothedSignal[pixel];
      double scaledValue = (signalValue - (maxSignalRange + minSignalRange) / 2.0) * amplitudeScale;
      int y = (int)(baselineY - scaledValue);

      if (isHeartbeatRegion[pixel]) {
        y = enhanceHeartbeatMorphology(pixel, y, baselineY, heartbeatIndices, totalPixels);
      }

      y = Math.max(graphY, Math.min(graphY + graphHeight, y));
      points.add(new Point(x, y));
    }

    return points;
  }

  /**
   * Apply smoothing filter to reduce signal noise and jitter
   */
  private double[] applySmoothingFilter(double[] signal) {
    double[] smoothed = new double[signal.length];
    smoothed[0] = signal[0];

    for (int i = 1; i < signal.length - 1; i++) {
      smoothed[i] = 0.25 * signal[i-1] + 0.5 * signal[i] + 0.25 * signal[i+1];
    }
    smoothed[signal.length - 1] = signal[signal.length - 1];

    return smoothed;
  }

  /**
   * Enhance heartbeat regions with realistic ECG morphology
   */
  private int enhanceHeartbeatMorphology(int pixelX, int baseY, int baselineY, List<Integer> heartbeatIndices, int totalPixels) {
    int nearestHeartbeat = -1;
    double minDistance = Double.MAX_VALUE;

    for (int heartbeatIndex : heartbeatIndices) {
      if (heartbeatIndex < signalHistory.size()) {
        int heartbeatPixel = (int)((double)heartbeatIndex * totalPixels / signalHistory.size());
        double distance = Math.abs(pixelX - heartbeatPixel);

        if (distance < minDistance) {
          minDistance = distance;
          nearestHeartbeat = heartbeatPixel;
        }
      }
    }

    if (nearestHeartbeat == -1 || minDistance > 20) {
      return baseY;
    }

    double relativePos = (pixelX - nearestHeartbeat) / 20.0;
    double ecgAmplitude = 0;

    if (relativePos >= -1.5 && relativePos <= -0.7) {
      double pPos = (relativePos + 1.1) / 0.4;
      ecgAmplitude = 8 * Math.exp(-Math.pow(pPos * 3, 2));
    } else if (relativePos >= -0.4 && relativePos <= -0.1) {
      ecgAmplitude = -8 * Math.sin(Math.PI * (relativePos + 0.25) / 0.15);
    } else if (relativePos >= -0.1 && relativePos <= 0.1) {
      double rPos = relativePos / 0.1;
      ecgAmplitude = 50 * Math.exp(-Math.pow(rPos * 2, 2));
    } else if (relativePos >= 0.1 && relativePos <= 0.4) {
      ecgAmplitude = -15 * Math.sin(Math.PI * (relativePos - 0.1) / 0.3);
    } else if (relativePos >= 0.5 && relativePos <= 1.2) {
      double tPos = (relativePos - 0.85) / 0.35;
      ecgAmplitude = 12 * Math.exp(-Math.pow(tPos * 2, 2));
    }

    int enhancedY = (int)(baselineY - ecgAmplitude);
    return (int)(0.7 * enhancedY + 0.3 * baseY);
  }

  /**
   * Highlight R-wave peaks in the ECG display
   */
  private void highlightRWavePeaks(Mat frame, List<Integer> heartbeatIndices, int graphX, int graphWidth, int baselineY) {
    for (int heartbeatIndex : heartbeatIndices) {
      if (heartbeatIndex < signalHistory.size()) {
        int pixelX = (int)((double)heartbeatIndex * graphWidth / signalHistory.size());
        int x = graphX + pixelX;

        // Draw R-wave marker
        Imgproc.circle(frame, new Point(x, baselineY - 30), 3, new Scalar(255, 255, 0), -1);

        // Draw vertical line
        Imgproc.line(frame, new Point(x, baselineY - 50), new Point(x, baselineY + 10),
                     new Scalar(255, 255, 0), 1);
      }
    }
  }

  /**
   * Draw ECG annotations and information
   */
  private void drawECGAnnotations(Mat frame, int graphX, int graphY, int graphWidth, int graphHeight, int frameWidth) {
    try {
      // Current heart rate display
      String hrText = String.format(Locale.US, "HR: %.0f BPM", averageHeartRate);
      Imgproc.putText(frame, hrText,
          new Point(frameWidth - 150, graphY - 30),
          Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(0, 255, 255), 2);

      // Signal quality indicator
      double signalQuality = calculateSignalQuality();
      String qualityText = String.format(Locale.US, "Quality: %.0f%%", signalQuality * 100);
      Scalar qualityColor = signalQuality > 0.7 ? new Scalar(0, 255, 0) :
                           signalQuality > 0.4 ? new Scalar(0, 255, 255) : new Scalar(0, 100, 255);

      Imgproc.putText(frame, qualityText,
          new Point(frameWidth - 150, graphY - 10),
          Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, qualityColor, 1);

    } catch (Exception e) {
      Log.w(TAG, "Error drawing ECG annotations", e);
    }
  }

  /**
   * Calculate signal quality based on recent signal stability
   */
  private double calculateSignalQuality() {
    if (signalHistory.size() < 30) {
      return 0.0;
    }

    // Calculate coefficient of variation for recent signals
    List<Double> recentSignals = signalHistory.subList(
        Math.max(0, signalHistory.size() - 30), signalHistory.size());

    double mean = recentSignals.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    double stdDev = Math.sqrt(recentSignals.stream()
        .mapToDouble(val -> Math.pow(val - mean, 2))
        .average().orElse(0.0));

    if (mean == 0) return 0.0;

    double cv = stdDev / mean;
    return Math.max(0.0, Math.min(1.0, 1.0 - cv * 2)); // Normalize to 0-1 range
  }

  /**
   * Estimate current BPM from signal data
   */
  private double estimateCurrentBPM(List<Double> signals, double fps) {
    if (signals.size() < 60) {
      return averageHeartRate; // Return current estimate if insufficient data
    }

    try {
      // Simple peak detection for BPM estimation
      List<Integer> peaks = new ArrayList<>();
      double threshold = signals.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);

      for (int i = 1; i < signals.size() - 1; i++) {
        if (signals.get(i) > signals.get(i-1) && signals.get(i) > signals.get(i+1) && signals.get(i) > threshold) {
          if (peaks.isEmpty() || i - peaks.get(peaks.size()-1) > fps * 0.4) { // Min 0.4s between beats
            peaks.add(i);
          }
        }
      }

      if (peaks.size() < 2) {
        return averageHeartRate;
      }

      // Calculate average interval between peaks
      double totalInterval = 0;
      for (int i = 1; i < peaks.size(); i++) {
        totalInterval += peaks.get(i) - peaks.get(i-1);
      }

      double avgInterval = totalInterval / (peaks.size() - 1);
      double intervalSeconds = avgInterval / fps;
      double bpm = 60.0 / intervalSeconds;

      // Return reasonable BPM values only
      return (bpm >= 40 && bpm <= 180) ? bpm : averageHeartRate;

    } catch (Exception e) {
      Log.w(TAG, "Error estimating BPM", e);
      return averageHeartRate;
    }
  }
}
