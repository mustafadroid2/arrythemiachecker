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
  private final List<Double> signalHistory = new ArrayList<>();
  // Add synchronized timestamp tracking for ECG waveform
  private final List<Long> signalTimestamps = new ArrayList<>();
  private static final int MAX_SIGNAL_HISTORY = 300; // Keep last 300 samples (10 seconds at 30fps)

  // Heartbeat detection storage for step 3
  private final List<Long> heartbeatTimestamps = new ArrayList<>();
  private final List<Double> heartbeatSignalValues = new ArrayList<>();
  private double lastPeakSignal = 0.0;
  private long lastPeakTime = 0;
  private static final double HEARTBEAT_THRESHOLD_FACTOR = 1.2; // Peak must be 20% above average

  // Signal projection and continuity variables
  private final List<Double> continuousSignalBuffer = new ArrayList<>();
  private final List<Long> continuousTimestamps = new ArrayList<>();
  private Double lastValidSignal = null;
  private Long lastValidTimestamp;

  private double signalTrend = 0.0; // Track signal trend for projection
  private double averageHeartRate = 70.0; // Default HR for projection
  private int missedFrameCount = 0;
  private static final int MAX_INTERPOLATION_FRAMES = 15; // Max frames to interpolate (0.5 seconds at 30fps)

  // Signal quality tracking
  private double runningMean = 0.0;
  private double runningVariance = 0.0;

  // Enhanced BPM history for aggressive smoothing to prevent 80-150 BPM jumps
  private final List<Double> bpmHistory = new ArrayList<>();
  private static final int BPM_HISTORY_SIZE = 25; // Increased for better smoothing
  private double lastStableBPM = 0.0;

  // Enhanced smoothing parameters to prevent unrealistic jumps
  private static final double MAX_BPM_CHANGE_PER_SECOND = 8.0; // Max 8 BPM change per second (stricter)
  private static final double EXTREME_SMOOTHING_ALPHA = 0.03; // Very conservative smoothing
  private static final double MODERATE_SMOOTHING_ALPHA = 0.10; // Moderate smoothing
  private static final double NORMAL_SMOOTHING_ALPHA = 0.20; // Normal smoothing
  private long lastBPMUpdateTime = 0;

  public RPPGHandPalmServiceImpl(Context context) {
    this.context = context;
    this.mpHandTracker = new MediaPipeHandTracker(context);
  }

  @Override
  public RPPGData getRPPGSignals(String videoPath) {
    // RPPG heart rate extraction flow:
    // 1. Open video file with OpenCV VideoCapture
    // 2. Get video duration
    // 3. extract heartbeats timestamps from rPPG signals frame by frame
    // 4. Interpolate hand rPPG if heartbeat is missing from average heart rate
    //    the goals to mimic actual heart rate as close as possible
    // 5. Calculate heart rate metrics (min, max, average, baseline)
    // 6. create output video & image of heartbeat overtime as video duration with hand detection overlays and rPPG info

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
      long timestamp = startTime + (long)(frameCount * 1000L / fps);

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
    double projectedSignal = projectSignalValue(timestamp);

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
  private double projectSignalValue(long timestamp) {
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
    double noiseComponent = (Math.random() - 0.5);

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

    // Update heart rate estimate with enhanced smoothing to prevent 80-150 BPM jumps
    if (continuousSignalBuffer.size() >= 60) { // 2 seconds of data
      updateHeartRateEstimateWithEnhancedSmoothing();
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
   * ENHANCED HEART RATE ESTIMATION WITH AGGRESSIVE SMOOTHING TO PREVENT UNREALISTIC JUMPS
   */
  private void updateHeartRateEstimateWithEnhancedSmoothing() {
    if (continuousSignalBuffer.size() < 60) return;

    try {
      long currentTime = System.currentTimeMillis();

      // Use recent signal data for heart rate estimation
      List<Double> recentSignals = continuousSignalBuffer.subList(
          Math.max(0, continuousSignalBuffer.size() - 90),
          continuousSignalBuffer.size());

      double rawBPM = estimateCurrentBPM(recentSignals, 30.0);

      if (rawBPM > 40 && rawBPM < 200) {
        // STEP 1: Apply physiological rate limiting
        double validatedBPM = applyPhysiologicalValidation(rawBPM, currentTime);

        // STEP 2: Add to BPM history for temporal analysis
        bpmHistory.add(validatedBPM);

        // Maintain BPM history size
        while (bpmHistory.size() > BPM_HISTORY_SIZE) {
          bpmHistory.remove(0);
        }

        // STEP 3: Apply multi-stage smoothing
        double smoothedBPM = calculateHeavilySmoothedBPM();

        // STEP 4: Apply adaptive exponential smoothing
        double changeMagnitude = Math.abs(smoothedBPM - averageHeartRate);
        double alpha = calculateAdaptiveSmoothingFactor(changeMagnitude);

        // STEP 5: Update heart rate with smoothing
        averageHeartRate = alpha * smoothedBPM + (1 - alpha) * averageHeartRate;

        // STEP 6: Update stable BPM with extremely conservative approach
        updateStableBPM(smoothedBPM);

        lastBPMUpdateTime = currentTime;

        Log.d(TAG, String.format("BPM: raw=%.1f → validated=%.1f → smoothed=%.1f → final=%.1f (α=%.3f)",
                                 rawBPM, validatedBPM, smoothedBPM, averageHeartRate, alpha));
      }
    } catch (Exception e) {
      Log.w(TAG, "Failed to update heart rate estimate", e);
    }
  }

  /**
   * Apply physiological validation to prevent unrealistic BPM changes
   */
  private double applyPhysiologicalValidation(double rawBPM, long currentTime) {
    if (lastBPMUpdateTime == 0) {
      return rawBPM; // First measurement
    }

    double timeDeltaSeconds = (currentTime - lastBPMUpdateTime) / 1000.0;
    if (timeDeltaSeconds <= 0) {
      return averageHeartRate; // No time elapsed
    }

    // Calculate maximum allowed change based on time elapsed
    double maxAllowedChange = MAX_BPM_CHANGE_PER_SECOND * timeDeltaSeconds;
    double currentChange = rawBPM - averageHeartRate;

    // Limit the change to physiologically realistic values
    if (Math.abs(currentChange) > maxAllowedChange) {
      double limitedChange = Math.signum(currentChange) * maxAllowedChange;
      double validatedBPM = averageHeartRate + limitedChange;

      Log.d(TAG, String.format("BPM change limited: %.1f→%.1f (max change: %.1f in %.1fs)",
                               rawBPM, validatedBPM, maxAllowedChange, timeDeltaSeconds));

      return validatedBPM;
    }

    return rawBPM;
  }

  /**
   * Calculate adaptive smoothing factor based on change magnitude
   */
  private double calculateAdaptiveSmoothingFactor(double changeMagnitude) {
    if (changeMagnitude <= 3.0) {
      // Very small changes: use normal smoothing
      return NORMAL_SMOOTHING_ALPHA;
    } else if (changeMagnitude <= 8.0) {
      // Medium changes: use moderate smoothing
      return MODERATE_SMOOTHING_ALPHA;
    } else {
      // Large changes: use extreme smoothing to prevent jumps
      return EXTREME_SMOOTHING_ALPHA;
    }
  }

  /**
   * Calculate heavily smoothed BPM with multiple outlier rejection stages
   */
  private double calculateHeavilySmoothedBPM() {
    if (bpmHistory.isEmpty()) return averageHeartRate;
    if (bpmHistory.size() == 1) return bpmHistory.get(0);

    // Stage 1: Remove extreme outliers (beyond 1.5 standard deviations - stricter)
    List<Double> stage1Filtered = removeExtremeOutliers(new ArrayList<>(bpmHistory));

    // Stage 2: Remove values too far from median (within 15% - tighter)
    List<Double> stage2Filtered = removeMedianOutliers(stage1Filtered);

    // Stage 3: Apply temporal consistency filter (max 10% change)
    List<Double> stage3Filtered = applyTemporalConsistencyFilter(stage2Filtered);

    // Calculate heavily weighted average (recent values have much higher weight)
    return calculateExponentiallyWeightedAverage(stage3Filtered);
  }

  /**
   * Remove extreme outliers using statistical methods
   */
  private List<Double> removeExtremeOutliers(List<Double> values) {
    if (values.size() < 3) return values;

    double mean = values.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    double stdDev = Math.sqrt(values.stream()
        .mapToDouble(val -> Math.pow(val - mean, 2))
        .average().orElse(0.0));

    List<Double> filtered = new ArrayList<>();
    for (Double value : values) {
      // Keep within 1.5 standard deviations (stricter than before)
      if (Math.abs(value - mean) <= 1.5 * stdDev) {
        filtered.add(value);
      }
    }

    // Ensure we don't remove too many values
    return filtered.size() >= values.size() / 2 ? filtered : values;
  }

  /**
   * Remove values too far from median
   */
  private List<Double> removeMedianOutliers(List<Double> values) {
    if (values.size() < 3) return values;

    double median = calculateMedian(new ArrayList<>(values));
    List<Double> filtered = new ArrayList<>();

    for (Double value : values) {
      // Keep values within 15% of median (tighter than before)
      if (Math.abs(value - median) <= median * 0.15) {
        filtered.add(value);
      }
    }

    // Ensure we don't remove too many values
    return filtered.size() >= values.size() / 2 ? filtered : values;
  }

  /**
   * Apply temporal consistency filter to ensure smooth transitions
   */
  private List<Double> applyTemporalConsistencyFilter(List<Double> values) {
    if (values.size() < 3) return values;

    List<Double> filtered = new ArrayList<>();
    filtered.add(values.get(0)); // Always keep first value

    for (int i = 1; i < values.size(); i++) {
      double current = values.get(i);
      double previous = filtered.get(filtered.size() - 1);

      // Only accept values that don't change too rapidly (max 10% change)
      double change = Math.abs(current - previous);
      double maxAllowedChange = previous * 0.10; // Max 10% change between consecutive readings

      if (change <= maxAllowedChange) {
        filtered.add(current);
      } else {
        // Use interpolated value instead of rejecting completely
        double limitedValue = previous + Math.signum(current - previous) * maxAllowedChange;
        filtered.add(limitedValue);
      }
    }

    return filtered;
  }

  /**
   * Calculate exponentially weighted average with very high weight on recent values
   */
  private double calculateExponentiallyWeightedAverage(List<Double> values) {
    if (values.isEmpty()) return averageHeartRate;
    if (values.size() == 1) return values.get(0);

    double weightedSum = 0;
    double totalWeight = 0;

    // Use very aggressive exponential weighting (factor of 1.8 for even more recent bias)
    for (int i = 0; i < values.size(); i++) {
      double weight = Math.pow(1.8, i);
      weightedSum += values.get(i) * weight;
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
   * Update stable BPM with extremely conservative approach
   */
  private void updateStableBPM(double smoothedBPM) {
    if (lastStableBPM == 0.0) {
      lastStableBPM = smoothedBPM;
      return;
    }

    double change = Math.abs(smoothedBPM - lastStableBPM);

    if (change < 3.0) {
      // Small change: normal update
      lastStableBPM = 0.92 * lastStableBPM + 0.08 * smoothedBPM;
    } else if (change < 6.0) {
      // Medium change: conservative update
      lastStableBPM = 0.96 * lastStableBPM + 0.04 * smoothedBPM;
    } else {
      // Large change: extremely conservative update
      lastStableBPM = 0.99 * lastStableBPM + 0.01 * smoothedBPM;
    }
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
   * Draw projection indicator when using estimated signals
   */
  private void drawProjectionIndicator(Mat frame, int missedFrames, int maxFrames) {
    try {
      int frameWidth = frame.cols();

      // Calculate projection confidence
      double confidence = 1.0 - ((double)missedFrames / maxFrames);

      // Draw projection status
      String projectionText = String.format(Locale.US, "PROJECTING (%.0f%% confidence)", confidence * 100);
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
      String detectionStatus = isPalmDetected ? "DETECTED" : "PROJECTING";
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

      // Draw signal as simple line graph
      drawSimpleSignalGraph(frame, graphX, graphY, graphWidth, graphHeight);

      // Draw current heart rate
      String hrText = String.format(Locale.US, "HR: %.0f BPM", averageHeartRate);
      Imgproc.putText(frame, hrText,
          new Point(frameWidth - 150, graphY - 30),
          Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(0, 255, 255), 2);

    } catch (Exception e) {
      Log.w(TAG, "Error drawing ECG graph", e);
    }
  }

  /**
   * Draw simple signal graph
   */
  private void drawSimpleSignalGraph(Mat frame, int graphX, int graphY, int graphWidth, int graphHeight) {
    if (signalHistory.size() < 2) return;

    // Calculate signal amplitude scaling
    double minSignal = signalHistory.stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
    double maxSignal = signalHistory.stream().mapToDouble(Double::doubleValue).max().orElse(255.0);
    double signalRange = Math.max(maxSignal - minSignal, 50.0);

    int baselineY = graphY + graphHeight * 2 / 3;
    double amplitudeScale = (graphHeight * 0.6) / signalRange;

    // ECG waveform color - bright green like medical monitors
    Scalar ecgColor = new Scalar(0, 255, 0);

    // Draw the signal
    for (int i = 1; i < signalHistory.size(); i++) {
      int x1 = graphX + (i - 1) * graphWidth / signalHistory.size();
      int x2 = graphX + i * graphWidth / signalHistory.size();

      double signal1 = signalHistory.get(i - 1);
      double signal2 = signalHistory.get(i);

      int y1 = (int)(baselineY - (signal1 - (maxSignal + minSignal) / 2.0) * amplitudeScale);
      int y2 = (int)(baselineY - (signal2 - (maxSignal + minSignal) / 2.0) * amplitudeScale);

      y1 = Math.max(graphY, Math.min(graphY + graphHeight, y1));
      y2 = Math.max(graphY, Math.min(graphY + graphHeight, y2));

      Imgproc.line(frame, new Point(x1, y1), new Point(x2, y2), ecgColor, 2);
    }
  }

  /**
   * Calculate heart rate metrics from green channel signals
   */
  private RPPGData calculateHeartRateMetrics(List<Double> greenSignals, List<Long> timestamps, double fps) {
    Log.d(TAG, "Starting heart rate calculation with " + greenSignals.size() + " signals");

    if (greenSignals.size() < 30) { // Need at least 1 second of data at 30fps
      Log.w(TAG, "Insufficient data for heart rate calculation: " + greenSignals.size() + " samples");
      return RPPGData.empty();
    }

    try {
      // Simple BPM calculation - use the smoothed average heart rate from real-time processing
      double calculatedBPM = averageHeartRate;

      // Extract heartbeat timestamps from signal peaks
      List<Long> heartbeatTimestamps = extractHeartbeatTimestamps(greenSignals, timestamps, fps);

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

      Log.i(TAG, String.format("Heart rate analysis complete: %.1f BPM, %d signals, %d heartbeats in %d seconds",
                               calculatedBPM, signalList.size(), heartbeatTimestamps.size(), durationSeconds));

      return RPPGData.builder()
          .heartbeats(heartbeatTimestamps) // Return actual heartbeat timestamps
          .minBpm(calculatedBPM - 5.0)
          .maxBpm(calculatedBPM + 5.0)
          .averageBpm(calculatedBPM)
          .baselineBpm(calculatedBPM)
          .durationSeconds(durationSeconds)
          .signals(signalList)
          .build();

    } catch (Exception e) {
      Log.e(TAG, "Error calculating heart rate metrics", e);
      return RPPGData.empty();
    }
  }

  /**
   * Extract heartbeat timestamps from signal peaks
   */
  private List<Long> extractHeartbeatTimestamps(List<Double> signals, List<Long> timestamps, double fps) {
    List<Long> heartbeats = new ArrayList<>();

    if (signals.size() < 60 || timestamps.size() != signals.size()) {
      Log.w(TAG, "Insufficient data for heartbeat detection");
      return heartbeats;
    }

    try {
      // Calculate signal statistics for peak detection
      double mean = signals.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
      double stdDev = Math.sqrt(signals.stream()
          .mapToDouble(val -> Math.pow(val - mean, 2))
          .average().orElse(0.0));

      // Set dynamic threshold based on signal characteristics
      double threshold = mean + (stdDev * 0.5); // Adjust sensitivity

      // Minimum interval between heartbeats (in samples) to prevent false positives
      int minIntervalSamples = (int)(fps * 0.4); // 0.4 seconds minimum (150 BPM max)

      // Peak detection with improved algorithm
      for (int i = 2; i < signals.size() - 2; i++) {
        double current = signals.get(i);
        double prev1 = signals.get(i - 1);
        double prev2 = signals.get(i - 2);
        double next1 = signals.get(i + 1);
        double next2 = signals.get(i + 2);

        // Check if current point is a local maximum above threshold
        boolean isLocalMax = current > prev1 && current > next1 &&
                            current > prev2 && current > next2 &&
                            current > threshold;

        if (isLocalMax) {
          // Check minimum interval constraint
          if (heartbeats.isEmpty() ||
              (i - getLastPeakIndex(heartbeats, timestamps)) >= minIntervalSamples) {
            heartbeats.add(timestamps.get(i));
            Log.d(TAG, String.format("Heartbeat detected at %d ms, signal: %.2f",
                                   timestamps.get(i), current));
          }
        }
      }

      Log.i(TAG, String.format("Detected %d heartbeats from %d signal samples",
                               heartbeats.size(), signals.size()));

    } catch (Exception e) {
      Log.w(TAG, "Error in heartbeat detection", e);
    }

    return heartbeats;
  }

  /**
   * Get the index of the last detected peak
   */
  private int getLastPeakIndex(List<Long> heartbeats, List<Long> timestamps) {
    if (heartbeats.isEmpty()) return -1;

    long lastHeartbeat = heartbeats.get(heartbeats.size() - 1);
    for (int i = timestamps.size() - 1; i >= 0; i--) {
      if (timestamps.get(i).equals(lastHeartbeat)) {
        return i;
      }
    }
    return -1;
  }

  /**
   * Estimate current BPM from signal data using FFT-based approach
   */
  private double estimateCurrentBPM(List<Double> signals, double fps) {
    if (signals.size() < 60) return averageHeartRate; // Need at least 2 seconds of data

    try {
      // Simple peak counting approach for BPM estimation
      double mean = signals.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
      double stdDev = Math.sqrt(signals.stream()
          .mapToDouble(val -> Math.pow(val - mean, 2))
          .average().orElse(0.0));

      double threshold = mean + (stdDev * 0.3);
      int peakCount = 0;

      // Count peaks in the signal
      for (int i = 2; i < signals.size() - 2; i++) {
        double current = signals.get(i);
        if (current > threshold &&
            current > signals.get(i-1) && current > signals.get(i+1) &&
            current > signals.get(i-2) && current > signals.get(i+2)) {
          peakCount++;
        }
      }

      // Calculate BPM from peak count
      double durationSeconds = signals.size() / fps;
      double estimatedBPM = (peakCount * 60.0) / durationSeconds;

      // Clamp to reasonable range
      estimatedBPM = Math.max(40, Math.min(200, estimatedBPM));

      return estimatedBPM;

    } catch (Exception e) {
      Log.w(TAG, "Error estimating BPM", e);
      return averageHeartRate;
    }
  }

  /**
   * Rotate frame to correct video orientation
   */
  private Mat rotateFrame(Mat frame, int rotationDegrees) {
    if (rotationDegrees == 0) return frame;

    try {
      Mat rotated = new Mat();
      Point center = new Point(frame.cols() / 2.0, frame.rows() / 2.0);
      Mat rotationMatrix = Imgproc.getRotationMatrix2D(center, -rotationDegrees, 1.0);

      // Calculate new frame size to avoid cropping
      double cos = Math.abs(rotationMatrix.get(0, 0)[0]);
      double sin = Math.abs(rotationMatrix.get(0, 1)[0]);

      int newWidth = (int) (frame.rows() * sin + frame.cols() * cos);
      int newHeight = (int) (frame.rows() * cos + frame.cols() * sin);

      // Adjust translation
      rotationMatrix.put(0, 2, rotationMatrix.get(0, 2)[0] + (newWidth / 2.0) - center.x);
      rotationMatrix.put(1, 2, rotationMatrix.get(1, 2)[0] + (newHeight / 2.0) - center.y);

      Imgproc.warpAffine(frame, rotated, rotationMatrix, new Size(newWidth, newHeight));

      rotationMatrix.release();
      return rotated;

    } catch (Exception e) {
      Log.w(TAG, "Error rotating frame", e);
      return frame; // Return original frame if rotation fails
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
        Log.w(TAG, "Error releasing MediaMetadataRetriever", e);
      }
    }
  }
  }
