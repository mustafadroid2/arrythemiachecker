package com.gunadarma.heartratearrhythmiachecker.service.rppg;

import android.content.Context;
import android.media.MediaMetadataRetriever;
import android.util.Log;

import com.gunadarma.heartratearrhythmiachecker.constant.AppConstant;
import com.gunadarma.heartratearrhythmiachecker.model.RPPGData;
import com.gunadarma.heartratearrhythmiachecker.service.MediaPipeHandTracker;

import org.opencv.core.Core;
import org.opencv.core.Mat;
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
  private static final double MIN_HR_BPM = 48;
  private static final double MAX_HR_BPM = 240.0;
  private static final double SKIN_CONFIDENCE_THRESHOLD = 0.05; // Very low threshold for testing
  private static final double PEAK_THRESHOLD = 0.3; // Lowered for more sensitive detection
  private static final double BANDPASS_LOW_FREQ = 0.83; // 50 BPM in Hz
  private static final double BANDPASS_HIGH_FREQ = 3.0; // 180 BPM in Hz

  // Additional constants for improved processing
  private static final int MIN_SIGNAL_LENGTH = 30; // Even lower - just 1 second at 30fps for testing
  private static final double SIGNAL_AMPLIFICATION = 100.0; // Amplify small rPPG signals

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
          .baselineBpm(result.baselineBpm) // Added baseline BPM
          .durationSeconds(videoDurationSeconds)
          .signals(signals)
          .build();

      Log.i(TAG, String.format("Palm rPPG extraction completed: %.1f BPM baseline, %.1f BPM average, %d heartbeats detected",
                               result.baselineBpm, result.averageBpm, result.heartbeats.size()));

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

          // Log skin detection details for debugging
          if (frameCount % 300 == 0) { // Log every 10 seconds at 30fps
            Log.d(TAG, String.format("Frame %d - Skin ratio: %.3f (threshold: %.3f), RGB means: [%.1f, %.1f, %.1f]",
                   frameCount, skinRatio, SKIN_CONFIDENCE_THRESHOLD, means.val[0], means.val[1], means.val[2]));
          }

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
          } else {
            // Log when skin detection fails
            if (frameCount % 300 == 0) {
              Log.d(TAG, String.format("Frame %d - Skin detection failed: ratio %.3f < threshold %.3f",
                     frameCount, skinRatio, SKIN_CONFIDENCE_THRESHOLD));
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

    // Use multiple color spaces for robust skin detection

    // 1. YCrCb color space (most reliable for skin detection)
    Mat ycrcb = new Mat();
    Imgproc.cvtColor(rgb, ycrcb, Imgproc.COLOR_RGB2YCrCb);
    Mat mask1 = new Mat();
    Core.inRange(ycrcb,
      new Scalar(0, 133, 77),   // Lower bound
      new Scalar(255, 173, 127), // Upper bound
      mask1);

    // 2. HSV color space for hue-based detection
    Mat hsv = new Mat();
    Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV);
    Mat mask2 = new Mat();
    Core.inRange(hsv,
      new Scalar(0, 30, 60),   // Lower bound (wider range)
      new Scalar(25, 255, 255), // Upper bound
      mask2);

    // 3. RGB-based skin detection (complementary approach)
    Mat mask3 = new Mat();
    List<Mat> rgbChannels = new ArrayList<>();
    rgbChannels.add(new Mat());
    rgbChannels.add(new Mat());
    rgbChannels.add(new Mat());
    Core.split(rgb, rgbChannels);

    // RGB skin detection rules: R > G > B, R > 95, G > 40, B > 20
    Mat rMask = new Mat();
    Mat gMask = new Mat();
    Mat bMask = new Mat();
    Mat rgMask = new Mat();
    Mat gbMask = new Mat();

    Core.compare(rgbChannels.get(0), new Scalar(95), rMask, Core.CMP_GT);  // R > 95
    Core.compare(rgbChannels.get(1), new Scalar(40), gMask, Core.CMP_GT);  // G > 40
    Core.compare(rgbChannels.get(2), new Scalar(20), bMask, Core.CMP_GT);  // B > 20
    Core.compare(rgbChannels.get(0), rgbChannels.get(1), rgMask, Core.CMP_GT); // R > G
    Core.compare(rgbChannels.get(1), rgbChannels.get(2), gbMask, Core.CMP_GT); // G > B

    // Combine RGB conditions
    Core.bitwise_and(rMask, gMask, mask3);
    Core.bitwise_and(mask3, bMask, mask3);
    Core.bitwise_and(mask3, rgMask, mask3);
    Core.bitwise_and(mask3, gbMask, mask3);

    // Combine all three masks using OR operation for better coverage
    Mat combinedMask = new Mat();
    Core.bitwise_or(mask1, mask2, combinedMask);
    Core.bitwise_or(combinedMask, mask3, skinMask);

    // Apply morphological operations to clean up the mask
    Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5, 5));
    Imgproc.morphologyEx(skinMask, skinMask, Imgproc.MORPH_OPEN, kernel);
    Imgproc.morphologyEx(skinMask, skinMask, Imgproc.MORPH_CLOSE, kernel);

    // Cleanup
    ycrcb.release();
    hsv.release();
    mask1.release();
    mask2.release();
    mask3.release();
    combinedMask.release();
    for (Mat channel : rgbChannels) {
      channel.release();
    }
    rMask.release();
    gMask.release();
    bMask.release();
    rgMask.release();
    gbMask.release();
    kernel.release();

    return skinMask;
  }

  private RPPGProcessingResult processRPPGSignals(List<RPPGData.Signal> signals, double fps) {
    if (signals.size() < MIN_SIGNAL_LENGTH) {
      Log.w(TAG, "Insufficient signal data for processing: " + signals.size() + " signals (minimum required: " + MIN_SIGNAL_LENGTH + ")");
      return new RPPGProcessingResult(new ArrayList<>(), 0, 0, 0, 0);
    }

    Log.d(TAG, "Processing " + signals.size() + " signals for heart rate detection");

    // Use green channel for better rPPG signal (most sensitive to blood volume changes)
    List<Double> greenSignal = new ArrayList<>();
    for (RPPGData.Signal signal : signals) {
      greenSignal.add(signal.getGreenChannel());
    }

    // Log initial signal statistics
    double signalMean = greenSignal.stream().mapToDouble(d -> d).average().orElse(0.0);
    double signalStd = Math.sqrt(greenSignal.stream().mapToDouble(d -> Math.pow(d - signalMean, 2)).average().orElse(0.0));
    Log.d(TAG, String.format("Initial signal stats - Mean: %.3f, Std: %.3f, Range: %.3f to %.3f",
           signalMean, signalStd,
           greenSignal.stream().mapToDouble(d -> d).min().orElse(0),
           greenSignal.stream().mapToDouble(d -> d).max().orElse(0)));

    // Apply improved preprocessing
    List<Double> processedSignal = preprocessSignalImproved(greenSignal);

    // Apply improved bandpass filtering
    List<Double> filteredSignal = applyImprovedBandpassFilter(processedSignal, fps);

    // Extract heartbeats using improved peak detection
    List<Long> heartbeats = extractHeartbeatsImproved(filteredSignal, signals, fps);

    // Calculate statistics
    double averageBpm = calculateAverageBPM(heartbeats);
    double minBpm = calculateMinBPM(heartbeats);
    double maxBpm = calculateMaxBPM(heartbeats);

    // Calculate baseline BPM using median of heartbeats
    double baselineBpm = calculateBaselineBPM(heartbeats);

    Log.i(TAG, String.format("Heart rate processing complete: %.1f BPM average, %d heartbeats detected",
                             averageBpm, heartbeats.size()));

    return new RPPGProcessingResult(heartbeats, minBpm, maxBpm, averageBpm, baselineBpm);
  }

  /**
   * Improved signal preprocessing that preserves rPPG characteristics
   */
  private List<Double> preprocessSignalImproved(List<Double> signal) {
    List<Double> processed = new ArrayList<>();

    if (signal.isEmpty()) return processed;

    // Apply gentle detrending using moving average
    int windowSize = Math.max(30, signal.size() / 10); // Adaptive window size
    List<Double> detrended = new ArrayList<>();

    for (int i = 0; i < signal.size(); i++) {
      double trend = 0;
      int count = 0;
      int start = Math.max(0, i - windowSize/2);
      int end = Math.min(signal.size() - 1, i + windowSize/2);

      for (int j = start; j <= end; j++) {
        trend += signal.get(j);
        count++;
      }
      trend /= count;

      detrended.add(signal.get(i) - trend);
    }

    // Apply gentle smoothing with smaller window
    for (int i = 0; i < detrended.size(); i++) {
      double smoothed = detrended.get(i);

      // Use 3-point smoothing only
      if (i > 0 && i < detrended.size() - 1) {
        smoothed = (detrended.get(i-1) + 2 * detrended.get(i) + detrended.get(i+1)) / 4.0;
      }

      // Amplify the signal slightly to improve detection
      processed.add(smoothed * SIGNAL_AMPLIFICATION);
    }

    // Log preprocessing results
    double processedMean = processed.stream().mapToDouble(d -> d).average().orElse(0.0);
    double processedStd = Math.sqrt(processed.stream().mapToDouble(d -> Math.pow(d - processedMean, 2)).average().orElse(0.0));
    Log.d(TAG, String.format("Processed signal stats - Mean: %.3f, Std: %.3f", processedMean, processedStd));

    return processed;
  }

  /**
   * Improved bandpass filter with better frequency response
   */
  private List<Double> applyImprovedBandpassFilter(List<Double> signal, double fps) {
    if (signal.size() < 10) return signal;

    List<Double> filtered = new ArrayList<>(signal);

    // Apply high-pass filter to remove DC and low-frequency drift
    int hpWindowSize = Math.min((int)(fps * 2), signal.size() / 3); // 2-second window for high-pass
    for (int i = hpWindowSize; i < signal.size(); i++) {
      double trend = 0;
      for (int j = i - hpWindowSize; j < i; j++) {
        trend += signal.get(j);
      }
      trend /= hpWindowSize;
      filtered.set(i, signal.get(i) - trend);
    }

    // Apply low-pass filter to remove high-frequency noise
    int lpWindowSize = Math.max(2, (int)(fps / 6)); // Remove frequencies above 6 Hz
    List<Double> lowPassed = new ArrayList<>();
    for (int i = 0; i < filtered.size(); i++) {
      double sum = 0;
      int count = 0;
      int start = Math.max(0, i - lpWindowSize/2);
      int end = Math.min(filtered.size() - 1, i + lpWindowSize/2);

      for (int j = start; j <= end; j++) {
        sum += filtered.get(j);
        count++;
      }
      lowPassed.add(sum / count);
    }

    Log.d(TAG, String.format("Filter applied - HP window: %d, LP window: %d", hpWindowSize, lpWindowSize));
    return lowPassed;
  }

  /**
   * Improved peak detection specifically for palm-based rPPG
   */
  private List<Long> extractHeartbeatsImproved(List<Double> signal, List<RPPGData.Signal> originalSignals, double fps) {
    List<Long> heartbeats = new ArrayList<>();

    if (signal.size() < 10) return heartbeats;

    // Calculate dynamic threshold based on signal statistics
    double mean = signal.stream().mapToDouble(d -> d).average().orElse(0.0);
    double variance = signal.stream().mapToDouble(d -> Math.pow(d - mean, 2)).average().orElse(0.0);
    double stdDev = Math.sqrt(variance);

    // Use adaptive threshold
    double threshold = mean + stdDev * 0.3; // Lower threshold for palm detection

    // Minimum distance between peaks (in samples)
    int minPeakDistance = (int)(0.4 * fps); // Minimum 0.4 seconds between heartbeats
    int maxPeakDistance = (int)(1.5 * fps); // Maximum 1.5 seconds between heartbeats

    List<Integer> peakIndices = new ArrayList<>();

    Log.d(TAG, String.format("Peak detection: threshold=%.3f, minDist=%d, signal_size=%d",
                             threshold, minPeakDistance, signal.size()));

    // Find peaks using a sliding window approach
    for (int i = 2; i < signal.size() - 2; i++) {
      double current = signal.get(i);

      // Check if current point is a local maximum
      if (current > signal.get(i-1) && current > signal.get(i+1) &&
          current > signal.get(i-2) && current > signal.get(i+2) &&
          current > threshold) {

        // Check minimum distance from previous peak
        boolean validPeak = true;
        for (int peakIdx : peakIndices) {
          if (Math.abs(i - peakIdx) < minPeakDistance) {
            // If current peak is higher, replace the previous one
            if (current > signal.get(peakIdx)) {
              peakIndices.remove((Integer)peakIdx);
              break;
            } else {
              validPeak = false;
              break;
            }
          }
        }

        if (validPeak) {
          peakIndices.add(i);
        }
      }
    }

    // Post-process peaks to ensure reasonable heart rate
    List<Integer> filteredPeaks = new ArrayList<>();
    for (int i = 0; i < peakIndices.size(); i++) {
      int currentPeak = peakIndices.get(i);

      if (filteredPeaks.isEmpty()) {
        filteredPeaks.add(currentPeak);
      } else {
        int lastPeak = filteredPeaks.get(filteredPeaks.size() - 1);
        int distance = currentPeak - lastPeak;

        // Check if interval is within reasonable heart rate range
        if (distance >= minPeakDistance && distance <= maxPeakDistance) {
          filteredPeaks.add(currentPeak);
        } else if (distance > maxPeakDistance) {
          // Gap too large, might have missed a beat - keep this peak
          filteredPeaks.add(currentPeak);
        }
        // If distance too small, skip this peak
      }
    }

    // Convert peak indices to timestamps
    for (int peakIdx : filteredPeaks) {
      if (peakIdx < originalSignals.size()) {
        heartbeats.add(originalSignals.get(peakIdx).getTimestamp());
      }
    }

    Log.d(TAG, String.format("Peak detection found %d raw peaks, %d filtered peaks, %d heartbeats",
                             peakIndices.size(), filteredPeaks.size(), heartbeats.size()));

    return heartbeats;
  }

  /**
   * Calculate instantaneous BPM using simpler approach
   */
  private double calculateInstantaneousBPM(List<RPPGData.Signal> signals, int windowSize) {
    if (signals.size() < 10) return 0;

    int startIdx = Math.max(0, signals.size() - windowSize);
    List<RPPGData.Signal> recentSignals = signals.subList(startIdx, signals.size());
    
    if (recentSignals.size() < 10) return 0;

    // Use green channel for instantaneous BPM calculation
    List<Double> greenValues = new ArrayList<>();
    for (RPPGData.Signal signal : recentSignals) {
      greenValues.add(signal.getGreenChannel());
    }

    // Simple peak counting approach
    double mean = greenValues.stream().mapToDouble(d -> d).average().orElse(0.0);
    double threshold = mean + greenValues.stream().mapToDouble(d -> Math.abs(d - mean)).average().orElse(0.0) * 0.2;

    int peaks = 0;
    boolean wasAboveThreshold = false;

    for (double value : greenValues) {
      boolean isAboveThreshold = value > threshold;

      if (isAboveThreshold && !wasAboveThreshold) {
        peaks++;
      }
      wasAboveThreshold = isAboveThreshold;
    }
    
    // Convert to BPM
    double timeSpanSeconds = (recentSignals.get(recentSignals.size() - 1).getTimestamp() -
                             recentSignals.get(0).getTimestamp()) / 1000.0;
    
    double bpm = timeSpanSeconds > 0 ? (peaks * 60.0) / timeSpanSeconds : 0;

    // Clamp to reasonable heart rate range
    return Math.max(40, Math.min(200, bpm));
  }

  /**
   * Calculate average BPM from heartbeat intervals
   */
  private double calculateAverageBPM(List<Long> heartbeats) {
    if (heartbeats.size() < 2) return 0;

    List<Long> intervals = new ArrayList<>();
    for (int i = 1; i < heartbeats.size(); i++) {
      intervals.add(heartbeats.get(i) - heartbeats.get(i-1));
    }

    double avgInterval = intervals.stream().mapToLong(l -> l).average().orElse(0);
    return avgInterval > 0 ? 60000.0 / avgInterval : 0;
  }

  /**
   * Calculate minimum BPM from heartbeat intervals
   */
  private double calculateMinBPM(List<Long> heartbeats) {
    if (heartbeats.size() < 2) return 0;

    List<Long> intervals = new ArrayList<>();
    for (int i = 1; i < heartbeats.size(); i++) {
      intervals.add(heartbeats.get(i) - heartbeats.get(i-1));
    }

    long maxInterval = Collections.max(intervals);
    return maxInterval > 0 ? 60000.0 / maxInterval : 0;
  }

  /**
   * Calculate maximum BPM from heartbeat intervals
   */
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
   * Calculate baseline BPM from heartbeat intervals using median
   */
  private double calculateBaselineBPM(List<Long> heartbeats) {
    if (heartbeats.size() < 2) return 0;

    List<Long> intervals = new ArrayList<>();
    for (int i = 1; i < heartbeats.size(); i++) {
      intervals.add(heartbeats.get(i) - heartbeats.get(i-1));
    }

    // Calculate median interval
    Collections.sort(intervals);
    long medianInterval;
    if (intervals.size() % 2 == 0) {
      medianInterval = (intervals.get(intervals.size() / 2 - 1) + intervals.get(intervals.size() / 2)) / 2;
    } else {
      medianInterval = intervals.get(intervals.size() / 2);
    }

    return medianInterval > 0 ? 60000.0 / medianInterval : 0;
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
   * Result container for rPPG processing with baseline
   */
  private static class RPPGProcessingResult {
    final List<Long> heartbeats;
    final double minBpm;
    final double maxBpm;
    final double averageBpm;
    final double baselineBpm;

    RPPGProcessingResult(List<Long> heartbeats, double minBpm, double maxBpm, double averageBpm, double baselineBpm) {
      this.heartbeats = heartbeats;
      this.minBpm = minBpm;
      this.maxBpm = maxBpm;
      this.averageBpm = averageBpm;
      this.baselineBpm = baselineBpm;
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
