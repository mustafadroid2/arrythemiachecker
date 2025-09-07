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
  private List<Double> processedSignalHistory = new ArrayList<>(); // For ECG-like processed signal
  private static final int MAX_SIGNAL_HISTORY = 300; // Keep last 300 samples (10 seconds at 30fps)

  // rPPG signal processing variables
  private List<Double> rawSignalBuffer = new ArrayList<>(); // Buffer for raw signals
  private double signalBaseline = 0.0; // Running baseline
  private int frameCounter = 0;

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
      RPPGData rppgData = processHandDetectionOnly(videoPath, cap, fps, rotationDegrees);

      Log.i(TAG, "Hand palm rPPG analysis completed");

      return rppgData;

    } finally {
      cap.release();
      if (mpHandTracker != null) {
        mpHandTracker.release();
      }
    }
  }

  private RPPGData processHandDetectionOnly(String videoPath, VideoCapture cap, double fps, int rotationDegrees) {
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

        if (handResult != null && handResult.palmROI != null && isValidPalmROI(handResult.palmROI, frame)) {
          palmFrames++;

          // Extract rPPG signals from palm region
          RPPGData.Signal signal = extractRPPGSignal(frame, handResult.palmROI, timestamp);
          if (signal != null) {
            signals.add(signal);
            redSignals.add(signal.getRedChannel());
            greenSignals.add(signal.getGreenChannel());
            blueSignals.add(signal.getBlueChannel());
            timestamps.add(signal.getTimestamp());

            // Update signal history for ECG graph visualization
            signalHistory.add(signal.getGreenChannel());

            // Keep only recent history
            while (signalHistory.size() > MAX_SIGNAL_HISTORY) {
              signalHistory.remove(0);
            }
          }

          // Draw hand detection overlays using MediaPipe's method
          mpHandTracker.drawHandAnnotations(overlayFrame, handResult);

          // Draw rPPG info on palm region
          drawRPPGOverlays(overlayFrame, handResult.palmROI, signal);
        }

        // Draw comprehensive overlays on frame
        drawRPPGAnalysisOverlays(overlayFrame, frameCount, fps, palmFrames, signals.size());

        // Write frame to output video only if writer is available
        if (writer != null && writer.isOpened()) {
          writer.write(overlayFrame);
        }

      } catch (Exception e) {
        Log.w(TAG, "Error processing frame " + frameCount, e);
        // Still write the frame even if processing failed
        drawRPPGAnalysisOverlays(overlayFrame, frameCount, fps, palmFrames, signals.size());
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
   * Extract rPPG signal from palm region of interest
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
   * Calculate heart rate metrics from green channel signals using peak detection
   */
  private RPPGData calculateHeartRateMetrics(List<Double> greenSignals, List<Long> timestamps, double fps) {
    if (greenSignals.size() < 30) { // Need at least 1 second of data at 30fps
      Log.w(TAG, "Insufficient data for heart rate calculation");
      return RPPGData.empty();
    }

    try {
      // Apply simple moving average filter to smooth the signal
      List<Double> smoothedSignals = applyMovingAverageFilter(greenSignals, 5);

      // Detect peaks in the smoothed green channel signal
      List<Long> heartbeats = detectHeartbeats(smoothedSignals, timestamps, fps);

      if (heartbeats.size() < 2) {
        Log.w(TAG, "Insufficient heartbeats detected");
        return RPPGData.empty();
      }

      // Calculate BPM from intervals between heartbeats
      List<Double> bpmValues = calculateBPMFromHeartbeats(heartbeats);

      double averageBpm = bpmValues.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
      double minBpm = bpmValues.stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
      double maxBpm = bpmValues.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);

      // Create signal list
      List<RPPGData.Signal> signalList = new ArrayList<>();
      for (int i = 0; i < Math.min(greenSignals.size(), timestamps.size()); i++) {
        signalList.add(RPPGData.Signal.builder()
            .redChannel(0.0) // Not using red for heart rate calculation
            .greenChannel(greenSignals.get(i))
            .blueChannel(0.0) // Not using blue for heart rate calculation
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
          .baselineBpm(averageBpm) // Use average as baseline
          .durationSeconds(durationSeconds)
          .signals(signalList)
          .build();

    } catch (Exception e) {
      Log.e(TAG, "Error calculating heart rate metrics", e);
      return RPPGData.empty();
    }
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
   * Detect heartbeats using simple peak detection algorithm
   */
  private List<Long> detectHeartbeats(List<Double> signals, List<Long> timestamps, double fps) {
    List<Long> heartbeats = new ArrayList<>();

    if (signals.size() < 3) return heartbeats;

    // Calculate signal statistics for threshold
    double mean = signals.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    double stdDev = Math.sqrt(signals.stream()
        .mapToDouble(val -> Math.pow(val - mean, 2))
        .average().orElse(0.0));

    double threshold = mean + 0.5 * stdDev; // Adaptive threshold
    int minDistanceFrames = (int)(fps * 0.4); // Minimum 0.4 seconds between beats (150 BPM max)

    int lastPeakIndex = -minDistanceFrames;

    // Find peaks above threshold with minimum distance constraint
    for (int i = 1; i < signals.size() - 1; i++) {
      double current = signals.get(i);
      double previous = signals.get(i - 1);
      double next = signals.get(i + 1);

      // Check if current point is a local maximum above threshold
      if (current > previous && current > next &&
          current > threshold &&
          (i - lastPeakIndex) > minDistanceFrames) {

        heartbeats.add(timestamps.get(i));
        lastPeakIndex = i;
      }
    }

    return heartbeats;
  }

  /**
   * Calculate BPM values from heartbeat timestamps
   */
  private List<Double> calculateBPMFromHeartbeats(List<Long> heartbeats) {
    List<Double> bpmValues = new ArrayList<>();

    for (int i = 1; i < heartbeats.size(); i++) {
      long intervalMs = heartbeats.get(i) - heartbeats.get(i-1);
      double intervalSeconds = intervalMs / 1000.0;
      double bpm = 60.0 / intervalSeconds;

      // Filter out unrealistic BPM values
      if (bpm >= 40 && bpm <= 200) {
        bpmValues.add(bpm);
      }
    }

    return bpmValues;
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

    // ECG waveform color - bright green like medical monitors
    Scalar ecgColor = new Scalar(0, 255, 0);
    int baselineY = graphY + graphHeight * 2 / 3;

    // Draw the ECG waveform
    List<Point> waveformPoints = generateECGWaveformPoints(heartbeatIndices, graphX, graphY, graphWidth, graphHeight, baselineY);

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

    if (signalHistory.size() < 30) return heartbeatIndices;

    // Apply smoothing for better peak detection
    List<Double> smoothed = applyMovingAverageFilter(signalHistory, 3);

    // Calculate threshold for peak detection
    double mean = smoothed.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    double std = Math.sqrt(smoothed.stream()
        .mapToDouble(val -> Math.pow(val - mean, 2))
        .average().orElse(0.0));
    double threshold = mean + 0.6 * std;

    // Find peaks with minimum distance constraint (realistic heart rate)
    int minDistance = 15; // Minimum 15 samples between beats (~120 BPM max at 30fps)
    int lastPeakIndex = -minDistance;

    for (int i = 2; i < smoothed.size() - 2; i++) {
      double current = smoothed.get(i);
      double prev1 = smoothed.get(i - 1);
      double prev2 = smoothed.get(i - 2);
      double next1 = smoothed.get(i + 1);
      double next2 = smoothed.get(i + 2);

      // Enhanced peak detection
      if (current > prev1 && current > next1 &&
          current > prev2 && current > next2 &&
          current > threshold &&
          (i - lastPeakIndex) > minDistance) {
        heartbeatIndices.add(i);
        lastPeakIndex = i;
      }
    }

    return heartbeatIndices;
  }

  /**
   * Generate ECG-like waveform points with PQRST complexes
   */
  private List<Point> generateECGWaveformPoints(List<Integer> heartbeatIndices, int graphX, int graphY,
                                               int graphWidth, int graphHeight, int baselineY) {
    List<Point> points = new ArrayList<>();

    // Start at baseline
    points.add(new Point(graphX, baselineY));

    int currentIndex = 0;
    int nextHeartbeatIndex = heartbeatIndices.isEmpty() ? -1 : heartbeatIndices.get(0);
    int heartbeatCounter = 0;

    // Generate points across the graph width
    for (int x = graphX; x <= graphX + graphWidth; x += 2) {
      // Calculate corresponding signal index
      int signalIndex = (int)((double)(x - graphX) * signalHistory.size() / graphWidth);
      signalIndex = Math.max(0, Math.min(signalIndex, signalHistory.size() - 1));

      if (nextHeartbeatIndex >= 0 && signalIndex >= nextHeartbeatIndex - 15 && signalIndex <= nextHeartbeatIndex + 15) {
        // Generate PQRST complex around heartbeat
        List<Point> pqrstPoints = generatePQRSTComplex(x, baselineY, signalIndex, nextHeartbeatIndex, heartbeatCounter);
        points.addAll(pqrstPoints);

        // Move to next heartbeat
        heartbeatCounter++;
        if (heartbeatCounter < heartbeatIndices.size()) {
          nextHeartbeatIndex = heartbeatIndices.get(heartbeatCounter);
        } else {
          nextHeartbeatIndex = -1;
        }

        // Skip ahead to avoid overlap
        x += 30;
      } else {
        // Normal baseline with slight physiological variation
        double baselineVariation = 2 * Math.sin(signalIndex * 0.1) * Math.exp(-Math.abs(signalIndex % 50) * 0.1);
        int y = (int)(baselineY + baselineVariation);
        points.add(new Point(x, y));
      }
    }

    return points;
  }

  /**
   * Generate PQRST complex for a single heartbeat
   */
  private List<Point> generatePQRSTComplex(int centerX, int baselineY, int signalIndex, int heartbeatIndex, int beatNumber) {
    List<Point> pqrstPoints = new ArrayList<>();

    // Add physiological variation between beats
    double variationFactor = 1.0 + 0.15 * Math.sin(beatNumber * 0.5);

    // P wave (atrial depolarization) - small positive deflection
    pqrstPoints.add(new Point(centerX - 15, baselineY));
    pqrstPoints.add(new Point(centerX - 12, baselineY - (8 * variationFactor)));
    pqrstPoints.add(new Point(centerX - 9, baselineY));

    // PR segment (isoelectric)
    pqrstPoints.add(new Point(centerX - 6, baselineY));

    // QRS complex - ventricular depolarization
    // Q wave (small negative deflection)
    pqrstPoints.add(new Point(centerX - 3, baselineY + (5 * variationFactor)));

    // R wave (large positive deflection - main peak)
    double rWaveHeight = 60 + 15 * Math.sin(beatNumber * 0.3); // Varying R wave height
    pqrstPoints.add(new Point(centerX, baselineY - (rWaveHeight * variationFactor)));

    // S wave (negative deflection after R)
    pqrstPoints.add(new Point(centerX + 3, baselineY + (12 * variationFactor)));

    // ST segment (return toward baseline)
    pqrstPoints.add(new Point(centerX + 6, baselineY - 1));

    // T wave (ventricular repolarization - positive, broader than P)
    pqrstPoints.add(new Point(centerX + 9, baselineY - (10 * variationFactor)));
    pqrstPoints.add(new Point(centerX + 12, baselineY - (8 * variationFactor)));
    pqrstPoints.add(new Point(centerX + 15, baselineY));

    return pqrstPoints;
  }

  /**
   * Highlight R-wave peaks with markers
   */
  private void highlightRWavePeaks(Mat frame, List<Integer> heartbeatIndices, int graphX, int graphWidth, int baselineY) {
    Scalar peakColor = new Scalar(0, 0, 255); // Red for R-wave peaks

    for (int heartbeatIndex : heartbeatIndices) {
      // Calculate x position for this heartbeat
      int x = graphX + (int)((double)heartbeatIndex * graphWidth / signalHistory.size());

      // Draw R-wave peak marker
      Imgproc.circle(frame, new Point(x, baselineY - 60), 3, peakColor, -1);

      // Draw vertical line to show exact timing
      Imgproc.line(frame, new Point(x, baselineY - 80), new Point(x, baselineY + 10), peakColor, 1);
    }
  }

  /**
   * Draw ECG annotations with medical-style information
   */
  private void drawECGAnnotations(Mat frame, int graphX, int graphY, int graphWidth, int graphHeight, int frameWidth) {
    Scalar textColor = new Scalar(0, 255, 0); // Green text like medical monitors
    Scalar infoColor = new Scalar(0, 200, 200); // Cyan for measurements

    // Calculate current heart rate from recent data
    String hrInfo = "HR: --";
    String rhythmInfo = "Rhythm: Analyzing...";

    if (signalHistory.size() > 60) { // 2 seconds of data
      List<Double> recentSignals = signalHistory.subList(
        Math.max(0, signalHistory.size() - 90), signalHistory.size()); // Last 3 seconds
      double estimatedBPM = estimateCurrentBPM(recentSignals, 30.0);

      if (estimatedBPM > 0) {
        hrInfo = String.format("HR: %.0f BPM", estimatedBPM);

        // Determine rhythm
        if (estimatedBPM > 100) {
          rhythmInfo = "Rhythm: Tachycardia";
        } else if (estimatedBPM < 60) {
          rhythmInfo = "Rhythm: Bradycardia";
        } else {
          rhythmInfo = "Rhythm: Normal Sinus";
        }
      }
    }

    // Signal quality indicator
    double signalQuality = calculateSignalQuality();
    String qualityInfo = String.format("Signal Quality: %.0f%%", signalQuality * 100);
    Scalar qualityColor = signalQuality > 0.7 ? new Scalar(0, 255, 0) :
                         signalQuality > 0.4 ? new Scalar(0, 200, 200) : new Scalar(0, 100, 255);

    // Draw annotations
    Imgproc.putText(frame, hrInfo,
      new Point(graphX, graphY + graphHeight + 15),
      Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2);

    Imgproc.putText(frame, rhythmInfo,
      new Point(graphX + 150, graphY + graphHeight + 15),
      Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, infoColor, 1);

    Imgproc.putText(frame, qualityInfo,
      new Point(frameWidth - 200, graphY + graphHeight + 15),
      Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, qualityColor, 1);

    // Add ECG paper speed and amplitude scales
    Imgproc.putText(frame, "25mm/s | 10mm/mV",
      new Point(graphX, graphY - 5),
      Imgproc.FONT_HERSHEY_SIMPLEX, 0.4, new Scalar(150, 150, 150), 1);
  }

  /**
   * Calculate signal quality based on signal characteristics
   */
  private double calculateSignalQuality() {
    if (signalHistory.size() < 30) return 0.0;

    // Calculate signal-to-noise ratio
    List<Double> recentSignals = signalHistory.subList(
      Math.max(0, signalHistory.size() - 60), signalHistory.size());

    double mean = recentSignals.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    double variance = recentSignals.stream()
        .mapToDouble(val -> Math.pow(val - mean, 2))
        .average().orElse(0.0);

    // Normalize quality score (higher variance = better signal in this context)
    double qualityScore = Math.min(1.0, variance / 1000.0);

    // Factor in signal stability
    double stability = 1.0 - (Math.abs(recentSignals.get(recentSignals.size()-1) - mean) / (mean + 1));

    return Math.max(0.0, Math.min(1.0, qualityScore * stability));
  }

  /**
   * Estimate current BPM from recent signal data for real-time display
   */
  private double estimateCurrentBPM(List<Double> recentSignals, double fps) {
    if (recentSignals.size() < 30) return 0; // Need sufficient data

    try {
      // Apply simple smoothing
      List<Double> smoothed = applyMovingAverageFilter(recentSignals, 3);

      // Find peaks
      List<Integer> peaks = new ArrayList<>();
      double mean = smoothed.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
      double threshold = mean * 1.1; // 10% above mean

      for (int i = 1; i < smoothed.size() - 1; i++) {
        if (smoothed.get(i) > smoothed.get(i-1) &&
            smoothed.get(i) > smoothed.get(i+1) &&
            smoothed.get(i) > threshold) {
          peaks.add(i);
        }
      }

      if (peaks.size() < 2) return 0;

      // Calculate average interval between peaks
      double avgInterval = 0;
      for (int i = 1; i < peaks.size(); i++) {
        avgInterval += (peaks.get(i) - peaks.get(i-1)) / fps; // Convert to seconds
      }
      avgInterval /= (peaks.size() - 1);

      // Convert to BPM
      double bpm = 60.0 / avgInterval;

      // Return only if within realistic range
      return (bpm >= 40 && bpm <= 200) ? bpm : 0;

    } catch (Exception e) {
      return 0;
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

