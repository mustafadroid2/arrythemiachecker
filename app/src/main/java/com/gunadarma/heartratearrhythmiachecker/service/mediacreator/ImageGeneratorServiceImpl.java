package com.gunadarma.heartratearrhythmiachecker.service.mediacreator;

import android.content.Context;
import android.util.Log;

import com.gunadarma.heartratearrhythmiachecker.constant.AppConstant;
import com.gunadarma.heartratearrhythmiachecker.model.RPPGData;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class ImageGeneratorServiceImpl {

  private final Context context;
  private static final String TAG = "ImageGeneratorService";

  public ImageGeneratorServiceImpl(Context context) {
    this.context = context;
  }

  public void createHeartBeatsImage(RPPGData rppgData, Long id) {
    if (rppgData == null || rppgData.getSignals().isEmpty()) {
      Log.w(TAG, "No rPPG signal data available for image generation");
      return;
    }

    Log.d(TAG, "Generating heartbeats image with " + rppgData.getSignals().size() + " rPPG signals and " + rppgData.getHeartbeats().size() + " heartbeats");

    // Create an ECG-style visualization
    final int WIDTH = 1200;
    final int HEIGHT = 400;
    final int PADDING = 60;
    final int GRID_SIZE = 20;

    Mat graph = Mat.zeros(HEIGHT, WIDTH, CvType.CV_8UC3);

    // Draw ECG paper-like background (light cream color)
    graph.setTo(new Scalar(248, 248, 240));

    // Extract rPPG signal timestamps
    List<Long> signalTimestamps = new ArrayList<>();
    for (com.gunadarma.heartratearrhythmiachecker.model.RPPGData.Signal signal : rppgData.getSignals()) {
      signalTimestamps.add(signal.getTimestamp());
    }

    if (signalTimestamps.isEmpty()) {
      Log.w(TAG, "No signal timestamps found");
      return;
    }

    // Calculate time scale for signal data
    long startTime = signalTimestamps.get(0);
    long endTime = signalTimestamps.get(signalTimestamps.size() - 1);
    long totalDuration = endTime - startTime;
    double pixelsPerMs = (WIDTH - 2.0 * PADDING) / (double)totalDuration;

    // Draw ECG paper grid pattern
    drawECGGrid(graph, WIDTH, HEIGHT, GRID_SIZE);

    // Draw ECG baseline
    int baselineY = HEIGHT * 2 / 3;
    Scalar baselineColor = new Scalar(180, 180, 180);
    Imgproc.line(graph, new Point(PADDING, baselineY),
      new Point(WIDTH - PADDING, baselineY),
      baselineColor, 1, Imgproc.LINE_AA);

    // Generate simple frequency-based waveform from heartbeat detection timestamps
    List<Point> ecgWaveform = generateFrequencyBasedWaveform(rppgData.getHeartbeats(), signalTimestamps,
                                                            startTime, pixelsPerMs, baselineY, PADDING, WIDTH);

    // Draw the waveform
    drawECGWaveform(graph, ecgWaveform);

    // Add annotations using actual rPPG data
    addECGAnnotationsFromRPPGData(graph, rppgData, WIDTH, HEIGHT, PADDING, baselineY);

    // Save the image
    String imagePath = String.format("%s/%s/%s/heartbeats.jpg",
        context.getExternalFilesDir(null).getAbsolutePath(),
        AppConstant.DATA_DIR, id);

    File imageFile = new File(imagePath);
    File parentDir = imageFile.getParentFile();
    if (parentDir != null && !parentDir.exists()) {
      parentDir.mkdirs();
    }

    boolean success = org.opencv.imgcodecs.Imgcodecs.imwrite(imagePath, graph);
    if (success) {
      Log.d(TAG, "Heartbeats image saved successfully: " + imagePath);
    } else {
      Log.e(TAG, "Failed to save heartbeats image: " + imagePath);
    }

    graph.release();
  }

  /**
   * Generate frequency-based waveform that shows smooth rounded frequency patterns
   * instead of sharp ECG-style signals
   */
  private List<Point> generateFrequencyBasedWaveform(List<Long> heartbeats, List<Long> signalTimestamps,
                                                    long startTime, double pixelsPerMs, int baselineY,
                                                    int padding, int width) {
    List<Point> waveform = new ArrayList<>();

    Log.d(TAG, "Generating smooth rounded frequency waveform from " + heartbeats.size() + " heartbeats and " + signalTimestamps.size() + " signal points");

    // Calculate the total width for drawing
    int totalWidth = width - 2 * padding;

    // Increase sample count for smoother curves
    int sampleCount = totalWidth; // One point per pixel for maximum smoothness

    // First pass: calculate raw amplitudes
    List<Double> rawAmplitudes = new ArrayList<>();
    for (int i = 0; i < sampleCount; i++) {
      long currentTime = startTime + (long)((i * (signalTimestamps.get(signalTimestamps.size() - 1) - startTime)) / (double)(sampleCount - 1));

      // Calculate amplitude for this time point
      double amplitude = calculateFrequencySignalAtTime(currentTime, heartbeats, signalTimestamps);
      rawAmplitudes.add(amplitude);
    }

    // Second pass: apply moving average smoothing for rounded appearance
    List<Double> smoothedAmplitudes = applyMovingAverageSmoothing(rawAmplitudes, 5);

    // Third pass: create final waveform points
    for (int i = 0; i < sampleCount; i++) {
      double x = padding + (i * totalWidth) / (double)(sampleCount - 1);
      double y = baselineY + smoothedAmplitudes.get(i);

      waveform.add(new Point(x, y));
    }

    return waveform;
  }

  /**
   * Apply moving average smoothing to create more rounded curves
   */
  private List<Double> applyMovingAverageSmoothing(List<Double> values, int windowSize) {
    List<Double> smoothed = new ArrayList<>();
    int halfWindow = windowSize / 2;

    for (int i = 0; i < values.size(); i++) {
      double sum = 0.0;
      int count = 0;

      // Calculate average within window
      for (int j = Math.max(0, i - halfWindow); j <= Math.min(values.size() - 1, i + halfWindow); j++) {
        sum += values.get(j);
        count++;
      }

      smoothed.add(sum / count);
    }

    return smoothed;
  }

  /**
   * Calculate frequency signal amplitude at a given time based on heartbeat detections
   * and data interpolation timestamps - creates smooth, rounded frequency graph style
   */
  private double calculateFrequencySignalAtTime(long currentTime, List<Long> heartbeats, List<Long> signalTimestamps) {
    double amplitude = 0.0;

    // Create smooth rounded peaks at heartbeat detection points - INCREASED 10x for visibility
    for (Long heartbeat : heartbeats) {
      long distance = Math.abs(currentTime - heartbeat);

      // Create wider, smoother peaks with Gaussian-like shape
      if (distance < 800) { // Wider influence area for smoother transitions
        double normalizedDistance = distance / 800.0; // Normalize to 0-1

        // Use Gaussian-like envelope for smooth, rounded peaks
        double gaussianEnvelope = Math.exp(-Math.pow(normalizedDistance * 3, 2));

        // Add subtle frequency modulation for visual interest
        double timeFromBeat = (currentTime - heartbeat) / 1000.0;
        double frequencyModulation = 1.0 + 0.3 * Math.sin(2 * Math.PI * 8.0 * timeFromBeat);

        // Create main rounded peak - INCREASED FROM 180 TO 1800 (10x higher)
        amplitude += gaussianEnvelope * 1800 * frequencyModulation;

        // Add harmonic component for richer frequency appearance - INCREASED FROM 60 TO 600 (10x higher)
        amplitude += gaussianEnvelope * 600 * Math.sin(2 * Math.PI * 12.0 * timeFromBeat) *
                    Math.exp(-normalizedDistance * 2);
      }
    }

    // Add continuous background frequency from signal data for smoother baseline - INCREASED 10x
    double backgroundFreq = 0.0;
    for (Long signalTime : signalTimestamps) {
      long signalDistance = Math.abs(currentTime - signalTime);

      if (signalDistance < 300) { // Smoother background signals
        double normalizedSignalDistance = signalDistance / 300.0;
        double smoothEnvelope = Math.exp(-Math.pow(normalizedSignalDistance * 2, 2));

        double timeFromSignal = (currentTime - signalTime) / 1000.0;
        // INCREASED FROM 25 TO 250 (10x higher)
        backgroundFreq += smoothEnvelope * 250 *
                         (Math.sin(2 * Math.PI * 25.0 * timeFromSignal) +
                          0.5 * Math.sin(2 * Math.PI * 40.0 * timeFromSignal));
      }
    }

    amplitude += backgroundFreq;

    // Apply smoothing filter to reduce sharp transitions
    return applySmoothing(amplitude);
  }

  /**
   * Apply smoothing filter for more rounded appearance - ADJUSTED for higher amplitudes
   */
  private double applySmoothing(double value) {
    // Use tanh for natural compression and smoothing - ADJUSTED scaling for 10x higher values
    return Math.tanh(value / 1000.0) * 1000.0;
  }

  /**
   * Add ECG-style annotations using actual rPPG data statistics
   */
  private void addECGAnnotationsFromRPPGData(Mat graph, RPPGData rppgData, int width, int height, int padding, int baselineY) {
    Scalar textColor = new Scalar(50, 50, 50); // Dark gray text
    Scalar measurementColor = new Scalar(0, 0, 150); // Blue for measurements

    // Use actual rPPG data statistics
    double avgBpm = rppgData.getAverageBpm();
    double minBpm = rppgData.getMinBpm();
    double maxBpm = rppgData.getMaxBpm();
    int duration = rppgData.getDurationSeconds();
    int heartbeatCount = rppgData.getHeartbeats().size();
    int signalCount = rppgData.getSignals().size();

    // Calculate signal quality from actual data
    double signalQuality = calculateRPPGSignalQuality(rppgData);

    // Add medical-style annotations with real data
    String hrText = String.format(Locale.US, "HR: %.1f BPM (%.1f-%.1f)", avgBpm, minBpm, maxBpm);
    String durationText = String.format(Locale.US, "Duration: %d sec (%d beats)", duration, heartbeatCount);
    String qualityText = String.format(Locale.US, "Signal Quality: %.0f%% (%d samples)", signalQuality * 100, signalCount);
    String rhythmText = determineRhythmFromRPPGData(rppgData);

    // Draw text annotations
    Imgproc.putText(graph, hrText, new Point(padding, 30),
        Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, textColor, 2);
    Imgproc.putText(graph, durationText, new Point(padding, 55),
        Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2);
    Imgproc.putText(graph, qualityText, new Point(padding, 80),
        Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, measurementColor, 2);
    Imgproc.putText(graph, rhythmText, new Point(padding, 105),
        Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, measurementColor, 2);

    // Add time scale markers
    addTimeScaleMarkers(graph, width, height, padding, baselineY);

    // Add amplitude scale
    addAmplitudeScale(graph, height, padding, baselineY);
  }

  /**
   * Calculate signal quality from actual rPPG data
   */
  private double calculateRPPGSignalQuality(RPPGData rppgData) {
    if (rppgData.getSignals().isEmpty()) return 0.0;

    // Calculate signal-to-noise ratio from green channel values
    List<Double> greenValues = new ArrayList<>();
    for (com.gunadarma.heartratearrhythmiachecker.model.RPPGData.Signal signal : rppgData.getSignals()) {
      greenValues.add(signal.getGreenChannel());
    }

    double mean = greenValues.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    double variance = greenValues.stream()
        .mapToDouble(val -> Math.pow(val - mean, 2))
        .average().orElse(0.0);

    // Simple quality metric: higher signal amplitude with lower relative variance indicates better quality
    double snr = mean > 0 ? mean / Math.sqrt(variance + 1) : 0;
    return Math.max(0.0, Math.min(1.0, snr / 50.0)); // Normalize to 0-1 range
  }

  /**
   * Determine rhythm classification from actual rPPG data
   */
  private String determineRhythmFromRPPGData(RPPGData rppgData) {
    List<Long> heartbeats = rppgData.getHeartbeats();
    double avgBpm = rppgData.getAverageBpm();

    if (heartbeats.size() < 2) return "Rhythm: Insufficient data";

    // Calculate RR interval variability from actual heartbeat data
    List<Long> rrIntervals = new ArrayList<>();
    for (int i = 1; i < heartbeats.size(); i++) {
      rrIntervals.add(heartbeats.get(i) - heartbeats.get(i - 1));
    }

    double avgRR = rrIntervals.stream().mapToLong(Long::longValue).average().orElse(0);
    double variability = rrIntervals.stream()
        .mapToDouble(rr -> Math.abs(rr - avgRR))
        .average().orElse(0);

    double variabilityPercent = avgRR > 0 ? (variability / avgRR) * 100 : 0;

    // Classify rhythm based on actual rPPG data
    if (avgBpm > 100) {
      return "Rhythm: Tachycardia";
    } else if (avgBpm < 60) {
      return "Rhythm: Bradycardia";
    } else if (variabilityPercent > 15) {
      return "Rhythm: Irregular";
    } else {
      return "Rhythm: Normal Sinus";
    }
  }

  /**
   * Draw ECG paper-style grid pattern
   */
  private void drawECGGrid(Mat graph, int width, int height, int gridSize) {
    // Fine grid lines (1mm equivalent)
    Scalar fineGridColor = new Scalar(230, 200, 200); // Light pink
    for (int x = 0; x < width; x += gridSize / 5) {
      Imgproc.line(graph, new Point(x, 0), new Point(x, height), fineGridColor, 1);
    }
    for (int y = 0; y < height; y += gridSize / 5) {
      Imgproc.line(graph, new Point(0, y), new Point(width, y), fineGridColor, 1);
    }

    // Major grid lines (5mm equivalent)
    Scalar majorGridColor = new Scalar(200, 150, 150); // Darker pink
    for (int x = 0; x < width; x += gridSize) {
      Imgproc.line(graph, new Point(x, 0), new Point(x, height), majorGridColor, 1);
    }
    for (int y = 0; y < height; y += gridSize) {
      Imgproc.line(graph, new Point(0, y), new Point(width, y), majorGridColor, 1);
    }

    // Bold grid lines (25mm equivalent - 1 second markers)
    Scalar boldGridColor = new Scalar(180, 120, 120);
    for (int x = 0; x < width; x += gridSize * 5) {
      Imgproc.line(graph, new Point(x, 0), new Point(x, height), boldGridColor, 2);
    }
    for (int y = 0; y < height; y += gridSize * 5) {
      Imgproc.line(graph, new Point(0, y), new Point(width, y), boldGridColor, 2);
    }
  }

  /**
   * Draw the ECG waveform with medical device styling
   */
  private void drawECGWaveform(Mat graph, List<Point> waveform) {
    if (waveform.size() < 2) return;

    // Draw ECG trace with characteristic medical device appearance
    Scalar ecgColor = new Scalar(0, 100, 0); // Dark green like medical monitors
    int thickness = 2;

    // Draw smooth lines between points
    for (int i = 1; i < waveform.size(); i++) {
      Point p1 = waveform.get(i - 1);
      Point p2 = waveform.get(i);

      Imgproc.line(graph, p1, p2, ecgColor, thickness, Imgproc.LINE_AA);
    }

    // Highlight frequency burst peaks for better visibility
    Scalar peakColor = new Scalar(0, 0, 200); // Red for signal peaks
    for (int i = 1; i < waveform.size() - 1; i++) {
      Point prev = waveform.get(i - 1);
      Point curr = waveform.get(i);
      Point next = waveform.get(i + 1);

      // Detect significant signal peaks (local maxima with significant amplitude)
      if (curr.y < prev.y && curr.y < next.y &&
          Math.abs(curr.y - prev.y) > 15 && Math.abs(curr.y - next.y) > 15) {
        Imgproc.circle(graph, curr, 2, peakColor, -1);
      }
    }
  }

  /**
   * Add time scale markers like medical ECG strips
   */
  private void addTimeScaleMarkers(Mat graph, int width, int height, int padding, int baselineY) {
    Scalar markerColor = new Scalar(100, 100, 100);
    Scalar textColor = new Scalar(80, 80, 80);

    // Add second markers every 200 pixels (representing 1 second intervals)
    int secondInterval = 200;
    for (int x = padding; x < width - padding; x += secondInterval) {
      // Draw vertical time markers
      Imgproc.line(graph, new Point(x, baselineY - 5), new Point(x, baselineY + 5), markerColor, 2);

      // Add time labels
      int seconds = (x - padding) / secondInterval;
      if (seconds > 0) {
        Imgproc.putText(graph, seconds + "s", new Point(x - 10, baselineY + 25),
            Imgproc.FONT_HERSHEY_SIMPLEX, 0.4, textColor, 1);
      }
    }
  }

  /**
   * Add amplitude scale markers like medical ECG
   */
  private void addAmplitudeScale(Mat graph, int height, int padding, int baselineY) {
    Scalar markerColor = new Scalar(100, 100, 100);
    Scalar textColor = new Scalar(80, 80, 80);

    // Add amplitude markers every 20 pixels (representing signal levels)
    int amplitudeInterval = 20;
    for (int y = baselineY - amplitudeInterval; y > padding; y -= amplitudeInterval) {
      // Positive amplitude markers
      Imgproc.line(graph, new Point(padding - 5, y), new Point(padding + 5, y), markerColor, 1);

      int amplitudeLevel = (baselineY - y) / amplitudeInterval;
      if (amplitudeLevel > 0) {
        Imgproc.putText(graph, "+" + amplitudeLevel, new Point(padding - 30, y + 3),
            Imgproc.FONT_HERSHEY_SIMPLEX, 0.3, textColor, 1);
      }
    }

    for (int y = baselineY + amplitudeInterval; y < height - padding; y += amplitudeInterval) {
      // Negative amplitude markers
      Imgproc.line(graph, new Point(padding - 5, y), new Point(padding + 5, y), markerColor, 1);

      int amplitudeLevel = (y - baselineY) / amplitudeInterval;
      Imgproc.putText(graph, "-" + amplitudeLevel, new Point(padding - 30, y + 3),
          Imgproc.FONT_HERSHEY_SIMPLEX, 0.3, textColor, 1);
    }
  }
}
