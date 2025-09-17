package com.gunadarma.heartratearrhythmiachecker.service.mediacreator;

import android.content.Context;
import android.util.Log;

import com.gunadarma.heartratearrhythmiachecker.constant.AppConstant;
import com.gunadarma.heartratearrhythmiachecker.model.HeartRateData;
import com.gunadarma.heartratearrhythmiachecker.model.RPPGData;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

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

    // Create an ECG-style visualization with medical device appearance
    final int WIDTH = 1200;  // Wider for better ECG strip visualization
    final int HEIGHT = 400;   // Taller for realistic ECG amplitude
    final int PADDING = 60;   // More padding for labels and scales
    final int GRID_SIZE = 20; // Finer grid like medical ECG paper

    Mat graph = Mat.zeros(HEIGHT, WIDTH, CvType.CV_8UC3);

    // Draw ECG paper-like background (light cream color)
    graph.setTo(new Scalar(248, 248, 240));

    // Extract actual rPPG signal data
    List<Double> rppgSignalValues = new ArrayList<>();
    List<Long> signalTimestamps = new ArrayList<>();

    for (com.gunadarma.heartratearrhythmiachecker.model.RPPGData.Signal signal : rppgData.getSignals()) {
      rppgSignalValues.add(signal.getGreenChannel()); // Use green channel for rPPG
      signalTimestamps.add(signal.getTimestamp());
    }

    if (rppgSignalValues.isEmpty()) {
      Log.w(TAG, "No rPPG signal values found");
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
    int baselineY = HEIGHT * 2 / 3; // Position baseline in lower 2/3 for realistic ECG
    Scalar baselineColor = new Scalar(180, 180, 180);
    Imgproc.line(graph, new Point(PADDING, baselineY),
      new Point(WIDTH - PADDING, baselineY),
      baselineColor, 1, Imgproc.LINE_AA);

    // Apply signal processing to the actual rPPG data (similar to video processing)
    List<Double> processedSignal = applyECGStyleProcessing(rppgSignalValues);

    // Generate ECG-like waveform from the processed rPPG signal data
    List<Point> ecgWaveform = generateECGWaveformFromRPPGSignals(processedSignal, signalTimestamps, rppgData.getHeartbeats(),
                                                                 startTime, pixelsPerMs, baselineY, PADDING);

    // Draw the ECG waveform with medical device styling
    drawECGWaveform(graph, ecgWaveform);

    // Add ECG-style annotations and measurements using actual rPPG data
    addECGAnnotationsFromRPPGData(graph, rppgData, WIDTH, HEIGHT, PADDING, baselineY);

    // Save the ECG-style image
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
   * Apply ECG-style processing to the actual rPPG signal data
   * This ensures the image uses the same signal processing as the video overlay
   */
  private List<Double> applyECGStyleProcessing(List<Double> rppgSignals) {
    if (rppgSignals.size() < 10) return rppgSignals;

    Log.d(TAG, "Applying ECG-style processing to " + rppgSignals.size() + " rPPG signal samples");

    // Step 1: DC removal (remove baseline drift) - same as video processing
    List<Double> dcRemoved = removeDCComponent(rppgSignals);

    // Step 2: Apply band-pass filtering for heart rate frequencies (0.5-4 Hz)
    List<Double> filtered = applyBandPassFilter(dcRemoved);

    // Step 3: Apply smoothing filter
    List<Double> smoothed = applyMovingAverageFilter(filtered, 3);

    // Step 4: Normalize and scale for ECG visualization
    List<Double> normalized = normalizeForECGVisualization(smoothed);

    return normalized;
  }

  /**
   * Apply band-pass filter for heart rate frequencies (0.5-4 Hz / 30-240 BPM)
   */
  private List<Double> applyBandPassFilter(List<Double> signal) {
    if (signal.size() < 5) return signal;

    // Simple butterworth-like filter approximation
    List<Double> filtered = new ArrayList<>();

    // Initialize with first few samples
    for (int i = 0; i < Math.min(3, signal.size()); i++) {
      filtered.add(signal.get(i));
    }

    // Apply filter
    for (int i = 3; i < signal.size(); i++) {
      // Simple band-pass filter approximation
      double highPass = signal.get(i) - signal.get(i-1);
      double bandPass = 0.1 * highPass + 0.9 * (i > 3 ? filtered.get(i-1) : highPass);
      filtered.add(bandPass);
    }

    return filtered;
  }

  /**
   * Normalize signal specifically for ECG visualization
   */
  private List<Double> normalizeForECGVisualization(List<Double> signal) {
    if (signal.isEmpty()) return signal;

    double mean = signal.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    double std = Math.sqrt(signal.stream()
        .mapToDouble(val -> Math.pow(val - mean, 2))
        .average().orElse(1.0));

    if (std == 0) std = 1.0; // Avoid division by zero

    List<Double> normalized = new ArrayList<>();
    for (Double value : signal) {
      // Normalize to z-score, then scale for ECG amplitude (±40 pixels)
      double normalizedValue = ((value - mean) / std) * 40.0;
      normalized.add(normalizedValue);
    }

    return normalized;
  }

  /**
   * Generate ECG waveform directly from processed rPPG signal data
   * This creates the waveform that matches what's shown in the video overlay
   */
  private List<Point> generateECGWaveformFromRPPGSignals(List<Double> processedSignal, List<Long> timestamps,
                                                        List<Long> heartbeats, long startTime, double pixelsPerMs,
                                                        int baselineY, int padding) {
    List<Point> waveform = new ArrayList<>();

    Log.d(TAG, "Generating ECG waveform from " + processedSignal.size() + " processed rPPG samples");

    // Generate waveform points directly from the processed rPPG signal
    for (int i = 0; i < Math.min(processedSignal.size(), timestamps.size()); i++) {
      long sampleTime = timestamps.get(i);
      double x = padding + (sampleTime - startTime) * pixelsPerMs;

      // Use the processed signal value directly (it's already normalized for display)
      double signalY = baselineY - processedSignal.get(i); // Negative because screen Y coordinates are inverted

      // Enhance with ECG morphology near actual heartbeat locations
      double enhancedY = enhanceWithRealECGMorphology(x, signalY, heartbeats, startTime, pixelsPerMs, baselineY, i, padding);

      waveform.add(new Point(x, enhancedY));
    }

    return waveform;
  }

  /**
   * Enhance signal with realistic ECG morphology near actual detected heartbeat locations
   */
  private double enhanceWithRealECGMorphology(double x, double baseY, List<Long> heartbeats, long startTime,
                                            double pixelsPerMs, int baselineY, int sampleIndex, int padding) {
    // Find nearest heartbeat from the actual detected heartbeats
    double nearestBeatX = -1;
    double minDistance = Double.MAX_VALUE;

    for (Long heartbeat : heartbeats) {
      double beatX = padding + (heartbeat - startTime) * pixelsPerMs;
      double distance = Math.abs(x - beatX);
      if (distance < minDistance) {
        minDistance = distance;
        nearestBeatX = beatX;
      }
    }

    // If we're close to an actual detected heartbeat, enhance with ECG morphology
    if (nearestBeatX > 0 && minDistance < 30) { // Within 30 pixels of a detected heartbeat
      double relativePos = (x - nearestBeatX) / 30.0; // Normalize to ±1
      double ecgEnhancement = generateRealisticECGMorphology(relativePos, sampleIndex);

      // Blend the enhancement with the actual signal data
      double blendFactor = Math.exp(-Math.pow(relativePos * 2.5, 2)); // Gaussian blend
      return baseY + ecgEnhancement * blendFactor * 0.7; // Reduce enhancement to let real signal show through
    }

    return baseY; // Return original processed signal value
  }

  /**
   * Generate realistic ECG morphology enhancement (PQRST complex)
   */
  private double generateRealisticECGMorphology(double relativePos, int sampleIndex) {
    // Add some variation between beats
    double variationFactor = 1.0 + 0.05 * Math.sin(sampleIndex * 0.1);

    // Generate PQRST complex based on relative position to detected heartbeat
    double amplitude = 0;

    if (relativePos >= -0.6 && relativePos <= -0.3) {
      // P wave - smaller, earlier
      double pPos = (relativePos + 0.45) / 0.15;
      amplitude = 5 * Math.exp(-Math.pow(pPos * 2, 2)) * variationFactor;
    } else if (relativePos >= -0.15 && relativePos <= 0.15) {
      // QRS complex - the main spike
      if (relativePos >= -0.05 && relativePos <= 0.05) {
        // R wave (main spike)
        double rPos = relativePos / 0.05;
        amplitude = 25 * Math.exp(-Math.pow(rPos * 3, 2)) * variationFactor;
      } else if (relativePos < -0.05) {
        // Q wave
        amplitude = -3 * variationFactor;
      } else {
        // S wave
        amplitude = -8 * variationFactor;
      }
    } else if (relativePos >= 0.2 && relativePos <= 0.6) {
      // T wave - broader, later
      double tPos = (relativePos - 0.4) / 0.2;
      amplitude = 8 * Math.exp(-Math.pow(tPos * 1.5, 2)) * variationFactor;
    }

    return amplitude;
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
    String hrText = String.format("HR: %.1f BPM (%.1f-%.1f)", avgBpm, minBpm, maxBpm);
    String durationText = String.format("Duration: %d sec (%d beats)", duration, heartbeatCount);
    String qualityText = String.format("Signal Quality: %.0f%% (%d samples)", signalQuality * 100, signalCount);
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

  // Keep the original method for backward compatibility but mark as deprecated
  @Deprecated
  public void createHeartBeatsImage(List<Long> heartbeats, Long id) {
    Log.w(TAG, "Using deprecated createHeartBeatsImage method. Consider updating to use RPPGData.");

    if (heartbeats.isEmpty()) {
      Log.w(TAG, "No heartbeat data available for image generation");
      return;
    }

    Log.d(TAG, "Generating heartbeats image with " + heartbeats.size() + " heartbeats");

    // Create an ECG-style visualization with medical device appearance
    final int WIDTH = 1200;  // Wider for better ECG strip visualization
    final int HEIGHT = 400;   // Taller for realistic ECG amplitude
    final int PADDING = 60;   // More padding for labels and scales
    final int GRID_SIZE = 20; // Finer grid like medical ECG paper

    Mat graph = Mat.zeros(HEIGHT, WIDTH, CvType.CV_8UC3);

    // Draw ECG paper-like background (light cream color)
    graph.setTo(new Scalar(248, 248, 240));

    // Calculate time scale for heartbeat data
    long startTime = heartbeats.get(0);
    long endTime = heartbeats.get(heartbeats.size() - 1);
    long totalDuration = endTime - startTime;
    double pixelsPerMs = (WIDTH - 2.0 * PADDING) / (double)totalDuration;

    // Draw ECG paper grid pattern
    drawECGGrid(graph, WIDTH, HEIGHT, GRID_SIZE);

    // Draw ECG baseline
    int baselineY = HEIGHT * 2 / 3; // Position baseline in lower 2/3 for realistic ECG
    Scalar baselineColor = new Scalar(180, 180, 180);
    Imgproc.line(graph, new Point(PADDING, baselineY),
      new Point(WIDTH - PADDING, baselineY),
      baselineColor, 1, Imgproc.LINE_AA);

    // Generate interpolated signal data similar to video processing
    List<Double> interpolatedSignal = generateInterpolatedSignalFromHeartbeats(heartbeats, totalDuration);

    // Apply signal processing similar to video processing
    List<Double> processedSignal = applySignalProcessing(interpolatedSignal);

    // Generate realistic ECG-like waveform with interpolated data
    List<Point> ecgWaveform = generateAdvancedECGWaveform(heartbeats, processedSignal, startTime, pixelsPerMs, baselineY, PADDING);

    // Draw the ECG waveform with medical device styling
    drawECGWaveform(graph, ecgWaveform);

    // Add ECG-style annotations and measurements
    addECGAnnotations(graph, heartbeats, WIDTH, HEIGHT, PADDING, baselineY);

    // Save the ECG-style image
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
   * Generate realistic ECG waveform from heartbeat timing data
   */
  private List<Point> generateECGWaveform(List<Long> heartbeats, long startTime, double pixelsPerMs, int baselineY, int padding) {
    List<Point> waveform = new ArrayList<>();

    // Start with baseline
    waveform.add(new Point(padding, baselineY));

    for (int i = 0; i < heartbeats.size(); i++) {
      long beatTime = heartbeats.get(i);
      double beatX = padding + (beatTime - startTime) * pixelsPerMs;

      // Add points leading up to the heartbeat (P-R interval)
      if (i > 0) {
        long prevBeat = heartbeats.get(i - 1);
        double prevX = padding + (prevBeat - startTime) * pixelsPerMs;

        // Calculate RR interval for this beat
        double rrInterval = beatTime - prevBeat; // in milliseconds
        double normalizedRR = Math.max(0.5, Math.min(2.0, rrInterval / 800.0)); // Normalize around 800ms

        // Generate inter-beat segment (T-P segment from previous beat)
        generateInterBeatSegment(waveform, prevX, beatX, baselineY, normalizedRR);
      }

      // Generate PQRST complex for this heartbeat
      generatePQRSTComplex(waveform, beatX, baselineY, i);
    }

    // Add final baseline segment
    if (!waveform.isEmpty()) {
      Point lastPoint = waveform.get(waveform.size() - 1);
      waveform.add(new Point(lastPoint.x + 50, baselineY));
    }

    return waveform;
  }

  /**
   * Generate inter-beat segment (T-P segment)
   */
  private void generateInterBeatSegment(List<Point> waveform, double startX, double endX, int baselineY, double normalizedRR) {
    if (waveform.isEmpty()) return;

    Point lastPoint = waveform.get(waveform.size() - 1);
    double segmentLength = endX - startX;
    int numPoints = Math.max(5, (int)(segmentLength / 10)); // Adaptive point density

    for (int i = 1; i <= numPoints; i++) {
      double t = (double)i / numPoints;
      double x = startX + t * segmentLength;

      // Gradual return to baseline with slight physiological variation
      double y = lastPoint.y + (baselineY - lastPoint.y) * Math.pow(t, 0.7);

      // Add subtle respiratory and other physiological variations
      double variation = 2 * Math.sin(t * Math.PI * 2 * normalizedRR) * Math.exp(-t * 2);
      y += variation;

      waveform.add(new Point(x, y));
    }
  }

  /**
   * Generate PQRST complex for a single heartbeat
   */
  private void generatePQRSTComplex(List<Point> waveform, double centerX, int baselineY, int beatIndex) {
    // PQRST complex timing (in pixels, representing milliseconds)
    double pWaveStart = centerX - 25;
    double pWavePeak = centerX - 20;
    double pWaveEnd = centerX - 15;
    double prSegmentEnd = centerX - 5;
    double qWave = centerX - 3;
    double rWavePeak = centerX; // Main R wave peak
    double sWave = centerX + 3;
    double stSegmentEnd = centerX + 8;
    double tWavePeak = centerX + 15;
    double tWaveEnd = centerX + 25;

    // Add some physiological variation between beats
    double variationFactor = 1.0 + 0.1 * Math.sin(beatIndex * 0.7) * Math.exp(-beatIndex * 0.05);

    // P wave - atrial depolarization (small positive deflection)
    waveform.add(new Point(pWaveStart, baselineY));
    waveform.add(new Point(pWavePeak, baselineY - 8 * variationFactor));
    waveform.add(new Point(pWaveEnd, baselineY));

    // P-R segment (isoelectric)
    waveform.add(new Point(prSegmentEnd, baselineY));

    // QRS complex - ventricular depolarization
    // Q wave (small negative deflection)
    waveform.add(new Point(qWave, baselineY + 5 * variationFactor));

    // R wave (large positive deflection - main spike)
    double rHeight = 80 + 20 * Math.sin(beatIndex * 0.3); // Varying R wave height
    waveform.add(new Point(rWavePeak, baselineY - rHeight * variationFactor));

    // S wave (negative deflection after R)
    waveform.add(new Point(sWave, baselineY + 15 * variationFactor));

    // S-T segment (return toward baseline)
    waveform.add(new Point(stSegmentEnd, baselineY - 2));

    // T wave - ventricular repolarization (positive, broader than P)
    waveform.add(new Point(tWavePeak, baselineY - 12 * variationFactor));
    waveform.add(new Point(tWaveEnd, baselineY));
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

    // Highlight R wave peaks for better visibility
    Scalar peakColor = new Scalar(0, 0, 200); // Red for R wave peaks
    for (int i = 1; i < waveform.size() - 1; i++) {
      Point prev = waveform.get(i - 1);
      Point curr = waveform.get(i);
      Point next = waveform.get(i + 1);

      // Detect R wave peaks (local maxima with significant amplitude)
      if (curr.y < prev.y && curr.y < next.y &&
          Math.abs(curr.y - prev.y) > 30 && Math.abs(curr.y - next.y) > 30) {
        Imgproc.circle(graph, curr, 3, peakColor, -1);
      }
    }
  }

  /**
   * Add ECG-style annotations and measurements
   */
  private void addECGAnnotations(Mat graph, List<Long> heartbeats, int width, int height, int padding, int baselineY) {
    Scalar textColor = new Scalar(50, 50, 50); // Dark gray text
    Scalar measurementColor = new Scalar(0, 0, 150); // Blue for measurements

    // Calculate heart rate statistics
    if (heartbeats.size() > 1) {
      // Calculate average heart rate
      long totalDuration = heartbeats.get(heartbeats.size() - 1) - heartbeats.get(0);
      double avgHeartRate = (heartbeats.size() - 1) * 60000.0 / totalDuration;

      // Calculate RR interval statistics
      List<Long> rrIntervals = new ArrayList<>();
      for (int i = 1; i < heartbeats.size(); i++) {
        rrIntervals.add(heartbeats.get(i) - heartbeats.get(i - 1));
      }

      double avgRR = rrIntervals.stream().mapToLong(Long::longValue).average().orElse(0);
      double minRR = rrIntervals.stream().mapToLong(Long::longValue).min().orElse(0);
      double maxRR = rrIntervals.stream().mapToLong(Long::longValue).max().orElse(0);

      // Add medical-style annotations
      String hrText = String.format("HR: %.0f BPM", avgHeartRate);
      String rrText = String.format("RR: %.0f ms (%.0f-%.0f)", avgRR, minRR, maxRR);
      String rhythmText = determineRhythmClassification(rrIntervals, avgHeartRate);

      // Draw text annotations (using simple text drawing since we can't access complex fonts)
      Imgproc.putText(graph, hrText, new Point(padding, 30),
          Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, textColor, 2);
      Imgproc.putText(graph, rrText, new Point(padding, 55),
          Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2);
      Imgproc.putText(graph, rhythmText, new Point(padding, 80),
          Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, measurementColor, 2);

      // Add time scale markers
      addTimeScaleMarkers(graph, width, height, padding, baselineY);

      // Add amplitude scale
      addAmplitudeScale(graph, height, padding, baselineY);
    }
  }

  /**
   * Determine rhythm classification based on RR intervals
   */
  private String determineRhythmClassification(List<Long> rrIntervals, double avgHeartRate) {
    if (rrIntervals.isEmpty()) return "Rhythm: Insufficient data";

    // Calculate RR interval variability
    double avgRR = rrIntervals.stream().mapToLong(Long::longValue).average().orElse(0);
    double variability = rrIntervals.stream()
        .mapToDouble(rr -> Math.abs(rr - avgRR))
        .average().orElse(0);

    double variabilityPercent = (variability / avgRR) * 100;

    // Classify rhythm
    if (avgHeartRate > 100) {
      return "Rhythm: Tachycardia";
    } else if (avgHeartRate < 60) {
      return "Rhythm: Bradycardia";
    } else if (variabilityPercent > 15) {
      return "Rhythm: Irregular";
    } else {
      return "Rhythm: Normal Sinus";
    }
  }

  /**
   * Add time scale markers like medical ECG strips
   */
  private void addTimeScaleMarkers(Mat graph, int width, int height, int padding, int baselineY) {
    Scalar markerColor = new Scalar(100, 100, 100);

    // Add 1-second markers (standard ECG time scale)
    int timeStep = 60; // pixels per second (adjust based on your scale)
    for (int x = padding; x < width - padding; x += timeStep) {
      // Small tick marks
      Imgproc.line(graph, new Point(x, height - 20), new Point(x, height - 10), markerColor, 1);

      // Time labels every 5 seconds
      if ((x - padding) % (timeStep * 5) == 0) {
        int seconds = (x - padding) / timeStep;
        Imgproc.putText(graph, seconds + "s", new Point(x - 10, height - 5),
            Imgproc.FONT_HERSHEY_SIMPLEX, 0.4, markerColor, 1);
      }
    }
  }

  /**
   * Add amplitude scale like medical ECG strips
   */
  private void addAmplitudeScale(Mat graph, int height, int padding, int baselineY) {
    Scalar scaleColor = new Scalar(100, 100, 100);

    // Add amplitude reference marks
    int[] amplitudes = {-20, -10, 0, 10, 20}; // in "mV" equivalent
    for (int amp : amplitudes) {
      int y = baselineY - amp * 2; // Scale factor for visualization
      if (y > 0 && y < height) {
        Imgproc.line(graph, new Point(10, y), new Point(20, y), scaleColor, 1);
        if (amp != 0) {
          Imgproc.putText(graph, String.valueOf(amp), new Point(25, y + 5),
              Imgproc.FONT_HERSHEY_SIMPLEX, 0.3, scaleColor, 1);
        }
      }
    }
  }

  /**
   * Generate interpolated signal data similar to video processing
   * This mimics the continuous signal generation used in RPPGHandPalmServiceImpl
   */
  private List<Double> generateInterpolatedSignalFromHeartbeats(List<Long> heartbeats, long totalDuration) {
    List<Double> signal = new ArrayList<>();

    // Target sample rate similar to video processing (30 FPS equivalent)
    double sampleRate = 30.0; // samples per second
    int totalSamples = (int)(totalDuration * sampleRate / 1000.0);

    Log.d(TAG, "Generating " + totalSamples + " interpolated samples from " + heartbeats.size() + " heartbeats");

    // Generate signal with heart rate pattern
    long startTime = heartbeats.get(0);
    for (int i = 0; i < totalSamples; i++) {
      long currentTime = startTime + (long)(i * 1000.0 / sampleRate);

      // Find nearest heartbeat
      int nearestBeatIndex = findNearestHeartbeat(heartbeats, currentTime);

      // Calculate distance to nearest heartbeat
      double distanceToNearestBeat = Math.abs(currentTime - heartbeats.get(nearestBeatIndex));

      // Generate signal value based on proximity to heartbeat and RR intervals
      double signalValue = generateSignalValue(heartbeats, nearestBeatIndex, distanceToNearestBeat, currentTime, startTime);

      signal.add(signalValue);
    }

    return signal;
  }

  /**
   * Find the nearest heartbeat to a given time
   */
  private int findNearestHeartbeat(List<Long> heartbeats, long targetTime) {
    int nearestIndex = 0;
    long minDistance = Math.abs(heartbeats.get(0) - targetTime);

    for (int i = 1; i < heartbeats.size(); i++) {
      long distance = Math.abs(heartbeats.get(i) - targetTime);
      if (distance < minDistance) {
        minDistance = distance;
        nearestIndex = i;
      }
    }

    return nearestIndex;
  }

  /**
   * Generate signal value based on heartbeat timing and physiological patterns
   */
  private double generateSignalValue(List<Long> heartbeats, int nearestBeatIndex, double distanceToNearestBeat, long currentTime, long startTime) {
    // Base signal level
    double baseSignal = 128.0;

    // Calculate RR interval for heart rate pattern
    double rrInterval = 800.0; // Default 800ms (75 BPM)
    if (nearestBeatIndex > 0) {
      rrInterval = heartbeats.get(nearestBeatIndex) - heartbeats.get(nearestBeatIndex - 1);
    } else if (nearestBeatIndex < heartbeats.size() - 1) {
      rrInterval = heartbeats.get(nearestBeatIndex + 1) - heartbeats.get(nearestBeatIndex);
    }

    // Generate heart rate component
    double heartRatePeriod = rrInterval / 1000.0; // Convert to seconds
    double phase = (distanceToNearestBeat / 1000.0) / heartRatePeriod * 2 * Math.PI;

    // Heart rate signal with harmonics (similar to RPPGHandPalmServiceImpl)
    double heartRateComponent = 15.0 * Math.sin(phase) + 5.0 * Math.sin(2 * phase) + 2.0 * Math.sin(3 * phase);

    // Breathing component (slower oscillation)
    double breathingPeriod = 4.0; // 4 second breathing cycle
    double breathingPhase = ((currentTime - startTime) / 1000.0) / breathingPeriod * 2 * Math.PI;
    double breathingComponent = 3.0 * Math.sin(breathingPhase);

    // Small amount of noise for realism
    double noiseComponent = 2.0 * (Math.random() - 0.5);

    return baseSignal + heartRateComponent + breathingComponent + noiseComponent;
  }

  /**
   * Apply signal processing similar to video processing
   * This mimics the signal enhancement used in RPPGHandPalmServiceImpl
   */
  private List<Double> applySignalProcessing(List<Double> rawSignal) {
    if (rawSignal.size() < 10) return rawSignal;

    Log.d(TAG, "Applying signal processing to " + rawSignal.size() + " samples");

    // Step 1: DC removal (remove baseline drift)
    List<Double> dcRemoved = removeDCComponent(rawSignal);

    // Step 2: Apply smoothing filter
    List<Double> smoothed = applyMovingAverageFilter(dcRemoved, 3);

    // Step 3: High-pass filter to remove low frequency noise
    List<Double> filtered = applyHighPassFilter(smoothed);

    // Step 4: Normalize signal
    List<Double> normalized = normalizeSignal(filtered);

    return normalized;
  }

  /**
   * Remove DC component (baseline)
   */
  private List<Double> removeDCComponent(List<Double> signal) {
    double mean = signal.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);

    List<Double> result = new ArrayList<>();
    for (Double value : signal) {
      result.add(value - mean);
    }

    return result;
  }

  /**
   * Apply moving average filter for smoothing
   */
  private List<Double> applyMovingAverageFilter(List<Double> signal, int windowSize) {
    List<Double> smoothed = new ArrayList<>();

    for (int i = 0; i < signal.size(); i++) {
      int start = Math.max(0, i - windowSize/2);
      int end = Math.min(signal.size(), i + windowSize/2 + 1);

      double sum = 0;
      int count = 0;
      for (int j = start; j < end; j++) {
        sum += signal.get(j);
        count++;
      }

      smoothed.add(sum / count);
    }

    return smoothed;
  }

  /**
   * Apply simple high-pass filter
   */
  private List<Double> applyHighPassFilter(List<Double> signal) {
    List<Double> filtered = new ArrayList<>();
    filtered.add(signal.get(0)); // First sample unchanged

    for (int i = 1; i < signal.size(); i++) {
      // Simple high-pass: y[n] = 0.95 * (y[n-1] + x[n] - x[n-1])
      double filtered_val = 0.95 * (filtered.get(i-1) + signal.get(i) - signal.get(i-1));
      filtered.add(filtered_val);
    }

    return filtered;
  }

  /**
   * Normalize signal to reasonable range
   */
  private List<Double> normalizeSignal(List<Double> signal) {
    if (signal.isEmpty()) return signal;

    double min = signal.stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
    double max = signal.stream().mapToDouble(Double::doubleValue).max().orElse(1.0);
    double range = max - min;

    if (range == 0) return signal; // Avoid division by zero

    List<Double> normalized = new ArrayList<>();
    for (Double value : signal) {
      // Normalize to 0-1 range, then scale to desired amplitude
      double normalizedValue = ((value - min) / range - 0.5) * 50.0; // ±25 amplitude
      normalized.add(normalizedValue);
    }

    return normalized;
  }

  /**
   * Generate advanced ECG waveform using interpolated and processed signal data
   * This combines the heartbeat timing with the processed signal for realistic morphology
   */
  private List<Point> generateAdvancedECGWaveform(List<Long> heartbeats, List<Double> processedSignal,
                                                 long startTime, double pixelsPerMs, int baselineY, int padding) {
    List<Point> waveform = new ArrayList<>();

    // Calculate sample spacing
    long totalDuration = heartbeats.get(heartbeats.size() - 1) - startTime;
    double samplesPerMs = processedSignal.size() / (double)totalDuration;

    Log.d(TAG, "Generating advanced ECG waveform with " + processedSignal.size() + " processed samples");

    // Generate waveform points from processed signal
    for (int i = 0; i < processedSignal.size(); i++) {
      long sampleTime = startTime + (long)(i / samplesPerMs);
      double x = padding + (sampleTime - startTime) * pixelsPerMs;

      // Base Y position from processed signal
      double signalY = baselineY + processedSignal.get(i);

      // Enhance with ECG morphology near heartbeats
      double enhancedY = enhanceWithECGMorphology(x, signalY, heartbeats, startTime, pixelsPerMs, baselineY, i);

      waveform.add(new Point(x, enhancedY));
    }

    return waveform;
  }

  /**
   * Enhance signal with realistic ECG morphology near heartbeat locations
   */
  private double enhanceWithECGMorphology(double x, double baseY, List<Long> heartbeats, long startTime,
                                        double pixelsPerMs, int baselineY, int sampleIndex) {
    // Find nearest heartbeat
    double nearestBeatX = -1;
    double minDistance = Double.MAX_VALUE;

    for (Long heartbeat : heartbeats) {
      double beatX = (heartbeat - startTime) * pixelsPerMs;
      double distance = Math.abs(x - beatX);
      if (distance < minDistance) {
        minDistance = distance;
        nearestBeatX = beatX;
      }
    }

    // If we're close to a heartbeat, enhance with ECG morphology
    if (nearestBeatX > 0 && minDistance < 40) { // Within 40 pixels of a heartbeat
      double relativePos = (x - nearestBeatX) / 40.0; // Normalize to ±1
      double ecgEnhancement = generateECGMorphology(relativePos, sampleIndex);

      // Blend the enhancement with the base signal
      double blendFactor = Math.exp(-Math.pow(relativePos * 2, 2)); // Gaussian blend
      return baseY + ecgEnhancement * blendFactor;
    }

    return baseY;
  }

  /**
   * Generate ECG morphology enhancement (PQRST complex)
   */
  private double generateECGMorphology(double relativePos, int sampleIndex) {
    // Add some variation between beats
    double variationFactor = 1.0 + 0.1 * Math.sin(sampleIndex * 0.7) * Math.exp(-sampleIndex * 0.05);

    // Generate PQRST complex based on relative position
    double amplitude = 0;

    if (relativePos >= -0.8 && relativePos <= -0.4) {
      // P wave
      double pPos = (relativePos + 0.6) / 0.2;
      amplitude = 8 * Math.exp(-Math.pow(pPos * 2, 2)) * variationFactor;
    } else if (relativePos >= -0.2 && relativePos <= 0.2) {
      // QRS complex
      if (relativePos >= -0.1 && relativePos <= 0.1) {
        // R wave (main spike)
        double rPos = relativePos / 0.1;
        amplitude = 60 * Math.exp(-Math.pow(rPos * 3, 2)) * variationFactor;
      } else if (relativePos < -0.1) {
        // Q wave
        amplitude = -8 * variationFactor;
      } else {
        // S wave
        amplitude = -15 * variationFactor;
      }
    } else if (relativePos >= 0.3 && relativePos <= 0.8) {
      // T wave
      double tPos = (relativePos - 0.55) / 0.25;
      amplitude = 15 * Math.exp(-Math.pow(tPos * 1.5, 2)) * variationFactor;
    }

    return amplitude;
  }
}
