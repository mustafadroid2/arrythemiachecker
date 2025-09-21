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

    // Create a single graph visualization based on actual rPPG data
    final int WIDTH = 1200;
    final int HEIGHT = 400;
    final int PADDING = 60;

    Mat graph = Mat.zeros(HEIGHT, WIDTH, CvType.CV_8UC3);

    // Draw white background
    graph.setTo(new Scalar(255, 255, 255));

    // Calculate rhythm classification from actual data
    boolean isIrregular = determineIfRhythmIsIrregular(rppgData);
    double avgBpm = rppgData.getAverageBpm();

    // Draw the main graph based on actual rPPG data
    drawRPPGGraph(graph, rppgData, WIDTH, HEIGHT, PADDING, isIrregular, avgBpm);

    // Save the image
    String imagePath = String.format("%s/%s/%s/heartbeats.jpg",
        context.getExternalFilesDir(null) != null ? context.getExternalFilesDir(null).getAbsolutePath() : "",
        AppConstant.DATA_DIR, id);

    File imageFile = new File(imagePath);
    File parentDir = imageFile.getParentFile();
    if (parentDir != null && !parentDir.exists()) {
      boolean created = parentDir.mkdirs();
      if (!created) {
        Log.w(TAG, "Failed to create parent directories");
      }
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
   * Determine if the rhythm is irregular based on rPPG data analysis
   */
  private boolean determineIfRhythmIsIrregular(RPPGData rppgData) {
    List<Long> heartbeats = rppgData.getHeartbeats();
    if (heartbeats.size() < 3) return false;

    // Calculate RR interval variability
    List<Long> rrIntervals = new ArrayList<>();
    for (int i = 1; i < heartbeats.size(); i++) {
      rrIntervals.add(heartbeats.get(i) - heartbeats.get(i - 1));
    }

    double avgRR = rrIntervals.stream().mapToLong(Long::longValue).average().orElse(0);
    double variability = rrIntervals.stream()
        .mapToDouble(rr -> Math.abs(rr - avgRR))
        .average().orElse(0);

    double variabilityPercent = avgRR > 0 ? (variability / avgRR) * 100 : 0;

    // Consider irregular if variability > 15% or if there are extreme outliers
    return variabilityPercent > 15 || hasExtremeRRVariations(rrIntervals, avgRR);
  }

  /**
   * Check for extreme RR interval variations that indicate irregular rhythm
   */
  private boolean hasExtremeRRVariations(List<Long> rrIntervals, double avgRR) {
    for (Long interval : rrIntervals) {
      // If any interval is more than 50% different from average, consider irregular
      if (Math.abs(interval - avgRR) > avgRR * 0.5) {
        return true;
      }
    }
    return false;
  }

  /**
   * Draw the main graph based on actual rPPG data
   */
  private void drawRPPGGraph(Mat graph, RPPGData rppgData, int width, int height, int padding,
                              boolean isIrregular, double avgBpm) {
    // Draw light background
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        graph.put(y, x, new double[]{248, 248, 248});
      }
    }

    // Draw subtle grid lines
    Scalar gridColor = new Scalar(220, 220, 220);
    for (int x = padding; x < width - padding; x += 40) {
      Imgproc.line(graph, new Point(x, 0), new Point(x, height), gridColor, 1);
    }
    for (int y = 20; y < height - 20; y += 20) {
      Imgproc.line(graph, new Point(padding, y), new Point(width - padding, y), gridColor, 1);
    }

    // Draw baseline
    int baselineY = height / 2;
    Scalar baselineColor = new Scalar(150, 150, 150);
    Imgproc.line(graph, new Point(padding, baselineY),
      new Point(width - padding, baselineY), baselineColor, 2);

    // Add label and BPM text
    Scalar textColor = new Scalar(50, 50, 50);
    Imgproc.putText(graph, "Detected Rhythm", new Point(padding + 80, 40),
        Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, textColor, 2);

    String bpmText = String.format(Locale.US, "%.0f BPM", avgBpm);
    Imgproc.putText(graph, bpmText, new Point(width - padding - 120, 40),
        Imgproc.FONT_HERSHEY_SIMPLEX, 0.7, textColor, 2);

    // Extract signal timestamps for time calculation
    List<Long> signalTimestamps = new ArrayList<>();
    for (com.gunadarma.heartratearrhythmiachecker.model.RPPGData.Signal signal : rppgData.getSignals()) {
      signalTimestamps.add(signal.getTimestamp());
    }

    if (signalTimestamps.isEmpty()) return;

    long startTime = signalTimestamps.get(0);
    long endTime = signalTimestamps.get(signalTimestamps.size() - 1);
    long totalDuration = endTime - startTime;
    double pixelsPerMs = (width - 2.0 * padding) / (double)totalDuration;

    generateHeartbeatsOnlyWaveform(rppgData.getHeartbeats(), startTime, endTime, baselineY, padding, width);
    drawStatusIndicator(graph, padding + 30, 30, true); // Green checkmark


    // Generate waveform from actual data
    List<Point> ecgWaveform = generateFrequencyBasedWaveform(
        rppgData.getHeartbeats(),
        signalTimestamps,
        startTime,
        pixelsPerMs,
        baselineY,
        padding,
        width
    );

    // Draw the actual waveform
    drawCleanECGTrace(graph, ecgWaveform);

    // Draw heart symbols at detected heartbeat positions
    Scalar heartColor = new Scalar(200, 50, 50);
    for (Long heartbeat : rppgData.getHeartbeats()) {
      double x = padding + (heartbeat - startTime) * pixelsPerMs;
      if (x >= padding && x <= width - padding) {
        drawHeartSymbol(graph, (int)x, 60, heartColor);
      }
    }
  }

  /**
   * Draw status indicator (green circle with checkmark for normal, red circle with X for irregular)
   */
  private void drawStatusIndicator(Mat graph, int centerX, int centerY, boolean isNormal) {
    int radius = 15;

    if (isNormal) {
      // Green circle with checkmark
      Scalar greenColor = new Scalar(0, 150, 0);
      Imgproc.circle(graph, new Point(centerX, centerY), radius, greenColor, -1);

      // Draw checkmark
      Scalar whiteColor = new Scalar(255, 255, 255);
      Imgproc.line(graph, new Point(centerX - 6, centerY),
                   new Point(centerX - 2, centerY + 4), whiteColor, 3);
      Imgproc.line(graph, new Point(centerX - 2, centerY + 4),
                   new Point(centerX + 6, centerY - 4), whiteColor, 3);
    } else {
      // Red circle with X
      Scalar redColor = new Scalar(0, 0, 200);
      Imgproc.circle(graph, new Point(centerX, centerY), radius, redColor, -1);

      // Draw X
      Scalar whiteColor = new Scalar(255, 255, 255);
      Imgproc.line(graph, new Point(centerX - 6, centerY - 6),
                   new Point(centerX + 6, centerY + 6), whiteColor, 3);
      Imgproc.line(graph, new Point(centerX - 6, centerY + 6),
                   new Point(centerX + 6, centerY - 6), whiteColor, 3);
    }
  }

  /**
   * Draw normal rhythm pattern with evenly spaced heart symbols and regular waveform
   */
  private void drawNormalRhythmPattern(Mat graph, int width, int height, int padding, int baselineY, double avgBpm) {
    // Calculate spacing for regular heartbeats (60 BPM = 1 beat per second)
    int totalWidth = width - 2 * padding;
    double secondsDisplayed = 10.0; // Show 10 seconds
    double beatsInDisplay = (avgBpm / 60.0) * secondsDisplayed;
    int heartSpacing = (int)(totalWidth / beatsInDisplay);

    // Draw heart symbols evenly spaced
    Scalar heartColor = new Scalar(200, 50, 50); // Red hearts
    for (int i = 0; i < beatsInDisplay && (padding + i * heartSpacing) < (width - padding); i++) {
      int x = padding + i * heartSpacing;
      int y = 60;
      drawHeartSymbol(graph, x, y, heartColor);
    }

    // Draw regular ECG waveform
    List<Point> waveform = generateRegularECGWaveform(width, padding, baselineY, avgBpm);
    drawCleanECGTrace(graph, waveform);
  }

  /**
   * Draw irregular rhythm pattern with unevenly spaced heart symbols and irregular waveform
   */
  private void drawIrregularRhythmPattern(Mat graph, int width, int height, int padding, int baselineY) {
    // Create irregular spacing pattern as described: 3 close beats, gap, then uneven spacing
    int[] heartPositions = {
        padding + 80,   // First beat
        padding + 120,  // Second beat (close)
        padding + 160,  // Third beat (close)
        padding + 320,  // Fourth beat (after gap)
        padding + 420,  // Fifth beat (medium spacing)
        padding + 580,  // Sixth beat (larger spacing)
        padding + 680,  // Seventh beat (smaller spacing)
        padding + 820   // Eighth beat (medium spacing)
    };

    // Draw heart symbols at irregular positions
    Scalar heartColor = new Scalar(200, 50, 50); // Red hearts
    for (int pos : heartPositions) {
      if (pos < width - padding) {
        drawHeartSymbol(graph, pos, 60, heartColor);
      }
    }

    // Draw irregular ECG waveform matching the heart positions
    List<Point> waveform = generateIrregularECGWaveform(width, padding, baselineY, heartPositions);
    drawCleanECGTrace(graph, waveform);
  }

  /**
   * Draw a simple heart symbol
   */
  private void drawHeartSymbol(Mat graph, int centerX, int centerY, Scalar color) {
    // Draw heart shape using circles and triangle
    int size = 8;

    // Two circles for top of heart
    Imgproc.circle(graph, new Point(centerX - size/2, centerY - size/2), size/2, color, -1);
    Imgproc.circle(graph, new Point(centerX + size/2, centerY - size/2), size/2, color, -1);

    // Triangle for bottom of heart
    Point[] trianglePoints = {
        new Point(centerX - size, centerY),
        new Point(centerX + size, centerY),
        new Point(centerX, centerY + size)
    };

    // Fill the triangle area manually using lines
    for (int i = 0; i < size; i++) {
      int y = centerY + i;
      int leftX = centerX - size + i;
      int rightX = centerX + size - i;
      Imgproc.line(graph, new Point(leftX, y), new Point(rightX, y), color, 1);
    }
  }

  /**
   * Generate regular ECG waveform with consistent pattern
   */
  private List<Point> generateRegularECGWaveform(int width, int padding, int baselineY, double avgBpm) {
    List<Point> waveform = new ArrayList<>();

    int totalWidth = width - 2 * padding;
    double secondsDisplayed = 10.0;
    double samplesPerSecond = totalWidth / secondsDisplayed;

    for (int i = 0; i < totalWidth; i++) {
      double time = i / samplesPerSecond; // Time in seconds
      double heartPhase = (time * avgBpm / 60.0) % 1.0; // Phase within heart cycle

      // Generate QRS complex pattern
      double amplitude = 0;
      if (heartPhase < 0.1) {
        // P wave
        amplitude = 10 * Math.sin(heartPhase * Math.PI / 0.1);
      } else if (heartPhase >= 0.15 && heartPhase < 0.25) {
        // QRS complex
        double qrsPhase = (heartPhase - 0.15) / 0.1;
        amplitude = 60 * Math.sin(qrsPhase * Math.PI);
      } else if (heartPhase >= 0.35 && heartPhase < 0.55) {
        // T wave
        double tPhase = (heartPhase - 0.35) / 0.2;
        amplitude = 20 * Math.sin(tPhase * Math.PI);
      }

      double x = padding + i;
      double y = baselineY - amplitude;
      waveform.add(new Point(x, y));
    }

    return waveform;
  }

  /**
   * Generate irregular ECG waveform matching heart positions
   */
  private List<Point> generateIrregularECGWaveform(int width, int padding, int baselineY, int[] heartPositions) {
    List<Point> waveform = new ArrayList<>();

    int totalWidth = width - 2 * padding;

    for (int i = 0; i < totalWidth; i++) {
      double x = padding + i;
      double amplitude = 0;

      // Add QRS complexes at heart positions
      for (int heartPos : heartPositions) {
        double distance = Math.abs(x - heartPos);
        if (distance < 15) {
          // Create QRS complex
          double normalizedDist = distance / 15.0;
          amplitude += 60 * Math.exp(-normalizedDist * normalizedDist * 9) *
                      Math.sin((1 - normalizedDist) * Math.PI);
        }
      }

      // Add some baseline noise
      amplitude += (Math.random() - 0.5) * 5;

      double y = baselineY - amplitude;
      waveform.add(new Point(x, y));
    }

    return waveform;
  }

  /**
   * Generate waveform showing only heartbeats as Gaussian-like peaks
   */
  private List<Point> generateHeartbeatsOnlyWaveform(List<Long> heartbeats, long startTime, long endTime,
                                                     int baselineY, int padding, int width) {
    List<Point> waveform = new ArrayList<>();

    int totalWidth = width - 2 * padding;
    long totalDuration = endTime - startTime;

    for (int i = 0; i < totalWidth; i++) {
      long currentTime = startTime + (long)((i * totalDuration) / (double)(totalWidth - 1));
      double amplitude = 0.0;

      // Create Gaussian peaks only at heartbeat detection points
      for (Long heartbeat : heartbeats) {
        long distance = Math.abs(currentTime - heartbeat);
        if (distance < 1000) { // Influence area around each heartbeat
          double normalizedDistance = distance / 1000.0;
          // Gaussian envelope for smooth, rounded peaks
          double gaussianPeak = Math.exp(-Math.pow(normalizedDistance * 3, 2));
          amplitude += gaussianPeak * 80; // Scale for appropriate display height
        }
      }

      double x = padding + i;
      double y = baselineY - amplitude; // Subtract to draw peaks above baseline
      waveform.add(new Point(x, y));
    }

    return waveform;
  }

  /**
   * Draw clean ECG trace
   */
  private void drawCleanECGTrace(Mat graph, List<Point> waveform) {
    if (waveform.size() < 2) return;

    Scalar ecgColor = new Scalar(0, 100, 0); // Dark green

    for (int i = 1; i < waveform.size(); i++) {
      Point p1 = waveform.get(i - 1);
      Point p2 = waveform.get(i);
      Imgproc.line(graph, p1, p2, ecgColor, 2, Imgproc.LINE_AA);
    }
  }


  /**
   * Generate frequency-based waveform from actual rPPG data
   */
  private List<Point> generateFrequencyBasedWaveform(List<Long> heartbeats, List<Long> signalTimestamps,
                                                    long startTime, double pixelsPerMs, int baselineY,
                                                    int padding, int width) {
    List<Point> waveform = new ArrayList<>();

    int totalWidth = width - 2 * padding;
    long endTime = signalTimestamps.get(signalTimestamps.size() - 1);
    long totalDuration = endTime - startTime;

    for (int i = 0; i < totalWidth; i++) {
      long currentTime = startTime + (long)((i * totalDuration) / (double)(totalWidth - 1));
      double amplitude = calculateFrequencySignalAtTime(currentTime, heartbeats, signalTimestamps);

      double x = padding + i;
      double y = baselineY - amplitude / 30.0; // Scale down amplitude for display
      waveform.add(new Point(x, y));
    }

    return waveform;
  }

  /**
   * Calculate frequency signal amplitude at a given time
   */
  private double calculateFrequencySignalAtTime(long currentTime, List<Long> heartbeats, List<Long> signalTimestamps) {
    double amplitude = 0.0;

    // Create peaks at heartbeat detection points
    for (Long heartbeat : heartbeats) {
      long distance = Math.abs(currentTime - heartbeat);
      if (distance < 800) {
        double normalizedDistance = distance / 800.0;
        double gaussianEnvelope = Math.exp(-Math.pow(normalizedDistance * 3, 2));
        amplitude += gaussianEnvelope * 1800;
      }
    }

    // Add background signal
    for (Long signalTime : signalTimestamps) {
      long signalDistance = Math.abs(currentTime - signalTime);
      if (signalDistance < 300) {
        double normalizedSignalDistance = signalDistance / 300.0;
        double smoothEnvelope = Math.exp(-Math.pow(normalizedSignalDistance * 2.5, 2));
        double timeFromSignal = (currentTime - signalTime) / 1000.0;
        amplitude += smoothEnvelope * 120 * Math.sin(2 * Math.PI * 25.0 * timeFromSignal);
      }
    }

    return Math.max(0, amplitude);
  }
}
