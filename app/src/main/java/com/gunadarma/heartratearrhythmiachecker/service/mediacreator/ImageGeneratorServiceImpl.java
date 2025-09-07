package com.gunadarma.heartratearrhythmiachecker.service.mediacreator;

import android.content.Context;

import com.gunadarma.heartratearrhythmiachecker.constant.AppConstant;
import com.gunadarma.heartratearrhythmiachecker.model.HeartRateData;

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

  public ImageGeneratorServiceImpl(Context context) {
    this.context = context;
  }

  public void createHeartBeatsImage(List<Long> heartbeats, Long id) {
    if (heartbeats.isEmpty()) return;

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

    // Generate realistic ECG-like waveform from heartbeat intervals
    List<Point> ecgWaveform = generateECGWaveform(heartbeats, startTime, pixelsPerMs, baselineY, PADDING);

    // Draw the ECG waveform with medical device styling
    drawECGWaveform(graph, ecgWaveform);

    // Add ECG-style annotations and measurements
    addECGAnnotations(graph, heartbeats, WIDTH, HEIGHT, PADDING, baselineY);

    // Save the ECG-style image
    String imagePath = String.format("%s/%s/%s/%s/heartbeats.jpg",
        context.getExternalFilesDir(null).getAbsolutePath(),
        AppConstant.DATA_DIR, id, "");

    File imageFile = new File(imagePath);
    File parentDir = imageFile.getParentFile();
    if (parentDir != null && !parentDir.exists()) {
      parentDir.mkdirs();
    }

    org.opencv.imgcodecs.Imgcodecs.imwrite(imagePath, graph);
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
}
