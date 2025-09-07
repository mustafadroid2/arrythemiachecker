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

    // Create an image showing the heart rate timeline with padding
    final int WIDTH = 1000;  // Increased width to accommodate padding
    final int HEIGHT = 300;   // Increased height for better visualization
    final int PADDING = 50;   // Padding on all sides
    final int GRID_SIZE = 25; // Grid size in pixels

    Mat graph = Mat.zeros(HEIGHT, WIDTH, CvType.CV_8UC3);

    // Draw white background
    graph.setTo(new Scalar(255, 255, 255));

    // Calculate time scale with padding (for heartbeat data only)
    long duration = heartbeats.get(0) - heartbeats.get(heartbeats.size());
    double pixelsPerMs = (WIDTH - 2.0 * PADDING) / duration;

    // Draw grid across the entire image
    Scalar gridColor = new Scalar(255, 220, 220); // Light pink grid
    // Vertical grid lines
    for (int x = 0; x < WIDTH; x += GRID_SIZE) {
      Imgproc.line(graph, new Point(x, 0), new Point(x, HEIGHT),
        gridColor, 1, Imgproc.LINE_AA);
    }
    // Horizontal grid lines
    for (int y = 0; y < HEIGHT; y += GRID_SIZE) {
      Imgproc.line(graph, new Point(0, y), new Point(WIDTH, y),
        gridColor, 1, Imgproc.LINE_AA);
    }

    // Draw darker major grid lines across the entire image
    Scalar majorGridColor = new Scalar(255, 200, 200);
    for (int x = 0; x < WIDTH; x += GRID_SIZE * 5) {
      Imgproc.line(graph, new Point(x, 0), new Point(x, HEIGHT),
        majorGridColor, 2, Imgproc.LINE_AA);
    }
    for (int y = 0; y < HEIGHT; y += GRID_SIZE * 5) {
      Imgproc.line(graph, new Point(0, y), new Point(WIDTH, y),
        majorGridColor, 2, Imgproc.LINE_AA);
    }

    // Draw baseline (within padding area)
    int baselineY = HEIGHT / 2;
    Scalar baselineColor = new Scalar(200, 200, 200);
    Imgproc.line(graph, new Point(PADDING, baselineY),
      new Point(WIDTH - PADDING, baselineY),
      baselineColor, 1, Imgproc.LINE_AA);

    // Draw heartbeats with ECG-like waveform
    Scalar beatColor = new Scalar(0, 0, 255); // Red color for heart beats
    int prevX = PADDING;
    int prevY = baselineY;

    for (int i = 0; i < heartbeats.size(); i++) {
      long t = heartbeats.get(i) - heartbeats.get(0);
      int x = (int)(t * pixelsPerMs) + PADDING;

      // Create ECG-like PQRST wave pattern
      List<Point> wavePoints = new ArrayList<>();
      // P wave (small bump before the peak)
      wavePoints.add(new Point(x - 15, baselineY - 10));
      // Q wave (small dip)
      wavePoints.add(new Point(x - 10, baselineY + 5));
      // R wave (main spike)
      wavePoints.add(new Point(x, baselineY - 100));
      // S wave (dip after spike)
      wavePoints.add(new Point(x + 10, baselineY + 20));
      // T wave (small bump after)
      wavePoints.add(new Point(x + 20, baselineY - 15));
      // Back to baseline
      wavePoints.add(new Point(x + 30, baselineY));

      // Draw the PQRST complex
      for (int j = 1; j < wavePoints.size(); j++) {
        Point p1 = wavePoints.get(j-1);
        Point p2 = wavePoints.get(j);
        Imgproc.line(graph, p1, p2, beatColor, 2, Imgproc.LINE_AA);
      }

      // Connect to previous beat with baseline
      if (i > 0) {
        Point lastT = new Point(prevX + 30, baselineY);
        Point nextP = new Point(x - 15, baselineY);
        Imgproc.line(graph, lastT, nextP, beatColor, 2, Imgproc.LINE_AA);
      }

      prevX = x;
    }

    // Draw border
    Scalar borderColor = new Scalar(255, 200, 200);
    Imgproc.rectangle(graph, new Point(PADDING, PADDING),
      new Point(WIDTH - PADDING, HEIGHT - PADDING),
      borderColor, 2, Imgproc.LINE_AA);

    // Save image
    File imageFile = new File(
      context.getExternalFilesDir(null),
      String.format("%s/%s/heartbeats.jpg", AppConstant.DATA_DIR, id)
    );
    imageFile.getParentFile().mkdirs();
    org.opencv.imgcodecs.Imgcodecs.imwrite(imageFile.getAbsolutePath(), graph);
  }
}
