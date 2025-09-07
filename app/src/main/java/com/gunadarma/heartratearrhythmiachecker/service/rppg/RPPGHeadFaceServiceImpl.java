package com.gunadarma.heartratearrhythmiachecker.service.rppg;

import android.content.Context;
import android.util.Log;

import com.gunadarma.heartratearrhythmiachecker.model.RPPGData;
import com.gunadarma.heartratearrhythmiachecker.service.MediaPipeFaceTracker;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class RPPGHeadFaceServiceImpl implements RPPGService {
  private static final String TAG = "RPPGHeadFaceService";
  private final Context context;
  private MediaPipeFaceTracker faceTracker;
  private CascadeClassifier faceDetector;

  // rPPG processing constants
  private static final double MIN_HR_BPM = 50.0;
  private static final double MAX_HR_BPM = 180.0;
  private static final double BANDPASS_LOW_FREQ = 0.83; // 50 BPM in Hz
  private static final double BANDPASS_HIGH_FREQ = 3.0; // 180 BPM in Hz
  private static final int WINDOW_SIZE_SECONDS = 6; // Processing window size
  private static final double PEAK_THRESHOLD = 0.3;

  public RPPGHeadFaceServiceImpl(Context context) {
    this.context = context;
    this.faceTracker = new MediaPipeFaceTracker(context);
    this.faceDetector = initializeFaceDetector();
  }

  @Override
  public RPPGData getRPPGSignals(String videoPath) {
    Log.i(TAG, "Starting rPPG signal extraction from: " + videoPath);

    VideoCapture cap = new VideoCapture(videoPath);
    if (!cap.isOpened()) {
      Log.e(TAG, "Failed to open video file: " + videoPath);
      return RPPGData.empty();
    }

    try {
      // Get video properties
      double fps = cap.get(Videoio.CAP_PROP_FPS);
      int totalFrames = (int) cap.get(Videoio.CAP_PROP_FRAME_COUNT);
      int videoDurationSeconds = (int) (totalFrames / fps);

      Log.d(TAG, String.format("Video properties: %.2f fps, %d frames, %d seconds",
                               fps, totalFrames, videoDurationSeconds));

      // Extract RGB signals from face/forehead region
      List<RPPGData.Signal> signals = extractFaceSignals(cap, fps);

      if (signals.isEmpty()) {
        Log.w(TAG, "No valid face signals extracted");
        return RPPGData.empty();
      }

      // Process signals to extract heart rate data
      RPPGProcessingResult result = processRPPGSignals(signals, fps);

      // Create and return RPPGData

      Log.i(TAG, String.format("rPPG extraction completed: %.1f BPM average, %d heartbeats detected",
                               result.averageBpm, result.heartbeats.size()));

      return RPPGData.builder()
        .heartbeats(result.heartbeats)
        .minBpm(result.minBpm)
        .maxBpm(result.maxBpm)
        .averageBpm(result.averageBpm)
        .durationSeconds(videoDurationSeconds)
        .signals(signals)
        .build();
    } finally {
      cap.release();
      if (faceTracker != null) {
        faceTracker.release();
      }
    }
  }

  private List<RPPGData.Signal> extractFaceSignals(VideoCapture cap, double fps) {
    List<RPPGData.Signal> signals = new ArrayList<>();
    Mat frame = new Mat();
    long startTime = System.currentTimeMillis();
    int frameCount = 0;
    int validFrames = 0;

    while (cap.read(frame)) {
      if (frame.empty()) continue;

      try {
        // Try MediaPipe face detection first
        MediaPipeFaceTracker.FaceDetectionResult faceResult = null;
        if (faceTracker != null) {
          faceResult = faceTracker.detectFace(frame);
        }

        Rect foreheadROI = null;

        if (faceResult != null) {
          foreheadROI = faceResult.foreheadROI;
        } else {
          // Fallback to OpenCV face detection
          foreheadROI = detectFaceWithOpenCV(frame);
        }

        if (foreheadROI != null) {
          // Extract color signals from forehead region
          Mat foreheadRegion = new Mat(frame, foreheadROI);
          Mat rgb = new Mat();
          Imgproc.cvtColor(foreheadRegion, rgb, Imgproc.COLOR_BGR2RGB);

          // Apply skin color filtering
          Mat skinMask = createSkinMask(rgb);

          // Calculate mean color values from skin pixels
          Scalar means = Core.mean(rgb, skinMask);

          // Only store if we have sufficient skin pixels
          if (Core.countNonZero(skinMask) > skinMask.total() * 0.1) {
            long timestamp = startTime + (long)(frameCount * (1000.0 / fps));
            signals.add(new RPPGData.Signal(means.val[0], means.val[1], means.val[2], timestamp));
            validFrames++;
          }

          // Cleanup
          foreheadRegion.release();
          rgb.release();
          skinMask.release();
        }

        if (faceResult != null && faceResult.roi != null) {
          faceResult.roi.release();
        }

      } catch (Exception e) {
        Log.w(TAG, "Error processing frame " + frameCount, e);
      }

      frameCount++;
      if (frameCount % 100 == 0) {
        Log.d(TAG, String.format("Processed %d frames, %d valid", frameCount, validFrames));
      }
    }

    Log.i(TAG, String.format("Extracted signals from %d/%d frames", validFrames, frameCount));
    return signals;
  }

  private Rect detectFaceWithOpenCV(Mat frame) {
    if (faceDetector == null || faceDetector.empty()) {
      return null;
    }

    try {
      MatOfRect faces = new MatOfRect();
      Mat grayFrame = new Mat();
      Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);

      faceDetector.detectMultiScale(grayFrame, faces, 1.1, 3, 0, new Size(30, 30));

      Rect[] faceArray = faces.toArray();
      if (faceArray.length > 0) {
        // Use the largest face
        Rect face = faceArray[0];

        // Calculate forehead ROI (upper 30% of face)
        int foreheadHeight = (int)(face.height * 0.3);
        int foreheadWidth = (int)(face.width * 0.6);
        int foreheadX = face.x + (face.width - foreheadWidth) / 2;
        int foreheadY = face.y;

        grayFrame.release();
        return new Rect(foreheadX, foreheadY, foreheadWidth, foreheadHeight);
      }

      grayFrame.release();
    } catch (Exception e) {
      Log.w(TAG, "OpenCV face detection failed", e);
    }

    return null;
  }

  private Mat createSkinMask(Mat rgb) {
    Mat skinMask = new Mat();
    // Skin color range in RGB
    Core.inRange(rgb,
      new Scalar(95, 40, 20),   // Minimum skin color
      new Scalar(255, 220, 180), // Maximum skin color
      skinMask);
    return skinMask;
  }

  private RPPGProcessingResult processRPPGSignals(List<RPPGData.Signal> signals, double fps) {
    if (signals.size() < fps * 10) { // Need at least 10 seconds of data
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

    // Combine channels (green channel is most sensitive to blood volume changes)
    Mat combinedSignal = new Mat();
    Core.addWeighted(greenProcessed, 0.7, redProcessed, 0.2, 0, combinedSignal);
    Core.addWeighted(combinedSignal, 1.0, blueProcessed, 0.1, 0, combinedSignal);

    // Apply bandpass filter
    Mat filteredSignal = applyBandpassFilter(combinedSignal, fps);

    // Extract heartbeats using peak detection
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
    combinedSignal.release();
    filteredSignal.release();

    return new RPPGProcessingResult(heartbeats, minBpm, maxBpm, averageBpm);
  }

  private Mat preprocessSignal(Mat signal) {
    // Detrend signal by removing linear trend
    Mat detrended = detrendSignal(signal);

    // Normalize to [0, 1]
    Mat normalized = new Mat();
    Core.normalize(detrended, normalized, 0, 1, Core.NORM_MINMAX);

    detrended.release();
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
    int minPeakDistance = (int)(0.4 * fps); // Minimum 0.4 seconds between peaks

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

  private CascadeClassifier initializeFaceDetector() {
    try {
      // Create temporary file for cascade classifier
      File cascadeDir = context.getDir("cascade", Context.MODE_PRIVATE);
      cascadeDir.mkdirs();
      File cascadeFile = new File(cascadeDir, "haarcascade_frontalface_alt.xml");

      // Copy cascade file from assets
      InputStream is = context.getAssets().open("haarcascades/haarcascade_frontalface_alt.xml");
      FileOutputStream os = new FileOutputStream(cascadeFile);
      byte[] buffer = new byte[4096];
      int bytesRead;
      while ((bytesRead = is.read(buffer)) != -1) {
        os.write(buffer, 0, bytesRead);
      }
      is.close();
      os.close();

      // Create classifier
      CascadeClassifier detector = new CascadeClassifier(cascadeFile.getAbsolutePath());

      // Cleanup
      cascadeFile.delete();
      cascadeDir.delete();

      if (detector.empty()) {
        Log.e(TAG, "Failed to load cascade classifier");
        return null;
      }

      return detector;
    } catch (Exception e) {
      Log.e(TAG, "Error initializing face detector", e);
      return null;
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
}
