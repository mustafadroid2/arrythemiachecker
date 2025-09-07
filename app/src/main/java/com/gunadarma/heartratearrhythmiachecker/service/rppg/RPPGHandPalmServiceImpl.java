package com.gunadarma.heartratearrhythmiachecker.service.rppg;

import android.content.Context;
import android.util.Log;

import com.gunadarma.heartratearrhythmiachecker.constant.AppConstant;
import com.gunadarma.heartratearrhythmiachecker.model.RPPGData;
import com.gunadarma.heartratearrhythmiachecker.service.MediaPipeHandTracker;

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
import org.opencv.videoio.VideoWriter;
import org.opencv.videoio.Videoio;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class RPPGHandPalmServiceImpl implements RPPGService {
  private static final String TAG = "RPPGHandPalmService";
  private final Context context;
  private MediaPipeHandTracker mpHandTracker;

  // rPPG processing constants for palm detection
  private static final double MIN_HR_BPM = 50.0;
  private static final double MAX_HR_BPM = 180.0;
  private static final double BANDPASS_LOW_FREQ = 0.83; // 50 BPM in Hz
  private static final double BANDPASS_HIGH_FREQ = 3.0; // 180 BPM in Hz
  private static final int WINDOW_SIZE_SECONDS = 6; // Processing window size
  private static final double PEAK_THRESHOLD = 0.25; // Lower threshold for palm signals
  private static final double SKIN_CONFIDENCE_THRESHOLD = 0.15; // Minimum skin pixel ratio

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

      Log.d(TAG, String.format("Video properties: %.2f fps, %d frames, %d seconds",
                               fps, totalFrames, videoDurationSeconds));

      // Extract RGB signals from palm region
      List<RPPGData.Signal> signals = extractPalmSignals(cap, fps);

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
          .durationSeconds(videoDurationSeconds)
          .signals(signals)
          .build();

      Log.i(TAG, String.format("Palm rPPG extraction completed: %.1f BPM average, %d heartbeats detected",
                               result.averageBpm, result.heartbeats.size()));

      return rppgData;

    } finally {
      cap.release();
      if (mpHandTracker != null) {
        mpHandTracker.release();
      }
    }
  }

  private List<RPPGData.Signal> extractPalmSignals(VideoCapture cap, double fps) {
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

    String outputPath = "result.mp4";
    int fourcc = VideoWriter.fourcc('m', 'p', '4', 'v');
    Size frameSize = new Size(frame.cols(), frame.rows());
    VideoWriter writer = new VideoWriter(outputPath, fourcc, fps, frameSize);

    // ENSURE ONLY USE MEDIAPIPE FOR HAND DETECTION FOR CONSISTENCY
    while (cap.read(frame)) {
      if (frame.empty()) continue;

      try {
        // Try MediaPipe hand detection first
        MediaPipeHandTracker.HandDetectionResult handResult = null;
        if (mpHandTracker != null) {
          handResult = mpHandTracker.detectHand(frame);
        }

        if (handResult != null && handResult.palmROI != null && isValidPalmROI(handResult.palmROI, frame)) {
          palmFrames++;
          Mat palmRegion = new Mat(frame, handResult.palmROI);
          Mat rgb = new Mat();
          Imgproc.cvtColor(palmRegion, rgb, Imgproc.COLOR_BGR2RGB);

          Mat skinMask = createPalmSkinMask(rgb);
          Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(3, 3));
          Imgproc.morphologyEx(skinMask, skinMask, Imgproc.MORPH_OPEN, kernel);
          Imgproc.morphologyEx(skinMask, skinMask, Imgproc.MORPH_CLOSE, kernel);

          Scalar means = Core.mean(rgb, skinMask);
          double skinRatio = (double) Core.countNonZero(skinMask) / skinMask.total();
          if (skinRatio > SKIN_CONFIDENCE_THRESHOLD) {
            long timestamp = startTime + (long) (frameCount * (1000.0 / fps));
            signals.add(new RPPGData.Signal(means.val[0], means.val[1], means.val[2], timestamp));
            validFrames++;
          }

          palmRegion.release();
          rgb.release();
          skinMask.release();
          kernel.release();
        }
      } catch (Exception e) {
        Log.w(TAG, "Error processing frame " + frameCount, e);
      }

      frameCount++;
      if (frameCount % 100 == 0) {
        Log.d(TAG, String.format("Processed %d frames, %d valid, %d with palm detection",
                                 frameCount, validFrames, palmFrames));
      }
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

    // Convert to YCrCb color space for better skin detection
    Mat ycrcb = new Mat();
    Imgproc.cvtColor(rgb, ycrcb, Imgproc.COLOR_RGB2YCrCb);

    // Define skin color range in YCrCb (more robust for different lighting)
    Mat mask1 = new Mat();
    Core.inRange(ycrcb,
      new Scalar(0, 133, 77),   // Lower bound
      new Scalar(255, 173, 127), // Upper bound
      mask1);

    // Additional HSV-based skin detection for robustness
    Mat hsv = new Mat();
    Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV);

    Mat mask2 = new Mat();
    Core.inRange(hsv,
      new Scalar(0, 20, 70),   // Lower bound in HSV
      new Scalar(20, 255, 255), // Upper bound in HSV
      mask2);

    // Combine both masks
    Core.bitwise_and(mask1, mask2, skinMask);

    // Cleanup
    ycrcb.release();
    hsv.release();
    mask1.release();
    mask2.release();

    return skinMask;
  }

  private RPPGProcessingResult processRPPGSignals(List<RPPGData.Signal> signals, double fps) {
    if (signals.size() < fps * 8) { // Need at least 8 seconds of data for palm
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

    // Apply CHROM algorithm for better palm rPPG signal extraction
    Mat chromSignal = applyCHROMAlgorithm(redProcessed, greenProcessed, blueProcessed);

    // Apply bandpass filter
    Mat filteredSignal = applyBandpassFilter(chromSignal, fps);

    // Extract heartbeats using peak detection with palm-specific parameters
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
    chromSignal.release();
    filteredSignal.release();

    return new RPPGProcessingResult(heartbeats, minBpm, maxBpm, averageBpm);
  }

  private Mat applyCHROMAlgorithm(Mat red, Mat green, Mat blue) {
    // CHROM algorithm is more robust for palm-based rPPG
    // X_s = 3*R - 2*G
    // Y_s = 1.5*R + G - 1.5*B

    Mat Xs = new Mat();
    Mat Ys = new Mat();
    Mat temp1 = new Mat();
    Mat temp2 = new Mat();
    Mat temp3 = new Mat();

    // Calculate X_s = 3*R - 2*G
    Core.multiply(red, new Scalar(3.0), temp1);
    Core.multiply(green, new Scalar(2.0), temp2);
    Core.subtract(temp1, temp2, Xs);

    // Calculate Y_s = 1.5*R + G - 1.5*B
    Core.multiply(red, new Scalar(1.5), temp1);
    Core.multiply(blue, new Scalar(1.5), temp2);
    Core.add(temp1, green, temp3);
    Core.subtract(temp3, temp2, Ys);

    // Calculate alpha = std(X_s) / std(Y_s)
    MatOfDouble meanXs = new MatOfDouble();
    MatOfDouble stdXs = new MatOfDouble();
    Core.meanStdDev(Xs, meanXs, stdXs);

    MatOfDouble meanYs = new MatOfDouble();
    MatOfDouble stdYs = new MatOfDouble();
    Core.meanStdDev(Ys, meanYs, stdYs);

    double alpha = stdXs.get(0, 0)[0] / Math.max(stdYs.get(0, 0)[0], 1e-6);

    // S = X_s - alpha * Y_s
    Mat alphaMat = new Mat();
    Core.multiply(Ys, new Scalar(alpha), alphaMat);
    Mat chromSignal = new Mat();
    Core.subtract(Xs, alphaMat, chromSignal);

    // Cleanup
    Xs.release();
    Ys.release();
    temp1.release();
    temp2.release();
    temp3.release();
    alphaMat.release();

    return chromSignal;
  }

  private Mat preprocessSignal(Mat signal) {
    // Detrend signal by removing linear trend
    Mat detrended = detrendSignal(signal);

    // Apply smoothing filter to reduce noise
    Mat smoothed = new Mat();
    Imgproc.GaussianBlur(detrended, smoothed, new Size(5, 1), 1.0);

    // Normalize to [0, 1]
    Mat normalized = new Mat();
    Core.normalize(smoothed, normalized, 0, 1, Core.NORM_MINMAX);

    detrended.release();
    smoothed.release();
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
    int minPeakDistance = (int)(0.35 * fps); // Minimum 0.35 seconds between peaks for palm

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
