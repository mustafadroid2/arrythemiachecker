package com.gunadarma.heartratearrhythmiachecker.util;

import android.content.Context;
import android.app.ProgressDialog;
import android.util.Log;
import android.widget.Toast;

import com.gunadarma.heartratearrhythmiachecker.model.RecordEntry;
import com.gunadarma.heartratearrhythmiachecker.service.DataRecordServiceImpl;
import com.gunadarma.heartratearrhythmiachecker.service.MainMediaProcessingServiceImpl;

/**
 * Utility class for shared video processing functionality between fragments
 */
public class VideoProcessingUtil {
    private static final String TAG = "VideoProcessingUtil";

    /**
     * Interface for video processing callbacks
     */
    public interface ProcessingCallback {
        void onProgressUpdate(int progress, String message);
        void onSuccess();
        void onError(Exception error);
        void runOnUiThread(Runnable runnable);
        Context getContext();
    }

    /**
     * Process video with standardized progress updates and error handling
     * @param recordEntry The record entry to process
     * @param callback Callback interface for UI updates
     */
    public static void processVideo(RecordEntry recordEntry, ProcessingCallback callback) {
        if (recordEntry == null) {
            callback.onError(new IllegalArgumentException("Record entry cannot be null"));
            return;
        }

        // Show processing dialog
        callback.runOnUiThread(() -> {
            ProgressDialog progressDialog = new ProgressDialog(callback.getContext());
            progressDialog.setTitle("Processing Video");
            progressDialog.setMessage("Analyzing heart rate from video...");
            progressDialog.setProgressStyle(ProgressDialog.STYLE_HORIZONTAL);
            progressDialog.setMax(100);
            progressDialog.setCancelable(false);
            progressDialog.show();

            // Execute processing in background thread
            new Thread(() -> {
                try {
                    DataRecordServiceImpl dataRecordService = new DataRecordServiceImpl(callback.getContext());

                    // Step 1: Update record status to PROCESSING
                    recordEntry.setStatus(RecordEntry.Status.UNCHECKED);
                    dataRecordService.saveData(recordEntry);

                    callback.runOnUiThread(() -> {
                        progressDialog.setProgress(10);
                        progressDialog.setMessage("Record saved. Starting video analysis...");
                    });

                    // Step 2: Extract heart rate signals
                    callback.runOnUiThread(() -> {
                        progressDialog.setProgress(20);
                        progressDialog.setMessage("Extracting heart rate signals...");
                    });

                    // Step 3: Process the video using MainMediaProcessingServiceImpl
                    MainMediaProcessingServiceImpl mediaProcessingService = new MainMediaProcessingServiceImpl(callback.getContext());
                    mediaProcessingService.createHeartBeatsVideo(recordEntry);

                    callback.runOnUiThread(() -> {
                        progressDialog.setProgress(80);
                        progressDialog.setMessage("Generating heart rate timeline...");
                    });

                    // Step 4: Update record status after processing
                    if (recordEntry != null) {
                        dataRecordService.saveData(recordEntry);
                    }

                    // Step 5: Final progress update
                    callback.runOnUiThread(() -> {
                        progressDialog.setProgress(100);
                        progressDialog.setMessage("Processing complete!");
                    });

                    // Small delay to show completion
                    Thread.sleep(500);

                    // Notify success
                    callback.runOnUiThread(() -> {
                        progressDialog.dismiss();
                        Toast.makeText(callback.getContext(),
                            "Video processing completed successfully!",
                            Toast.LENGTH_SHORT).show();
                        callback.onSuccess();
                    });

                } catch (Exception processingError) {
                    Log.e(TAG, "Video processing failed", processingError);

                    // Update record status to error
                    try {
                        recordEntry.setStatus(RecordEntry.Status.UNCHECKED);
                        DataRecordServiceImpl dataRecordService = new DataRecordServiceImpl(callback.getContext());
                        dataRecordService.saveData(recordEntry);
                    } catch (Exception dbError) {
                        Log.e(TAG, "Failed to update record status after processing error", dbError);
                    }

                    callback.runOnUiThread(() -> {
                        progressDialog.dismiss();
                        callback.onError(processingError);
                    });
                }
            }).start();
        });
    }

    /**
     * Show standardized error dialog with retry options
     * @param context Android context
     * @param error The error that occurred
     * @param onRetry Callback for retry action
     * @param onViewDetails Callback for view details action
     */
    public static void showProcessingErrorDialog(Context context, Exception error,
                                                Runnable onRetry, Runnable onViewDetails) {
        androidx.appcompat.app.AlertDialog.Builder builder = new androidx.appcompat.app.AlertDialog.Builder(context)
            .setTitle("Processing Error")
            .setMessage("Failed to process video: " + error.getMessage() + "\n\nWould you like to try again?")
            .setPositiveButton("Retry", (dialog, which) -> {
                if (onRetry != null) {
                    onRetry.run();
                }
            })
            .setNegativeButton("Cancel", (dialog, which) -> {
                dialog.dismiss();
            });

        if (onViewDetails != null) {
            builder.setNeutralButton("View Details", (dialog, which) -> {
                onViewDetails.run();
            });
        }

        builder.show();
    }

    /**
     * Show standardized database error message
     * @param context Android context
     * @param error The database error that occurred
     */
    public static void showDatabaseErrorToast(Context context, Exception error) {
        Toast.makeText(context,
            "Failed to save record: " + error.getMessage(),
            Toast.LENGTH_LONG).show();
    }
}
