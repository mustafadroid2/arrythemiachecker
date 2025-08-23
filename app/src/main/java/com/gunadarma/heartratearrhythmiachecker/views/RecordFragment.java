package com.gunadarma.heartratearrhythmiachecker.views;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.hardware.Camera;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.Surface;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;

import com.gunadarma.heartratearrhythmiachecker.R;
import com.gunadarma.heartratearrhythmiachecker.databinding.FragmentRecordBinding;
import com.gunadarma.heartratearrhythmiachecker.model.RecordEntry;
import com.gunadarma.heartratearrhythmiachecker.service.DataRecordServiceImpl;
import com.gunadarma.heartratearrhythmiachecker.service.MediaProcessingServiceImpl;
import com.gunadarma.heartratearrhythmiachecker.util.AppConstant;

import java.io.File;
import java.util.List;

public class RecordFragment extends Fragment {
    private DataRecordServiceImpl dataRecordService;
    private FragmentRecordBinding binding;
    private Camera camera;
    private CameraPreview cameraPreview;
    private long currentRecordID;
    private static final int CAMERA_PERMISSION_REQUEST = 1001;
    private static final int GALLERY_VIDEO_REQUEST = 1002;
    private static final int STORAGE_PERMISSION_REQUEST = 1003;

    // Video recording & MediaRecorder
    private android.media.MediaRecorder mediaRecorder;
    private boolean isRecording = false;
    private int videoWidth = 1280;
    private int videoHeight = 720;
    private String currentVideoPath;

    // Timer for recording
    private android.os.Handler timerHandler = new android.os.Handler();
    private int recordingSeconds = 0;
    private final Runnable timerRunnable = new Runnable() {
        @Override
        public void run() {
            recordingSeconds++;
            if (binding != null) {
                binding.timerRecording.setText(formatTime(recordingSeconds));
            }
            timerHandler.postDelayed(this, 1000);
        }
    };

    // main task
    // - by default open camera and show record button
    // - give option to select video from gallery at the most bottom
    // - after recording video / selecting video, show a tray
    // - tray show video player and two buttons "process" and "discard"
    // - on clicking "process", analyze the video using image processing to get timeline of heart beat along the video
    // - after the timeline is generated, show a graph of heart rate over time
    // - analyze the heart rate to detect arrhythmia


    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        binding = FragmentRecordBinding.inflate(inflater, container, false);
        dataRecordService = new DataRecordServiceImpl(requireContext());
        new Thread(() -> {
            currentRecordID = dataRecordService.getNextId();
            requireActivity().runOnUiThread(() -> {
                // Any UI updates after getting the ID, if needed
            });
        }).start();
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        // Recording Processes:
        // - [DONE] show camera preview & display record button
        // - [DONE] click record button trigger start recording video
        // - [DONE] click record button again to stop recording, stop camera preview & hide record button
        // - [ONGOING] show confirmation view section
        //    the goals is to show to user current video if it's already recorded correctl
        //    here there's two button process or record again
        //    -> record again will show camera preview again)
        //    -> click process button will analyze the video and show graph view with heart rate timeline
        // - after processing complete, go to DetailFragment to show:
        //   -> graph view with heart rate timeline
        //   -> arrhythmia detection result
        //   -> etc


        // Request camera permission and start camera preview
        if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(requireActivity(),
                    new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST);
        } else {
            startCamera();
        }

        // Record button click (floating action button style)
        binding.btnRecord.setOnClickListener(v -> {
            if (!isRecording) {
                boolean audioNotGranted = ContextCompat.checkSelfPermission(requireActivity(), Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED;
                boolean writeStorageNotGranted = android.os.Build.VERSION.SDK_INT <= 28 && ContextCompat.checkSelfPermission(requireActivity(), Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED;
                if (audioNotGranted || writeStorageNotGranted) {
                    ActivityCompat.requestPermissions(
                        requireActivity(),
                        new String[]{
                            Manifest.permission.RECORD_AUDIO,
                            Manifest.permission.WRITE_EXTERNAL_STORAGE
                        }, CAMERA_PERMISSION_REQUEST
                    );
                    return;
                }

                if (startRecordingVideo()) {
                    isRecording = true;
                    binding.btnRecord.setImageResource(android.R.drawable.ic_media_pause); // change icon to stop
                    binding.recordingOverlay.setVisibility(View.VISIBLE);
                    recordingSeconds = 0;
                    binding.timerRecording.setText("00:00");
                    timerHandler.postDelayed(timerRunnable, 1000);
                    binding.tray.setVisibility(View.GONE);
                    binding.graphView.setVisibility(View.GONE);
                }
                return;
            }

            // Stop recording
            stopRecordingVideo();
            isRecording = false;
            binding.btnRecord.setImageResource(android.R.drawable.ic_btn_speak_now); // change icon to record
            binding.recordingOverlay.setVisibility(View.GONE);
            timerHandler.removeCallbacks(timerRunnable);
            // Show tray and set video to videoView
            binding.tray.setVisibility(View.VISIBLE);
            binding.videoView.setVideoPath(currentVideoPath);
            binding.videoView.seekTo(1);
            binding.graphView.setVisibility(View.GONE);
        });


        // Select from gallery button click (bottom control)
        binding.btnSelectGallery.setOnClickListener(v -> {
            if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.READ_EXTERNAL_STORAGE)
              != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(requireActivity(),
                  new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, STORAGE_PERMISSION_REQUEST);
                return;
            }
            launchGalleryPicker();
        });


        /* Confirmation Components */
        // impl @+id/video_view
        // showing preview of previously recorded video
        if (currentVideoPath != null && !currentVideoPath.isEmpty()) {
            binding.videoView.setVideoPath(currentVideoPath);
            binding.videoView.setOnPreparedListener(mp -> {
                // Set looping playback
                mp.setLooping(true);
                // Start playing
                binding.videoView.start();
            });
        }

        // Discard button click (tray)
        binding.btnDiscard.setOnClickListener(v -> {
            binding.tray.setVisibility(View.GONE);
            binding.videoView.setVideoURI(null);
            binding.graphView.setVisibility(View.GONE);
            // Reset to allow new recording or gallery selection
            if (camera == null) {
                startCamera();
            }
            binding.btnRecord.setVisibility(View.VISIBLE);
        });

        MediaProcessingServiceImpl mediaProcessingService = new MediaProcessingServiceImpl();
        binding.btnProcess.setOnClickListener(v -> {
            new Thread(() -> {
                mediaProcessingService.createHeartBeatsVideo(currentRecordID);

                if (currentRecordID > 1) {
                    RecordEntry entry = RecordEntry.builder()
                        .id(currentRecordID)
                        .patientName("")
                        .createAt(System.currentTimeMillis())
                        .status(RecordEntry.Status.UNCHECKED)
                        .build();
                    dataRecordService.saveData(entry);
                }
                // Ensure navigation happens after record creation
                requireActivity().runOnUiThread(() -> {
                    binding.graphView.setVisibility(View.VISIBLE);
                    navigateToDetailFragment();
                });
            }).start();
        });
        // Process button click (tray)
    }

    private void navigateToDetailFragment() {
        // Create a Bundle to pass the record ID
        Bundle bundle = new Bundle();
        bundle.putString("id", String.valueOf(currentRecordID)); // Pass the record ID

        androidx.navigation.NavController navController = androidx.navigation.Navigation.findNavController(requireView());
        navController.navigate(R.id.action_recordFragment_to_detailFragment, bundle);
    }

    private void launchGalleryPicker() {
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType("video/*");
        startActivityForResult(intent, GALLERY_VIDEO_REQUEST);
    }
    
    @Override
    public void onActivityResult(int requestCode, int resultCode, android.content.Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == GALLERY_VIDEO_REQUEST && resultCode == android.app.Activity.RESULT_OK && data != null) {
            android.net.Uri videoUri = data.getData();
            if (videoUri != null) {
                binding.tray.setVisibility(View.VISIBLE);
                binding.videoView.setVideoURI(videoUri);
                binding.videoView.seekTo(1);
                binding.graphView.setVisibility(View.GONE);
            }
        }
    }

    // Format seconds to mm:ss
    private String formatTime(int seconds) {
        int min = seconds / 60;
        int sec = seconds % 60;
        return String.format(java.util.Locale.US, "%02d:%02d", min, sec);
    }

    private void startCamera() {
        try {
            camera = Camera.open();
            Camera.Parameters parameters = camera.getParameters();
            List<Camera.Size> supportedPreviewSizes = parameters.getSupportedPreviewSizes();
            List<Camera.Size> supportedVideoSizes = null;
            try {
                supportedVideoSizes = (List<Camera.Size>) Camera.class.getMethod("getSupportedVideoSizes").invoke(camera);
            } catch (Exception e) {
                // fallback: use preview sizes
                supportedVideoSizes = supportedPreviewSizes;
            }
            // Find a size supported by both preview and video, prefer 1280x720, then 640x480
            Camera.Size chosenSize = null;
            for (Camera.Size size : supportedPreviewSizes) {
                for (Camera.Size vsize : supportedVideoSizes) {
                    if (size.width == vsize.width && size.height == vsize.height) {
                        if ((size.width == 1280 && size.height == 720) || (size.width == 720 && size.height == 1280)) {
                            chosenSize = size;
                            break;
                        }
                        if ((size.width == 640 && size.height == 480) || (size.width == 480 && size.height == 640)) {
                            if (chosenSize == null) chosenSize = size;
                        }
                    }
                }
                if (chosenSize != null && (chosenSize.width == 1280 || chosenSize.width == 720)) break;
            }
            if (chosenSize == null && !supportedPreviewSizes.isEmpty()) {
                chosenSize = supportedPreviewSizes.get(0);
            }
            if (chosenSize != null) {
                parameters.setPreviewSize(chosenSize.width, chosenSize.height);
                // Save for video recording
                this.videoWidth = chosenSize.width;
                this.videoHeight = chosenSize.height;
                android.util.Log.i("RecordFragment", "Using preview/video size: " + chosenSize.width + "x" + chosenSize.height);
            }
            camera.setParameters(parameters);

            cameraPreview = new CameraPreview(requireContext(), camera);
            FrameLayout preview = binding.cameraPreview;
            preview.removeAllViews();
            preview.addView(cameraPreview);

            // Fix camera rotation
            int rotation = requireActivity().getWindowManager().getDefaultDisplay().getRotation();
            int degrees = 0;
            switch (rotation) {
                case Surface.ROTATION_0: degrees = 0; break;
                case Surface.ROTATION_90: degrees = 90; break;
                case Surface.ROTATION_180: degrees = 180; break;
                case Surface.ROTATION_270: degrees = 270; break;
            }
            Camera.CameraInfo info = new Camera.CameraInfo();
            Camera.getCameraInfo(Camera.CameraInfo.CAMERA_FACING_BACK, info);
            int result;
            if (info.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
                result = (info.orientation + degrees) % 360;
                result = (360 - result) % 360;  // compensate the mirror
            } else { // back-facing
                result = (info.orientation - degrees + 360) % 360;
            }
            camera.setDisplayOrientation(result);

            // Set aspect ratio of camera_preview container
            if (chosenSize != null) {
                int width = chosenSize.width;
                int height = chosenSize.height;
                if (result == 90 || result == 270) {
                    int temp = width;
                    width = height;
                    height = temp;
                }
                String ratioString = width + ":" + height;
                androidx.constraintlayout.widget.ConstraintLayout.LayoutParams params =
                        (androidx.constraintlayout.widget.ConstraintLayout.LayoutParams) preview.getLayoutParams();
                params.dimensionRatio = ratioString;
                preview.setLayoutParams(params);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // --- Video Recording Logic ---
    private boolean startRecordingVideo() {
        if (camera == null) return false;

        // createVideo
        try {
            camera.unlock();
            mediaRecorder = new android.media.MediaRecorder();

            // Order matters! Set sources first
            mediaRecorder.setCamera(camera);
            mediaRecorder.setAudioSource(android.media.MediaRecorder.AudioSource.CAMCORDER);
            mediaRecorder.setVideoSource(android.media.MediaRecorder.VideoSource.CAMERA);

            // Then set profile and output format
            mediaRecorder.setOutputFormat(android.media.MediaRecorder.OutputFormat.MPEG_4);
            mediaRecorder.setAudioEncoder(android.media.MediaRecorder.AudioEncoder.AAC);
            mediaRecorder.setVideoEncoder(android.media.MediaRecorder.VideoEncoder.H264);
            mediaRecorder.setVideoEncodingBitRate(10000000); // 10Mbps
            mediaRecorder.setVideoFrameRate(30);

            // Set the output resolution (make sure it matches preview size)
            mediaRecorder.setVideoSize(videoWidth, videoHeight);

            // Set orientation
            int rotation = requireActivity().getWindowManager().getDefaultDisplay().getRotation();
            int orientation = 0;
            switch (rotation) {
                case Surface.ROTATION_0: orientation = 90; break;
                case Surface.ROTATION_90: orientation = 0; break;
                case Surface.ROTATION_180: orientation = 270; break;
                case Surface.ROTATION_270: orientation = 180; break;
            }
            mediaRecorder.setOrientationHint(orientation);

            // Output file
            java.io.File videoFile = getOutputMediaFile();
            if (videoFile == null) {
                android.util.Log.e("RecordFragment", "Failed to create video file");
                return false;
            }
            currentVideoPath = videoFile.getAbsolutePath();
            mediaRecorder.setOutputFile(currentVideoPath);

            // Set preview display - must be done before prepare()
            mediaRecorder.setPreviewDisplay(cameraPreview.mSurfaceView.getHolder().getSurface());

            // Prepare the recorder
            try {
                mediaRecorder.prepare();
            } catch (Exception e) {
                android.util.Log.e("RecordFragment", "MediaRecorder prepare failed: " + e.getMessage(), e);
                releaseMediaRecorder();
                try { camera.reconnect(); } catch (Exception ignored) {}
                try { camera.startPreview(); } catch (Exception ignored) {}
                return false;
            }

            // Start recording
            try {
                mediaRecorder.start();
                return true;
            } catch (Exception e) {
                android.util.Log.e("RecordFragment", "MediaRecorder start failed: " + e.getMessage(), e);
                releaseMediaRecorder();
                try { camera.reconnect(); } catch (Exception ignored) {}
                try { camera.startPreview(); } catch (Exception ignored) {}
                return false;
            }
        } catch (Exception e) {
            android.util.Log.e("RecordFragment", "MediaRecorder setup failed: " + e.getMessage(), e);
            releaseMediaRecorder();
            try { camera.reconnect(); } catch (Exception ignored) {}
            try { camera.startPreview(); } catch (Exception ignored) {}
            return false;
        }
    }

    private void stopRecordingVideo() {
        try {
            if (mediaRecorder != null) {
                mediaRecorder.stop();
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            releaseMediaRecorder();
            if (camera != null) {
                try {
                    camera.lock();
                } catch (Exception ignored) {}
            }
        }
    }

    private void releaseMediaRecorder() {
        if (mediaRecorder != null) {
            mediaRecorder.reset();
            mediaRecorder.release();
            mediaRecorder = null;
        }
    }

    private File getOutputMediaFile() {
        // Create a new RecordEntry and save to DB
        File dataDirectory = new File(
            requireContext().getExternalFilesDir(null),
            String.format("%s/%s", AppConstant.DATA_DIR, currentRecordID)
        );
        if (!dataDirectory.exists()) {
            if (!dataDirectory.mkdirs()) {
                return null;
            }
        }
        String timeStamp = new java.text.SimpleDateFormat("yyyyMMdd_HHmmss", java.util.Locale.US).format(new java.util.Date());
        return new File(dataDirectory.getPath() + File.separator + "VID_" + timeStamp + ".mp4");
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_REQUEST) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCamera();
            }
        } else if (requestCode == STORAGE_PERMISSION_REQUEST) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                launchGalleryPicker();
            }
        }
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        timerHandler.removeCallbacks(timerRunnable);
        if (isRecording) {
            stopRecordingVideo();
            isRecording = false;
        }
        releaseMediaRecorder();
        if (camera != null) {
            camera.release();
            camera = null;
        }
        binding = null;
    }

    // CameraPreview class (can be inner or separate file)
    public class CameraPreview extends FrameLayout implements android.view.SurfaceHolder.Callback {
        private Camera mCamera;
        private android.view.SurfaceView mSurfaceView;
        private android.view.SurfaceHolder mHolder;

        public CameraPreview(android.content.Context context, Camera camera) {
            super(context);
            mCamera = camera;
            mSurfaceView = new android.view.SurfaceView(context);
            addView(mSurfaceView);

            mHolder = mSurfaceView.getHolder();
            mHolder.addCallback(this);

            mSurfaceView.setLayoutParams(new FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            ));
        }

        @Override
        public void surfaceCreated(android.view.SurfaceHolder holder) {
            try {
                Camera.Parameters parameters = mCamera.getParameters();

                // Force 16:9 aspect ratio, with long side as height (portrait 9:16)
                int targetWidth = 9;
                int targetHeight = 16;
                double targetRatio = (double) targetHeight / targetWidth; // 16:9 portrait

                // Find the supported preview size closest to 9:16 (portrait)
                List<Camera.Size> supportedSizes = parameters.getSupportedPreviewSizes();
                Camera.Size bestSize = null;
                double minDiff = Double.MAX_VALUE;
                for (Camera.Size size : supportedSizes) {
                    double ratio = (double) Math.max(size.width, size.height) / Math.min(size.width, size.height);
                    double diff = Math.abs(ratio - targetRatio);
                    if (diff < minDiff) {
                        minDiff = diff;
                        bestSize = size;
                    }
                }
                if (bestSize != null) {
                    parameters.setPreviewSize(bestSize.width, bestSize.height);
                    // Set the aspect ratio of the container to 9:16 (portrait)
                    requireActivity().runOnUiThread(() -> {
                        FrameLayout preview = binding.cameraPreview;
                        String ratioString = "9:16";
                        androidx.constraintlayout.widget.ConstraintLayout.LayoutParams params =
                                (androidx.constraintlayout.widget.ConstraintLayout.LayoutParams) preview.getLayoutParams();
                        params.dimensionRatio = ratioString;
                        preview.setLayoutParams(params);
                    });
                }

                // Set focus mode to continuous video if supported
                if (parameters.getSupportedFocusModes().contains(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO)) {
                    parameters.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO);
                }

                mCamera.setParameters(parameters);
                mCamera.setPreviewDisplay(holder);
                mCamera.startPreview();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        // Helper method to find the optimal preview size
        private Camera.Size getOptimalPreviewSize(List<Camera.Size> sizes, int width, int height) {
            double targetRatio = (double) width / height;
            Camera.Size optimalSize = null;
            double minDiff = Double.MAX_VALUE;

            for (Camera.Size size : sizes) {
                double ratio = (double) size.width / size.height;
                if (Math.abs(ratio - targetRatio) < minDiff) {
                    optimalSize = size;
                    minDiff = Math.abs(ratio - targetRatio);
                }
            }
            return optimalSize;
        }

        @Override
        public void surfaceDestroyed(android.view.SurfaceHolder holder) {}

        @Override
        public void surfaceChanged(android.view.SurfaceHolder holder, int format, int width, int height) {
            if (mHolder.getSurface() == null) return;
            try {
                mCamera.stopPreview();
            } catch (Exception ignored) {}
            try {
                mCamera.setPreviewDisplay(mHolder);
                mCamera.startPreview();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}
