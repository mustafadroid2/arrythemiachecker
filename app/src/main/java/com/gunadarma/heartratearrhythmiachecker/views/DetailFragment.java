package com.gunadarma.heartratearrhythmiachecker.views;

import android.app.Dialog;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.navigation.fragment.NavHostFragment;

import com.gunadarma.heartratearrhythmiachecker.R;
import com.gunadarma.heartratearrhythmiachecker.constant.AppConstant;
import com.gunadarma.heartratearrhythmiachecker.databinding.FragmentDetailBinding;
import com.gunadarma.heartratearrhythmiachecker.model.RecordEntry;
import com.gunadarma.heartratearrhythmiachecker.service.DataRecordServiceImpl;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

import android.util.Log;
import android.widget.ImageButton;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.gunadarma.heartratearrhythmiachecker.service.MainMediaProcessingServiceImpl;
import com.gunadarma.heartratearrhythmiachecker.util.AppUtil;

public class DetailFragment extends Fragment {
    private FragmentDetailBinding binding;
    private DataRecordServiceImpl dataRecordService;
    private MainMediaProcessingServiceImpl mediaProcessingService;

    // main tasks
    // - show detail information about selected record
    //   -> show id, date, time, heart rate, arrhythmia status
    // - show heart rhythm timeline graph
    // - provide video playback of the recorded video (option for fullscreen mode when playing)
    // - provide option to delete the record
    // - provide option to share the record details

    private boolean isEditMode = false;
    private RecordEntry currentRecordEntry;

    // Video switching variables
    private boolean isShowingFinalVideo = false;
    private File originalVideoFile;
    private File finalVideoFile;

    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        binding = FragmentDetailBinding.inflate(inflater, container, false);
        dataRecordService = new DataRecordServiceImpl(requireContext());
        mediaProcessingService = new MainMediaProcessingServiceImpl(requireContext());

        // Get record ID from arguments
        if (getArguments() != null && getArguments().containsKey("id")) {
            String recordId = getArguments().getString("id");
            if (recordId != null && !recordId.isEmpty()) {
                new Thread(() -> {
                    RecordEntry record = dataRecordService.get(recordId);
                    if (record != null) {
                        requireActivity().runOnUiThread(() -> {
                            binding.textId.setText("ID: " + record.getId());
                            // Set status in tag view
                            binding.textStatus.setText(record.getStatus().toString());
                            // ...rest of the UI updates...
                        });
                    }
                }).start();
            }
        }

        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        // Setup pull-to-refresh
        binding.swipeRefreshLayout.setOnRefreshListener(() -> {
            refreshData();
        });

        // Configure refresh colors
        binding.swipeRefreshLayout.setColorSchemeResources(
            android.R.color.holo_blue_bright,
            android.R.color.holo_green_light,
            android.R.color.holo_orange_light,
            android.R.color.holo_red_light
        );

        // Log getArguments() as JSON
        Bundle args = getArguments();
        if (args != null) {
            try {
                ObjectMapper objectMapper = new ObjectMapper();
                Map<String, Object> map = new HashMap<>();
                for (String key : args.keySet()) {
                    map.put(key, args.get(key));
                }
                String json = objectMapper.writeValueAsString(map);
                Log.d("DetailFragmentArgs", json);
            } catch (Exception e) {
                Log.e("DetailFragmentArgs", "Failed to log arguments as JSON", e);
            }
        }

        // Show loading indicator (make sure you have a ProgressBar with id progressBar in your layout)
        if (binding.progressBar != null) {
            binding.progressBar.setVisibility(View.VISIBLE);
        }
        binding.detailContent.setVisibility(View.GONE); // Wrap your detail content in a layout with id detailContent

        // Setup floating action buttons
        binding.btnEdit.setOnClickListener(v -> {
            setEditMode(true);
            binding.btnEdit.setVisibility(View.GONE);
            binding.btnToggleVideo.setVisibility(View.GONE);
            binding.btnSaveFloat.setVisibility(View.VISIBLE);
            binding.btnCancel.setVisibility(View.VISIBLE);
        });

        binding.btnSaveFloat.setOnClickListener(v -> {
            if (currentRecordEntry != null) {
                // Update record with edited values
                currentRecordEntry.setPatientName(binding.editPatientName.getText().toString());
                currentRecordEntry.setNotes(binding.editNotes.getText().toString());

                // Update age and address
                String ageStr = binding.editAge.getText().toString();
                currentRecordEntry.setAge(ageStr.isEmpty() ? null : Integer.valueOf(ageStr));
                currentRecordEntry.setAddress(binding.editAddress.getText().toString());

                // Update gender based on radio button selection
                String selectedGender = null;
                if (binding.radioMale.isChecked()) {
                    selectedGender = "Male";
                } else if (binding.radioFemale.isChecked()) {
                    selectedGender = "Female";
                }
                currentRecordEntry.setGender(selectedGender);

                // Save in background thread
                new Thread(() -> {
                    DataRecordServiceImpl dataRecordService = new DataRecordServiceImpl(requireContext());
                    dataRecordService.saveData(currentRecordEntry);

                    // Update UI on main thread
                    requireActivity().runOnUiThread(() -> {
                        setEditMode(false);
                        binding.btnSaveFloat.setVisibility(View.GONE);
                        binding.btnCancel.setVisibility(View.GONE);
                        binding.btnEdit.setVisibility(View.VISIBLE);
                        binding.btnToggleVideo.setVisibility(View.VISIBLE);

                        // Hide keyboard
                        android.view.inputmethod.InputMethodManager imm = (android.view.inputmethod.InputMethodManager)
                            requireActivity().getSystemService(android.content.Context.INPUT_METHOD_SERVICE);

                        refreshEntryData();
                        android.widget.Toast.makeText(requireContext(), "Updated", android.widget.Toast.LENGTH_SHORT).show();
                    });
                }).start();
            }
        });

        binding.btnCancel.setOnClickListener(v -> {
            setEditMode(false);
            binding.btnSaveFloat.setVisibility(View.GONE);
            binding.btnCancel.setVisibility(View.GONE);
            binding.btnEdit.setVisibility(View.VISIBLE);
            binding.btnToggleVideo.setVisibility(View.VISIBLE);

            // Hide keyboard
            android.view.inputmethod.InputMethodManager imm = (android.view.inputmethod.InputMethodManager)
                requireActivity().getSystemService(android.content.Context.INPUT_METHOD_SERVICE);

            // Reset edit fields to current values
            refreshEntryData();
        });

        // Handle quick note EditText "Done" action
        // Remove the old save button click listener since we're using the floating button now
        binding.btnSave.setVisibility(View.GONE);

        // Fetch record data asynchronously
        if (args != null && args.containsKey("id")) {
            String dataId = String.valueOf(args.get("id"));
            new Thread(() -> {
                DataRecordServiceImpl dataRecordService = new DataRecordServiceImpl(requireContext());
                final RecordEntry data = dataRecordService.get(dataId);
                requireActivity().runOnUiThread(() -> {
                    currentRecordEntry = data;
                    if (data != null) {
                        requireActivity().setTitle(String.format("Detail #%s", currentRecordEntry.getId()));
                        binding.textId.setText(AppUtil.toDetailDate(data.getCreateAt()));
                        binding.textId.setTypeface(null, android.graphics.Typeface.BOLD);
                        binding.textId.setTextSize(14);
                        // adjust textId text color to gray
                        binding.textStatus.setTextColor(getResources().getColor(com.google.android.material.R.color.material_dynamic_neutral100, null));


                        binding.textStatus.setText(data.getStatus().getValue());

                        binding.textPatientName.setText("Patient name: " + AppUtil.patientNameOrDefault(currentRecordEntry));
                        if (currentRecordEntry.getPatientName() == null || currentRecordEntry.getPatientName().isEmpty()) {
                            binding.textPatientName.setTypeface(null, android.graphics.Typeface.ITALIC);
                        } else {
                            binding.textPatientName.setTypeface(null, android.graphics.Typeface.NORMAL);
                        }

                        binding.textHeartRate.setText("Heart Rate: " + data.getBeatsPerMinute() + " bpm");
                        binding.editPatientName.setText(data.getPatientName());
                        binding.textNotes.setText("Notes: " + (data.getNotes() != null ? data.getNotes() : ""));
                        binding.editNotes.setText(data.getNotes() != null ? data.getNotes() : "");
                        binding.textDuration.setText("Duration: " + data.getDuration() + " seconds");

                        // Set up video playback using the actual video file
                        try {
                            finalVideoFile = new File(
                                requireContext().getExternalFilesDir(null),
                                String.format("%s/%s/%s", AppConstant.DATA_DIR, data.getId(), AppConstant.FINAL_VIDEO_NAME)
                            );
                            originalVideoFile = new File(
                                requireContext().getExternalFilesDir(null),
                                String.format("%s/%s/%s", AppConstant.DATA_DIR, data.getId(), AppConstant.ORIGINAL_VIDEO_NAME)
                            );

                            // Initialize video switch pills based on available videos
                            updateVideoSwitchPills();

                            File videoToPlay = finalVideoFile.exists() ? finalVideoFile : originalVideoFile;

                            if (videoToPlay.exists()) {
                                android.widget.MediaController mediaController = new android.widget.MediaController(requireContext());
                                binding.videoView.setMediaController(mediaController);
                                mediaController.setAnchorView(binding.videoView);

                                // Set audio attributes to not interfere with background playback
                                binding.videoView.setAudioFocusRequest(android.media.AudioManager.AUDIOFOCUS_NONE);

                                binding.videoView.setOnClickListener(v -> {
                                    if (binding.videoView.isPlaying()) {
                                        binding.videoView.pause();
                                    } else {
                                        binding.videoView.start();
                                    }
                                });
                            } else {
                                Log.e("DetailFragment", "No video file found at either path");
                                binding.videoContainer.setVisibility(View.GONE);
                            }
                        } catch (Exception e) {
                            Log.e("DetailFragment", "Failed to load video: " + e.getMessage());
                            binding.videoContainer.setVisibility(View.GONE);
                        }
                    }
                    if (binding.progressBar != null) {
                        binding.progressBar.setVisibility(View.GONE);
                    }
                    binding.detailContent.setVisibility(View.VISIBLE);
                });
            }).start();
        }

        // Load heart rhythm timeline graph from data directory
        if (args != null && args.containsKey("id")) {
            String dataId = String.valueOf(args.get("id"));
            try {
                File heartbeatsFile = new File(
                    requireContext().getExternalFilesDir(null),
                    String.format("%s/%s/heartbeats.jpg", AppConstant.DATA_DIR, dataId)
                );
                if (heartbeatsFile.exists()) {
                    binding.graphView.setImageURI(android.net.Uri.fromFile(heartbeatsFile));
                } else {
                    Log.e("DetailFragment", "Heartbeats graph not found: " + heartbeatsFile.getAbsolutePath());
                    binding.graphView.setVisibility(View.GONE);
                }
            } catch (Exception e) {
                Log.e("DetailFragment", "Failed to load heartbeats.jpg", e);
                binding.graphView.setVisibility(View.GONE);
            }
        }

        // Delete record
        binding.btnDelete.setOnClickListener(v -> {
            if (args != null && args.containsKey("id")) {
                new androidx.appcompat.app.AlertDialog.Builder(requireContext())
                        .setTitle("Delete Record")
                        .setMessage("Are you sure you want to delete this record?")
                        .setPositiveButton("Yes", (dialog, which) -> {
                            String dataId = String.valueOf(args.get("id"));
                            new Thread(() -> {
                                DataRecordServiceImpl dataRecordService = new DataRecordServiceImpl(requireContext());
                                RecordEntry data = dataRecordService.get(dataId);
                                if (data != null) {
                                    dataRecordService.removeData(data);
                                }
                                requireActivity().runOnUiThread(() -> {
                                    NavHostFragment.findNavController(DetailFragment.this)
                                            .navigate(R.id.action_DetailFragment_to_HomeFragment);
                                });
                            }).start();
                        })
                        .setNegativeButton("No", null)
                        .show();
            }
        });

        // Share record
        binding.btnShare.setOnClickListener(v -> {
            android.content.Intent shareIntent = new android.content.Intent(android.content.Intent.ACTION_SEND);
            shareIntent.setType("text/plain");
            shareIntent.putExtra(android.content.Intent.EXTRA_TEXT, "Hello World");
            startActivity(android.content.Intent.createChooser(shareIntent, "Share via"));
        });

        // Set up click listener for graph image zoom
        binding.graphView.setOnClickListener(v -> {
            showFullscreenImage();
        });

        // Add process button click handler
        binding.btnProcess.setOnClickListener(v -> {
            if (currentRecordEntry == null) {
                android.widget.Toast.makeText(requireContext(),
                    "No record available to process",
                    android.widget.Toast.LENGTH_SHORT).show();
                return;
            }

            // Use shared video processing utility
            com.gunadarma.heartratearrhythmiachecker.util.VideoProcessingUtil.processVideo(currentRecordEntry,
                new com.gunadarma.heartratearrhythmiachecker.util.VideoProcessingUtil.ProcessingCallback() {
                    @Override
                    public void onProgressUpdate(int progress, String message) {
                        // Progress updates are handled internally by the utility
                    }

                    @Override
                    public void onSuccess() {
                        // Update status display and refresh data
                        if (currentRecordEntry != null) {
                            binding.textStatus.setText(currentRecordEntry.getStatus().getValue());
                        }
                        // Refresh entry data to show updated videos and graphs
                        refreshEntryData();
                        // Update video switch pills visibility after processing
                        updateVideoSwitchPills();
                    }

                    @Override
                    public void onError(Exception error) {
                        // Show error dialog with retry option
                        com.gunadarma.heartratearrhythmiachecker.util.VideoProcessingUtil.showProcessingErrorDialog(
                            requireContext(),
                            error,
                            () -> binding.btnProcess.performClick(), // Retry
                            null // No view details action needed in detail fragment
                        );
                    }

                    @Override
                    public void runOnUiThread(Runnable runnable) {
                        requireActivity().runOnUiThread(runnable);
                    }

                    @Override
                    public android.content.Context getContext() {
                        return requireContext();
                    }
                });
        });

        binding.btnToggleVideo.setOnClickListener(v -> {
            isShowingFinalVideo = !isShowingFinalVideo;
            switchToVideo(isShowingFinalVideo);
            updateToggleButtonText();
        });
    }

    private void showFullscreenImage() {
        Dialog dialog = new Dialog(requireContext(), android.R.style.Theme_Black_NoTitleBar_Fullscreen);
        dialog.setContentView(R.layout.dialog_fullscreen_image);

        com.github.chrisbanes.photoview.PhotoView photoView = dialog.findViewById(R.id.fullscreen_image);
        ImageButton closeButton = dialog.findViewById(R.id.btn_close);

        // Copy the current drawable from the graph view
        photoView.setImageDrawable(binding.graphView.getDrawable());

        closeButton.setOnClickListener(v -> dialog.dismiss());

        dialog.show();
    }

    private void setEditMode(boolean isEditing) {
        int editVisibility = isEditing ? View.VISIBLE : View.GONE;
        int textVisibility = isEditing ? View.GONE : View.VISIBLE;

        binding.editPatientName.setVisibility(editVisibility);
        binding.editNotes.setVisibility(editVisibility);
        binding.editAge.setVisibility(editVisibility);
        binding.editAddress.setVisibility(editVisibility);

        // Show/hide gender radio group in edit mode
        binding.radioGroupGender.setVisibility(editVisibility);

        binding.textPatientName.setVisibility(textVisibility);
        binding.textNotes.setVisibility(textVisibility);
        binding.textAge.setVisibility(textVisibility);
        binding.textAddress.setVisibility(textVisibility);
        binding.textDuration.setVisibility(textVisibility);
        binding.textGender.setVisibility(textVisibility);
    }

    private void refreshEntryData() {
        if (currentRecordEntry != null) {
            // Patient Name
            binding.textPatientName.setText("Patient name: " + AppUtil.patientNameOrDefault(currentRecordEntry));
            binding.editPatientName.setText(currentRecordEntry.getPatientName());
            if (currentRecordEntry.getPatientName() == null || currentRecordEntry.getPatientName().isEmpty()) {
                binding.textPatientName.setTypeface(null, android.graphics.Typeface.ITALIC);
            } else {
            }
            binding.textPatientName.setVisibility((currentRecordEntry.getPatientName() == null || currentRecordEntry.getPatientName().isEmpty()) ? View.GONE : View.VISIBLE);

            // Age
            if (currentRecordEntry.getAge() != null) {
                binding.textAge.setText("Age: " + currentRecordEntry.getAge());
                binding.textAge.setVisibility(View.VISIBLE);
            } else {
                binding.textAge.setVisibility(View.GONE);
            }
            binding.editAge.setText(currentRecordEntry.getAge() != null ? String.valueOf(currentRecordEntry.getAge()) : "");

            // Gender
            if (currentRecordEntry.getGender() != null && !currentRecordEntry.getGender().isEmpty()) {
                binding.textGender.setText("Gender: " + currentRecordEntry.getGender());
                binding.textGender.setVisibility(View.VISIBLE);
            } else {
                binding.textGender.setVisibility(View.GONE);
            }
            // Set radio button selection based on current gender
            binding.radioMale.setChecked("Male".equals(currentRecordEntry.getGender()));
            binding.radioFemale.setChecked("Female".equals(currentRecordEntry.getGender()));

            // Address
            if (currentRecordEntry.getAddress() != null && !currentRecordEntry.getAddress().isEmpty()) {
                binding.textAddress.setText("Address: " + currentRecordEntry.getAddress());
                binding.textAddress.setVisibility(View.VISIBLE);
            } else {
                binding.textAddress.setVisibility(View.GONE);
            }
            binding.editAddress.setText(currentRecordEntry.getAddress() != null ? currentRecordEntry.getAddress() : "");

            // Notes
            if (currentRecordEntry.getNotes() != null && !currentRecordEntry.getNotes().isEmpty()) {
                binding.textNotes.setText("Notes: " + currentRecordEntry.getNotes());
                binding.textNotes.setVisibility(View.VISIBLE);
            } else {
                binding.textNotes.setVisibility(View.GONE);
            }
            binding.editNotes.setText(currentRecordEntry.getNotes() != null ? currentRecordEntry.getNotes() : "");

            binding.textDuration.setText("Duration: " + currentRecordEntry.getDuration() + " seconds");

            // refresh video
            if (binding.videoView != null) {
                binding.videoView.stopPlayback();
                try {
                    File videoToPlay = finalVideoFile.exists() ? finalVideoFile : originalVideoFile;

                    if (videoToPlay.exists()) {
                        binding.videoView.setVideoPath(videoToPlay.getAbsolutePath());
                        binding.videoContainer.setVisibility(View.VISIBLE);
                        binding.videoView.seekTo(1); // Show first frame
                    } else {
                        binding.videoContainer.setVisibility(View.GONE);
                    }
                } catch (Exception e) {
                    Log.e("DetailFragment", "Failed to reload video: " + e.getMessage());
                    binding.videoContainer.setVisibility(View.GONE);
                }
            }

            // refresh heartbeats image
            try {
                File heartbeatsFile = new File(
                    requireContext().getExternalFilesDir(null),
                    String.format("%s/%s/heartbeats.jpg", AppConstant.DATA_DIR, currentRecordEntry.getId())
                );
                if (heartbeatsFile.exists()) {
                    // Clear the current image first
                    binding.graphView.setImageDrawable(null);
                    // Add a timestamp to the URI to prevent caching
                    android.net.Uri imageUri = android.net.Uri.fromFile(heartbeatsFile);
                    String uniqueUri = imageUri.toString() + "?timestamp=" + System.currentTimeMillis();
                    binding.graphView.setImageURI(android.net.Uri.parse(uniqueUri));
                    binding.graphView.setVisibility(View.VISIBLE);
                } else {
                    Log.e("DetailFragment", "Heartbeats graph not found: " + heartbeatsFile.getAbsolutePath());
                    binding.graphView.setVisibility(View.GONE);
                }
            } catch (Exception e) {
                Log.e("DetailFragment", "Failed to load heartbeats.jpg", e);
                binding.graphView.setVisibility(View.GONE);
            }
        }
    }

    private void refreshData() {
        Bundle args = getArguments();
        if (args != null && args.containsKey("id")) {
            String dataId = String.valueOf(args.get("id"));

            // Show refresh indicator
            binding.swipeRefreshLayout.setRefreshing(true);

            new Thread(() -> {
                try {
                    // Reload record data from database
                    DataRecordServiceImpl dataRecordService = new DataRecordServiceImpl(requireContext());
                    final RecordEntry data = dataRecordService.get(dataId);

                    requireActivity().runOnUiThread(() -> {
                        currentRecordEntry = data;
                        if (data != null) {
                            // Update title
                            requireActivity().setTitle(String.format("Detail #%s", currentRecordEntry.getId()));

                            // Update basic info
                            binding.textId.setText(AppUtil.toDetailDate(data.getCreateAt()));
                            binding.textStatus.setText(data.getStatus().getValue());
                            binding.textHeartRate.setText("Heart Rate: " + data.getBeatsPerMinute() + " bpm");
                            binding.textDuration.setText("Duration: " + data.getDuration() + " seconds");

                            // Refresh all entry data including age, gender, address
                            refreshEntryData();

                            // Reload video files
                            finalVideoFile = new File(
                                requireContext().getExternalFilesDir(null),
                                String.format("%s/%s/%s", AppConstant.DATA_DIR, data.getId(), AppConstant.FINAL_VIDEO_NAME)
                            );
                            originalVideoFile = new File(
                                requireContext().getExternalFilesDir(null),
                                String.format("%s/%s/%s", AppConstant.DATA_DIR, data.getId(), AppConstant.ORIGINAL_VIDEO_NAME)
                            );

                            // Update video display
                            updateVideoSwitchPills();

                            android.widget.Toast.makeText(requireContext(), "Data refreshed", android.widget.Toast.LENGTH_SHORT).show();
                        } else {
                            android.widget.Toast.makeText(requireContext(), "Record not found", android.widget.Toast.LENGTH_SHORT).show();
                        }

                        // Hide refresh indicator
                        binding.swipeRefreshLayout.setRefreshing(false);
                    });
                } catch (Exception e) {
                    Log.e("DetailFragment", "Error refreshing data", e);
                    requireActivity().runOnUiThread(() -> {
                        binding.swipeRefreshLayout.setRefreshing(false);
                        android.widget.Toast.makeText(requireContext(), "Failed to refresh", android.widget.Toast.LENGTH_SHORT).show();
                    });
                }
            }).start();
        } else {
            // No data to refresh
            binding.swipeRefreshLayout.setRefreshing(false);
            android.widget.Toast.makeText(requireContext(), "No data to refresh", android.widget.Toast.LENGTH_SHORT).show();
        }
    }

    private void switchToVideo(boolean showFinalVideo) {
        isShowingFinalVideo = showFinalVideo;

        File videoToPlay = showFinalVideo ? finalVideoFile : originalVideoFile;

        if (videoToPlay != null && videoToPlay.exists()) {
            // Stop current video playback
            if (binding.videoView.isPlaying()) {
                binding.videoView.stopPlayback();
            }

            binding.videoView.setVideoPath(videoToPlay.getAbsolutePath());
            binding.videoContainer.setVisibility(View.VISIBLE);

            // Show first frame without starting playback
            binding.videoView.setOnPreparedListener(mp -> {
                mp.setLooping(false);
                binding.videoView.seekTo(1);
            });
        } else {
            // If selected video doesn't exist, show a toast message
            String videoType = showFinalVideo ? "Switch Final" : "Switch Original";
            android.widget.Toast.makeText(requireContext(),
                videoType + " video not available",
                android.widget.Toast.LENGTH_SHORT).show();
            return;
        }

        updateToggleButtonText();
    }

    private void updateToggleButtonText() {
        if (binding.btnToggleVideo != null) {
            binding.btnToggleVideo.setText(isShowingFinalVideo ? "Switch Original" : "Switch Final");
        }
    }

    private void updateVideoSwitchPills() {
        // Replaced by single toggle button logic
        boolean originalExists = originalVideoFile != null && originalVideoFile.exists();
        boolean finalExists = finalVideoFile != null && finalVideoFile.exists();
        binding.btnToggleVideo.setVisibility((originalExists || finalExists) ? View.VISIBLE : View.GONE);
        // Set initial video selection - prefer final if both exist
        if (finalExists) {
            switchToVideo(true);
        } else if (originalExists) {
            switchToVideo(false);
        }
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
