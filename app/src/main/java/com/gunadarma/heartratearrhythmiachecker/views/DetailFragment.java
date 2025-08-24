package com.gunadarma.heartratearrhythmiachecker.views;

import android.app.Dialog;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
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
import com.gunadarma.heartratearrhythmiachecker.service.MediaProcessingServiceImpl;
import com.gunadarma.heartratearrhythmiachecker.util.AppUtil;

public class DetailFragment extends Fragment {
    private FragmentDetailBinding binding;

    // main tasks
    // - show detail information about selected record
    //   -> show id, date, time, heart rate, arrhythmia status
    // - show heart rhythm timeline graph
    // - provide video playback of the recorded video (option for fullscreen mode when playing)
    // - provide option to delete the record
    // - provide option to share the record details

    private boolean isEditMode = false;
    private RecordEntry currentRecordEntry;

    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        binding = FragmentDetailBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

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
            binding.btnSaveFloat.setVisibility(View.VISIBLE);
            binding.btnCancel.setVisibility(View.VISIBLE);
        });

        binding.btnSaveFloat.setOnClickListener(v -> {
            if (currentRecordEntry != null) {
                // Update record with edited values
                currentRecordEntry.setPatientName(binding.editPatientName.getText().toString());
                currentRecordEntry.setNotes(binding.editNotes.getText().toString());
                try {
                    currentRecordEntry.setDuration(Integer.parseInt(binding.editDuration.getText().toString()));
                } catch (NumberFormatException e) {
                    // Handle invalid duration input
                    android.widget.Toast.makeText(requireContext(), "Invalid duration value", android.widget.Toast.LENGTH_SHORT).show();
                    return;
                }

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
                        updateDisplayValues();
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
            // Reset edit fields to current values
            updateDisplayValues();
        });

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
                        String recordDate = AppUtil.toDate(data.getCreateAt());
                        requireActivity().setTitle(String.format("Detail #%s", currentRecordEntry.getId()));
                        binding.textId.setText("#" + currentRecordEntry.getId() + " - " + recordDate);
                        binding.textPatientName.setText("Name: " + data.getPatientName());
                        binding.editDateLabel.setText(recordDate);
                        binding.textArrhythmia.setText("Status: " + data.getStatus().getValue());
                        binding.textHeartRate.setText("Heart Rate: " + data.getBeatsPerMinute());
                        binding.editPatientName.setText(data.getPatientName());
                        binding.textNotes.setText("Notes: " + (data.getNotes() != null ? data.getNotes() : ""));
                        binding.editNotes.setText(data.getNotes() != null ? data.getNotes() : "");
                        binding.textDuration.setText("Duration: " + data.getDuration());
                        binding.editDuration.setText(String.valueOf(data.getDuration()));

                        // Set up video playback using the actual video file
                        try {
                            File videoFile = new File(
                                requireContext().getExternalFilesDir(null),
                                String.format("%s/%s/original.mp4", AppConstant.DATA_DIR, data.getId())
                            );
                            if (videoFile.exists()) {
                                android.widget.MediaController mediaController = new android.widget.MediaController(requireContext());
                                binding.videoView.setMediaController(mediaController);
                                mediaController.setAnchorView(binding.videoView);

                                // Set audio attributes to not interfere with background playback
                                binding.videoView.setAudioFocusRequest(android.media.AudioManager.AUDIOFOCUS_NONE);
                                binding.videoView.setVideoPath(videoFile.getAbsolutePath());

                                // Show first frame without starting playback
                                binding.videoView.setOnPreparedListener(mp -> {
                                    mp.setLooping(true);
                                    binding.videoView.seekTo(1);
                                });

                                binding.videoView.setOnClickListener(v -> {
                                    if (binding.videoView.isPlaying()) {
                                        binding.videoView.pause();
                                    } else {
                                        binding.videoView.start();
                                    }
                                });
                            } else {
                                Log.e("DetailFragment", "Video file not found: " + videoFile.getAbsolutePath());
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

        // Show heart rhythm timeline graph from res/raw/mock_heartbeats.webp
        try {
            int graphResId = getResources().getIdentifier("mock_heartbeats", "raw", requireContext().getPackageName());
            if (graphResId != 0) {
                binding.graphView.setImageResource(graphResId);
            }
        } catch (Exception e) {
            Log.e("DetailFragment", "Failed to load mock_heartbeats.webp", e);
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
            // Show loading indicator
            android.app.ProgressDialog progressDialog = new android.app.ProgressDialog(requireContext());
            progressDialog.setMessage("Processing video...");
            progressDialog.setCancelable(false);
            progressDialog.show();

            // Create and run processing task
            new Thread(() -> {
                try {
                    MediaProcessingServiceImpl mediaProcessingService = new MediaProcessingServiceImpl(requireContext());
                    mediaProcessingService.createHeartBeatsVideo(currentRecordEntry.getId());

                    // Update record status after processing
                    if (currentRecordEntry != null) {
                        currentRecordEntry.setStatus(RecordEntry.Status.UNCHECKED);
                        DataRecordServiceImpl dataRecordService = new DataRecordServiceImpl(requireContext());
                        dataRecordService.saveData(currentRecordEntry);
                    }

                    // Update UI on completion
                    requireActivity().runOnUiThread(() -> {
                        progressDialog.dismiss();
                        // Refresh the status display
                        if (currentRecordEntry != null) {
                            binding.textArrhythmia.setText("Status: " + currentRecordEntry.getStatus().getValue());
                        }
                        // Show success message
                        android.widget.Toast.makeText(requireContext(),
                            "Video processing completed",
                            android.widget.Toast.LENGTH_SHORT).show();
                    });
                } catch (Exception e) {
                    requireActivity().runOnUiThread(() -> {
                        progressDialog.dismiss();
                        android.widget.Toast.makeText(requireContext(),
                            "Error processing video: " + e.getMessage(),
                            android.widget.Toast.LENGTH_LONG).show();
                    });
                }
            }).start();
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
        binding.editDuration.setVisibility(editVisibility);
        binding.btnPickDate.setVisibility(editVisibility);
        binding.editDateLabel.setVisibility(editVisibility);

        binding.textPatientName.setVisibility(textVisibility);
        binding.textNotes.setVisibility(textVisibility);
        binding.textDuration.setVisibility(textVisibility);
    }

    private void updateDisplayValues() {
        if (currentRecordEntry != null) {
            binding.textPatientName.setText("Name: " + currentRecordEntry.getPatientName());
            binding.editPatientName.setText(currentRecordEntry.getPatientName());
            binding.textNotes.setText("Notes: " + (currentRecordEntry.getNotes() != null ? currentRecordEntry.getNotes() : ""));
            binding.editNotes.setText(currentRecordEntry.getNotes() != null ? currentRecordEntry.getNotes() : "");
            binding.textDuration.setText("Duration: " + currentRecordEntry.getDuration());
            binding.editDuration.setText(String.valueOf(currentRecordEntry.getDuration()));
        }
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
