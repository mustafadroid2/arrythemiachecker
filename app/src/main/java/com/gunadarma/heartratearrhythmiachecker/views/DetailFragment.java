package com.gunadarma.heartratearrhythmiachecker.views;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.navigation.fragment.NavHostFragment;

import com.gunadarma.heartratearrhythmiachecker.R;
import com.gunadarma.heartratearrhythmiachecker.databinding.FragmentDetailBinding;
import com.gunadarma.heartratearrhythmiachecker.model.RecordEntry;
import com.gunadarma.heartratearrhythmiachecker.service.DataRecordServiceImpl;
import java.util.HashMap;
import java.util.Map;

import android.util.Log;
import com.fasterxml.jackson.databind.ObjectMapper;
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
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
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

        binding.btnEdit.setOnClickListener(v -> {
            isEditMode = true;
            binding.textPatientName.setVisibility(View.GONE);
            binding.editPatientName.setVisibility(View.VISIBLE);
            binding.textDate.setVisibility(View.GONE);
            binding.btnPickDate.setVisibility(View.VISIBLE);
            binding.editDateLabel.setVisibility(View.VISIBLE);
            binding.textNotes.setVisibility(View.GONE);
            binding.editNotes.setVisibility(View.VISIBLE);
            binding.textDuration.setVisibility(View.GONE);
            binding.editDuration.setVisibility(View.VISIBLE);
            binding.btnEdit.setVisibility(View.GONE);
            binding.btnSave.setVisibility(View.VISIBLE);
            if (currentRecordEntry != null) {
                // @+id/DetailFragment
                binding.editPatientName.setText(currentRecordEntry.getPatientName());
                String dateStr = AppUtil.toDate(currentRecordEntry.getCreateAt());
                binding.editDateLabel.setText(dateStr);
                binding.editNotes.setText(currentRecordEntry.getNotes() != null ? currentRecordEntry.getNotes() : "");
                binding.editDuration.setText(String.valueOf(currentRecordEntry.getDuration()));
            }
        });

        binding.btnPickDate.setOnClickListener(v -> {
            java.util.Calendar calendar = java.util.Calendar.getInstance();
            String currentDate = binding.editDateLabel.getText().toString();
            java.text.SimpleDateFormat sdf = new java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.getDefault());
            try {
                java.util.Date date = sdf.parse(currentDate);
                calendar.setTime(date);
            } catch (Exception ignored) {}
            new android.app.DatePickerDialog(requireContext(), (view1, year, month, dayOfMonth) -> {
                calendar.set(year, month, dayOfMonth);
                binding.editDateLabel.setText(sdf.format(calendar.getTime()));
            }, calendar.get(java.util.Calendar.YEAR), calendar.get(java.util.Calendar.MONTH), calendar.get(java.util.Calendar.DAY_OF_MONTH)).show();
        });

        binding.btnSave.setOnClickListener(v -> {
            if (currentRecordEntry != null) {
                String newName = binding.editPatientName.getText().toString();
                String newDateStr = binding.editDateLabel.getText().toString();
                String newNotes = binding.editNotes.getText().toString();
                int newDuration = 0;
                try {
                    newDuration = Integer.parseInt(binding.editDuration.getText().toString());
                } catch (Exception ignored) {}
                currentRecordEntry.setPatientName(newName);
                currentRecordEntry.setNotes(newNotes);
                currentRecordEntry.setDuration(newDuration);
                try {
                    java.text.SimpleDateFormat sdf = new java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.getDefault());
                    java.util.Date newDate = sdf.parse(newDateStr);
                    currentRecordEntry.setCreateAt(newDate.getTime());
                } catch (Exception ignored) {}
                int finalNewDuration = newDuration;
                new Thread(() -> {
                    DataRecordServiceImpl dataRecordService = new DataRecordServiceImpl(requireContext());
                    dataRecordService.saveData(currentRecordEntry);
                    requireActivity().runOnUiThread(() -> {
                        binding.textPatientName.setText("Name: " + newName);
                        binding.textDate.setText("Date: " + newDateStr);
                        binding.textNotes.setText("Notes: " + newNotes);
                        binding.textDuration.setText("Duration: " + finalNewDuration);
                        binding.textPatientName.setVisibility(View.VISIBLE);
                        binding.editPatientName.setVisibility(View.GONE);
                        binding.textDate.setVisibility(View.VISIBLE);
                        binding.btnPickDate.setVisibility(View.GONE);
                        binding.editDateLabel.setVisibility(View.GONE);
                        binding.textNotes.setVisibility(View.VISIBLE);
                        binding.editNotes.setVisibility(View.GONE);
                        binding.textDuration.setVisibility(View.VISIBLE);
                        binding.editDuration.setVisibility(View.GONE);
                        binding.btnEdit.setVisibility(View.VISIBLE);
                        binding.btnSave.setVisibility(View.GONE);
                        isEditMode = false;
                    });
                }).start();
            }
        });

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
                        binding.textId.setText("ID: " + data.getId());
                        binding.textPatientName.setText("Name: " + data.getPatientName());
                        binding.textDate.setText("Date: " + recordDate);
                        binding.editDateLabel.setText(recordDate);
                        binding.textArrhythmia.setText("Status: " + data.getStatus().getValue());
                        binding.textHeartRate.setText("Heart Rate: " + data.getBeatsPerMinute());
                        binding.editPatientName.setText(data.getPatientName());
                        binding.textNotes.setText("Notes: " + (data.getNotes() != null ? data.getNotes() : ""));
                        binding.editNotes.setText(data.getNotes() != null ? data.getNotes() : "");
                        binding.textDuration.setText("Duration: " + data.getDuration());
                        binding.editDuration.setText(String.valueOf(data.getDuration()));
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

        // Provide video playback from res/raw/mock_video.mp4
        try {
            String videoPath = "android.resource://" + requireContext().getPackageName() + "/raw/mock_video";
            binding.videoView.setVideoPath(videoPath);
            android.widget.MediaController mediaController = new android.widget.MediaController(requireContext());
            binding.videoView.setMediaController(mediaController);
            mediaController.setAnchorView(binding.videoView);
            binding.videoView.seekTo(1); // Show first frame
            binding.videoView.setOnClickListener(v -> {
                if (binding.videoView.isPlaying()) {
                    binding.videoView.pause();
                } else {
                    binding.videoView.start();
                }
            });
        } catch (Exception e) {
            Log.e("DetailFragment", "Failed to load mock video", e);
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
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
