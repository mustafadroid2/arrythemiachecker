package com.gunadarma.heartratearrhythmiachecker.views;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.navigation.fragment.NavHostFragment;
import androidx.recyclerview.widget.LinearLayoutManager;
import com.gunadarma.heartratearrhythmiachecker.R;
import com.gunadarma.heartratearrhythmiachecker.databinding.FragmentHomeBinding;
import com.gunadarma.heartratearrhythmiachecker.service.DataRecordServiceImpl;
import com.gunadarma.heartratearrhythmiachecker.accessor.adapter.RecordAdapter;

import android.os.Environment;
import android.util.Log;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class HomeFragment extends Fragment {

    private FragmentHomeBinding binding;

    @Override
    public View onCreateView(
            @NonNull LayoutInflater inflater, ViewGroup container,
            Bundle savedInstanceState
    ) {
        binding = FragmentHomeBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        // Show brief description
        binding.textviewFirst.setText("Heart Rate Arrhythmia Checker helps you record, analyze, and detect arrhythmia from heart rate videos.");


        // Set up RecyclerView
        binding.recyclerView.setLayoutManager(new LinearLayoutManager(getContext()));
        binding.progressBar.setVisibility(View.VISIBLE); // Show loading
        binding.recyclerView.setVisibility(View.GONE);

        // Fetch data asynchronously
        new Thread(() -> {
            DataRecordServiceImpl dataRecordService = new DataRecordServiceImpl(requireContext());
            final java.util.List<com.gunadarma.heartratearrhythmiachecker.model.RecordEntry> records = dataRecordService.listRecords();
            requireActivity().runOnUiThread(() -> {
                RecordAdapter adapter = new RecordAdapter(records, this);
                binding.recyclerView.setAdapter(adapter);
                binding.progressBar.setVisibility(View.GONE); // Hide loading
                binding.recyclerView.setVisibility(View.VISIBLE);
                if (records == null || records.isEmpty()) {
                    binding.textviewFirst.setText("No data");
                    binding.recyclerView.setVisibility(View.GONE);
                } else {
                    binding.textviewFirst.setText("Heart Rate Arrhythmia Checker helps you record, analyze, and detect arrhythmia from heart rate videos.");
                    binding.recyclerView.setVisibility(View.VISIBLE);
                }
            });
        }).start();

        // Floating action button to add new record
        binding.fab.setOnClickListener(v -> {
            NavHostFragment.findNavController(HomeFragment.this)
                .navigate(R.id.action_HomeFragment_to_RecordFragment);
        });

        // Export CSV floating action button
        binding.fabExportCsv.setOnClickListener(v -> {
            // Show confirmation dialog before exporting
            new androidx.appcompat.app.AlertDialog.Builder(requireContext())
                    .setTitle("Export Data")
                    .setMessage("Export all heart rate data to CSV file in Downloads folder?")
                    .setPositiveButton("Yes", (dialog, which) -> {
                        exportDataToCsv();
                    })
                    .setNegativeButton("No", null)
                    .show();
        });

        // How to Use floating action button
        binding.fabHowToUse.setOnClickListener(v -> {
            binding.howToUseTray.setVisibility(View.VISIBLE);
        });

        // Close tray button
        binding.closeTrayButton.setOnClickListener(v -> {
            binding.howToUseTray.setVisibility(View.GONE);
        });

        // Seed button for demo (add to layout if not present)
        binding.seedButton.setOnClickListener(v -> {
            new Thread(() -> {
                DataRecordServiceImpl dataRecordService = new DataRecordServiceImpl(requireContext());
                dataRecordService.seedDemoData();
                requireActivity().runOnUiThread(() -> {
                    com.google.android.material.snackbar.Snackbar.make(binding.getRoot(), "Seeded demo data!", com.google.android.material.snackbar.Snackbar.LENGTH_SHORT).show();
                    // Refresh list after seeding
                    binding.progressBar.setVisibility(View.VISIBLE);
                    binding.recyclerView.setVisibility(View.GONE);
                    new Thread(() -> {
                        final java.util.List<com.gunadarma.heartratearrhythmiachecker.model.RecordEntry> records = dataRecordService.listRecords();
                        requireActivity().runOnUiThread(() -> {
                            RecordAdapter adapter = new RecordAdapter(records, this);
                            binding.recyclerView.setAdapter(adapter);
                            binding.progressBar.setVisibility(View.GONE);
                            if (records == null || records.isEmpty()) {
                                binding.textviewFirst.setText("No data");
                                binding.recyclerView.setVisibility(View.GONE);
                            } else {
                                binding.textviewFirst.setText("Heart Rate Arrhythmia Checker helps you record, analyze, and detect arrhythmia from heart rate videos.");
                                binding.recyclerView.setVisibility(View.VISIBLE);
                            }
                        });
                    }).start();
                });
            }).start();
        });
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }

    private void exportDataToCsv() {
        // Show loading state
        binding.progressBar.setVisibility(View.VISIBLE);

        new Thread(() -> {
            try {
                // Fetch all records from database
                DataRecordServiceImpl dataRecordService = new DataRecordServiceImpl(requireContext());
                List<com.gunadarma.heartratearrhythmiachecker.model.RecordEntry> records = dataRecordService.listRecords();

                if (records == null || records.isEmpty()) {
                    requireActivity().runOnUiThread(() -> {
                        binding.progressBar.setVisibility(View.GONE);
                        com.google.android.material.snackbar.Snackbar.make(binding.getRoot(),
                            "No data to export",
                            com.google.android.material.snackbar.Snackbar.LENGTH_SHORT).show();
                    });
                    return;
                }

                // Generate CSV content
                String csvContent = generateCsvContent(records);

                // Create filename with timestamp
                SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss", Locale.getDefault());
                String timestamp = dateFormat.format(new Date());
                String filename = "HeartRateData_" + timestamp + ".csv";

                // Save to Downloads folder
                File downloadsDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS);
                File csvFile = new File(downloadsDir, filename);

                // Write CSV content to file
                try (FileWriter writer = new FileWriter(csvFile)) {
                    writer.write(csvContent);
                    writer.flush();
                }

                // Show success message on UI thread
                requireActivity().runOnUiThread(() -> {
                    binding.progressBar.setVisibility(View.GONE);
                    com.google.android.material.snackbar.Snackbar.make(binding.getRoot(),
                        "CSV exported to Downloads: " + filename,
                        com.google.android.material.snackbar.Snackbar.LENGTH_LONG).show();
                });

                Log.i("CSV_Export", "Successfully exported " + records.size() + " records to: " + csvFile.getAbsolutePath());

            } catch (IOException e) {
                Log.e("CSV_Export", "Error writing CSV file", e);
                requireActivity().runOnUiThread(() -> {
                    binding.progressBar.setVisibility(View.GONE);
                    com.google.android.material.snackbar.Snackbar.make(binding.getRoot(),
                        "Error exporting CSV: " + e.getMessage(),
                        com.google.android.material.snackbar.Snackbar.LENGTH_LONG).show();
                });
            } catch (Exception e) {
                Log.e("CSV_Export", "Unexpected error during CSV export", e);
                requireActivity().runOnUiThread(() -> {
                    binding.progressBar.setVisibility(View.GONE);
                    com.google.android.material.snackbar.Snackbar.make(binding.getRoot(),
                        "Export failed: " + e.getMessage(),
                        com.google.android.material.snackbar.Snackbar.LENGTH_LONG).show();
                });
            }
        }).start();
    }

    private String generateCsvContent(List<com.gunadarma.heartratearrhythmiachecker.model.RecordEntry> records) {
        StringBuilder csvBuilder = new StringBuilder();
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault());

        // CSV Header
        csvBuilder.append("ID,Date,Patient Name,Age,Gender,Address,")
                  .append("Status,Duration (seconds),Heart Rate (BPM),Notes\n");

        // CSV Data rows
        for (com.gunadarma.heartratearrhythmiachecker.model.RecordEntry record : records) {
            csvBuilder.append(escapeCsvValue(String.valueOf(record.getId()))).append(",");
            csvBuilder.append(escapeCsvValue(dateFormat.format(new Date(record.getCreateAt())))).append(",");
            csvBuilder.append(escapeCsvValue(record.getPatientName() != null ? record.getPatientName() : "unnamed")).append(",");
            csvBuilder.append(escapeCsvValue(record.getAge() != null ? record.getAge().toString() : "")).append(",");
            csvBuilder.append(escapeCsvValue(record.getGender() != null ? record.getGender() : "")).append(",");
            csvBuilder.append(escapeCsvValue(record.getAddress() != null ? record.getAddress() : "-")).append(",");
            csvBuilder.append(escapeCsvValue(record.getStatus() != null ? record.getStatus().getValue() : "")).append(",");
            csvBuilder.append(escapeCsvValue(String.valueOf(record.getDuration()))).append(",");
            csvBuilder.append(escapeCsvValue(String.valueOf(record.getBeatsPerMinute()))).append(",");
            csvBuilder.append(escapeCsvValue(record.getNotes() != null ? record.getNotes() : "")).append("");

            csvBuilder.append("\n");
        }

        return csvBuilder.toString();
    }

    private String escapeCsvValue(String value) {
        if (value == null) {
            return "";
        }

        // Escape double quotes by doubling them and wrap in quotes if contains comma, quote, or newline
        if (value.contains(",") || value.contains("\"") || value.contains("\n") || value.contains("\r")) {
            return "\"" + value.replace("\"", "\"\"") + "\"";
        }

        return value;
    }
}