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
}