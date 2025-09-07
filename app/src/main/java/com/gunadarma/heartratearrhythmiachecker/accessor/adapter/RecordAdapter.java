package com.gunadarma.heartratearrhythmiachecker.accessor.adapter;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.navigation.fragment.NavHostFragment;
import androidx.recyclerview.widget.RecyclerView;

import com.gunadarma.heartratearrhythmiachecker.R;
import com.gunadarma.heartratearrhythmiachecker.model.RecordEntry;
import com.gunadarma.heartratearrhythmiachecker.util.AppUtil;

import java.util.List;

public class RecordAdapter extends RecyclerView.Adapter<RecordAdapter.RecordViewHolder> {
    private final Fragment parent;
    private final List<RecordEntry> records;

    public RecordAdapter(List<RecordEntry> records, Fragment parent) {
        this.records = records;
        this.parent = parent;
    }

    @NonNull
    @Override
    public RecordViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_record, parent, false); // Use the custom layout
        return new RecordViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull RecordViewHolder holder, int position) {
        RecordEntry record = records.get(position);
        String recordDate = AppUtil.toDate(record.getCreateAt());

        // Bind data to the views
        holder.recordId.setText(String.format("#%s", record.getId()));
        holder.patientName.setText(AppUtil.patientNameOrDefault(record, true));
        holder.status.setText(record.getStatus().getValue());
        holder.date.setText(recordDate);

        // Navigate to DetailFragment on item click
        holder.itemView.setOnClickListener(v -> {
            // Create a Bundle to pass the record ID
            Bundle bundle = new Bundle();
            bundle.putString("id", String.valueOf(record.getId())); // Pass the record ID

            // Navigate to DetailFragment with the bundle
            NavHostFragment
                    .findNavController(parent)
                    .navigate(R.id.action_HomeFragment_to_DetailFragment, bundle);
        });
    }

    @Override
    public int getItemCount() {
        return records != null ? records.size() : 0;
    }

    static class RecordViewHolder extends RecyclerView.ViewHolder {
        TextView recordId;
        TextView patientName;
        TextView status;
        TextView date;

        public RecordViewHolder(@NonNull View itemView) {
            super(itemView);
            recordId = itemView.findViewById(R.id.text_record_id);
            patientName = itemView.findViewById(R.id.text_patient_name);
            status = itemView.findViewById(R.id.text_status);
            date = itemView.findViewById(R.id.text_record_date);
        }
    }
}
