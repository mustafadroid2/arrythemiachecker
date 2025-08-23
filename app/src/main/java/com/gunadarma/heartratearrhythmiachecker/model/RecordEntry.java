package com.gunadarma.heartratearrhythmiachecker.model;

import androidx.room.Entity;
import androidx.room.PrimaryKey;

import java.util.ArrayList;
import java.util.List;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.Getter;


@Entity(tableName = "records")
@Builder
@Data
@AllArgsConstructor
public class RecordEntry {
    @PrimaryKey(autoGenerate = true)
    private long id;
    private String patientName;
    private String notes;        // additional info if any
    private long createAt;
    private long updatedAt;

    private Status status;       // test result
    private int duration;        // data capture duration
    private int beatsPerMinute;  // total heart beats per minute => heartbeats.count/60
    @Builder.Default
    private List<Long> heartbeats          = new ArrayList<>(); // heartbeats timestamps in milliseconds
    @Builder.Default
    private List<Double> heartbeatsQuality = new ArrayList<>(); // (optional) confidence level of detecting heartbeats

    @Getter
    public enum Status {
        UNCHECKED("Unchecked"),
        NORMAL("Normal"),
        ARRHYTHMIA_TACHYCARDIA("Tachycardia"),
        ARRHYTHMIA_BRADYCARDIA("Bradycardia"),
        ARRHYTHMIA_OTHER("Other");

        private final String value;

        private Status(String value) {
            this.value = value;
        }

        public String getValue() {
            return value;
        }
    }
}
