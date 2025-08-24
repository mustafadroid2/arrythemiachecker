package com.gunadarma.heartratearrhythmiachecker.model;

import androidx.room.ColumnInfo;
import androidx.room.Entity;
import androidx.room.Ignore;
import androidx.room.PrimaryKey;
import java.util.List;
import lombok.Builder;
import lombok.Data;
import lombok.Getter;
import lombok.NoArgsConstructor;


@Entity(tableName = "records")
@Builder
@Data
@NoArgsConstructor
public class RecordEntry {
    @PrimaryKey(autoGenerate = true)
    private long id;
    private String patientName;
    private String notes;        // additional info if any
    private long createAt;
    private long updatedAt;

    private Status status;       // test result
    private int duration;        // data capture duration
    private int beatsPerMinute;  // average heart rate
    @ColumnInfo(name = "heartbeats")
    private List<Long> heartbeats; // list of timestamps for each heartbeat

    @Ignore
    @Builder
    public RecordEntry(Long id, String patientName, Long createAt, Status status, String notes, int duration, int beatsPerMinute) {
        this.id = id;
        this.patientName = patientName;
        this.createAt = createAt;
        this.status = status;
        this.notes = notes;
        this.duration = duration;
        this.beatsPerMinute = beatsPerMinute;
    }

    // Room will use this constructor
    public RecordEntry(long id, String patientName, String notes, long createAt, long updatedAt,
                      Status status, int duration, int beatsPerMinute, List<Long> heartbeats) {
        this.id = id;
        this.patientName = patientName;
        this.notes = notes;
        this.createAt = createAt;
        this.updatedAt = updatedAt;
        this.status = status;
        this.duration = duration;
        this.beatsPerMinute = beatsPerMinute;
        this.heartbeats = heartbeats;
    }

    @Getter
    public enum Status {
        UNCHECKED("Unchecked"),
        NORMAL("Normal"),
        ARRHYTHMIA_TACHYCARDIA("Tachycardia"),
        ARRHYTHMIA_BRADYCARDIA("Bradycardia"),
        ARRHYTHMIA_IRREGULAR("Irregular"),
        ARRHYTHMIA_OTHER("Other");

        private final String value;

        private Status(String value) {
            this.value = value;
        }

        public String getValue() {
            return value;
        }
    }

    public int getBeatsPerMinute() {
        return beatsPerMinute;
    }

    public void setBeatsPerMinute(int beatsPerMinute) {
        this.beatsPerMinute = beatsPerMinute;
    }
}
