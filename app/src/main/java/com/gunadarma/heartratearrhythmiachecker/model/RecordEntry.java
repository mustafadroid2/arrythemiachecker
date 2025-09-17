package com.gunadarma.heartratearrhythmiachecker.model;

import androidx.room.ColumnInfo;
import androidx.room.Entity;
import androidx.room.PrimaryKey;
import java.util.List;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.Getter;
import lombok.NoArgsConstructor;


@Entity(tableName = "records")
@Builder
@Data
@NoArgsConstructor
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
    private int beatsPerMinute;  // average heart rate
    @ColumnInfo(name = "heartbeats")
    private List<Long> heartbeats; // list of timestamps for each heartbeat

    private Integer age; // patient's age, nullable
    private String gender; // patient's address, nullable, default empty string
    private String address; // patient's address, nullable, default empty string

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

    public String getAddress() {
        return address == null ? "" : address;
    }
    public void setAddress(String address) {
        this.address = address == null ? "" : address;
    }
}
