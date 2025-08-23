package com.gunadarma.heartratearrhythmiachecker.service;

import android.content.Context;

import com.gunadarma.heartratearrhythmiachecker.database.RecordDatabase;
import com.gunadarma.heartratearrhythmiachecker.database.RecordEntryDao;
import com.gunadarma.heartratearrhythmiachecker.model.RecordEntry;

import java.util.List;
import java.util.Random;

public class DataRecordServiceImpl {

    private final Random random = new Random(); // Initialize Random
    private final RecordEntry.Status[] allStatuses = RecordEntry.Status.values();
    private final RecordEntryDao recordEntryDao;

    public DataRecordServiceImpl(Context context) {
        RecordDatabase db = RecordDatabase.getInstance(context);
        this.recordEntryDao = db.recordEntryDao();
    }

    public void seedDemoData() {
        var current = recordEntryDao.listRecords();
        int additionalRecords = 2; // Number of records to add
        int lastId = current != null ? current.size() + 1 : 0;
        String[] patientNames = {"John", "Johnny", "Marco", "Phoenix", "Tarma", "Andrea", "Sammy", "Sasaki", "Henry", "Almira"};

        for (int i = lastId; i < lastId+additionalRecords; i++) {
            RecordEntry.Status randomStatus = allStatuses[random.nextInt(allStatuses.length)];
            String patientName = patientNames[i % patientNames.length];
            RecordEntry entry = RecordEntry.builder()
                    .patientName(patientName)
                    .createAt(System.currentTimeMillis() - (random.nextInt(10) * 86400000L))
                    .status(randomStatus)
                    .build();
            recordEntryDao.insert(entry);
        }
    }

    public RecordEntry get(String id) {
        return recordEntryDao.get(id);
    }

    public List<RecordEntry> listRecords() {
        return recordEntryDao.listRecords();
    }

    public RecordEntry saveData(RecordEntry record) {
        recordEntryDao.insert(record);
        // implement retuning the saved record
        // please remember that the id is auto-generated
        return recordEntryDao.get(String.valueOf(record.getId()));
    }

    public void removeData(RecordEntry record) {
        recordEntryDao.delete(record);
    }

    public long getNextId() {
        List<RecordEntry> current = recordEntryDao.listRecords();
        int lastIndex = current != null ? current.size() - 1 : -1;
        return current != null && lastIndex != -1 ? current.get(lastIndex).getId() + 1  : 0;
    }
}
