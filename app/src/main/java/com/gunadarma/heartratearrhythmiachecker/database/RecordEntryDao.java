package com.gunadarma.heartratearrhythmiachecker.database;

import androidx.room.Dao;
import androidx.room.Delete;
import androidx.room.Insert;
import androidx.room.OnConflictStrategy;
import androidx.room.Query;
import com.gunadarma.heartratearrhythmiachecker.model.RecordEntry;
import java.util.List;

@Dao
public interface RecordEntryDao {
    @Query("SELECT * FROM records WHERE id = :id LIMIT 1")
    RecordEntry get(String id);

    @Query("SELECT * FROM records")
    List<RecordEntry> listRecords();

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    void insert(RecordEntry record);

    @Delete
    void delete(RecordEntry record);
}