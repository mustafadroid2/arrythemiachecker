package com.gunadarma.heartratearrhythmiachecker.accessor;

import android.content.Context;
import androidx.room.Database;
import androidx.room.Room;
import androidx.room.RoomDatabase;
import androidx.room.TypeConverters;

import com.gunadarma.heartratearrhythmiachecker.model.RecordEntry;
import com.gunadarma.heartratearrhythmiachecker.util.Converters;

@Database(
    entities = {RecordEntry.class},
    version = 3, // incremented from 1 to 2 due to schema change
    exportSchema = false
)
@TypeConverters({Converters.class})
public abstract class RecordDatabase extends RoomDatabase {
    public abstract RecordEntryDao recordEntryDao();

    private static volatile RecordDatabase INSTANCE;

    public static RecordDatabase getInstance(Context context) {
        if (INSTANCE == null) {
            synchronized (RecordDatabase.class) {
                if (INSTANCE == null) {
                    INSTANCE = Room.databaseBuilder(context.getApplicationContext(),
                            RecordDatabase.class, "record_database")
                            .fallbackToDestructiveMigration()
                            .build();
                }
            }
        }
        return INSTANCE;
    }
}