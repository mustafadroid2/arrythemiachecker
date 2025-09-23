package com.gunadarma.heartratearrhythmiachecker.service;

import com.gunadarma.heartratearrhythmiachecker.model.RecordEntry;

public interface MainMediaProcessingService {

    /**
     * Interface for progress callbacks during video processing
     */
    interface ProgressCallback {
        void onProgressUpdate(int currentFrame, int totalFrames, String phase);
        void onPhaseChanged(String phase);
    }

    void createHeartBeatsVideo(RecordEntry recordEntry);

    /**
     * Process video with progress callbacks
     */
    void createHeartBeatsVideo(RecordEntry recordEntry, ProgressCallback progressCallback);
}
