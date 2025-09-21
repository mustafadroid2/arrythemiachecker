package com.gunadarma.heartratearrhythmiachecker.service;

import com.gunadarma.heartratearrhythmiachecker.model.RecordEntry;

public interface MainMediaProcessingService {
  void createHeartBeatsVideo(RecordEntry recordEntry);
}
