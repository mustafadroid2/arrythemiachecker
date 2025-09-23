package com.gunadarma.heartratearrhythmiachecker.service.rppg;

import com.gunadarma.heartratearrhythmiachecker.model.RPPGData;
import com.gunadarma.heartratearrhythmiachecker.service.MainMediaProcessingService;

public interface RPPGService {
  RPPGData getRPPGSignals(String videoPath, MainMediaProcessingService.ProgressCallback progressCallback);
}
