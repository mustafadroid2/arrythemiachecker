package com.gunadarma.heartratearrhythmiachecker.service.rppg;

import com.gunadarma.heartratearrhythmiachecker.model.RPPGData;

public interface RPPGService {
  RPPGData getRPPGSignals(String videoPath);
}
