package com.gunadarma.heartratearrhythmiachecker.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Builder
@Data
@NoArgsConstructor
@AllArgsConstructor
public class HeartRateData {
  private double intensity;
  private long timestamp;
}
