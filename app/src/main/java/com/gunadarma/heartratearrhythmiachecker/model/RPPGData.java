package com.gunadarma.heartratearrhythmiachecker.model;

import java.util.ArrayList;
import java.util.List;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Builder
@AllArgsConstructor
@Getter
public class RPPGData {
  @Builder.Default
  private List<Long> heartbeats = new ArrayList<>();
  private double minBpm;
  private double maxBpm;
  private double averageBpm;
  private double baselineBpm; // Added baseline heart rate for rPPG
  private int durationSeconds;
  @Builder.Default
  private List<Signal> signals = new ArrayList<>();

  public static RPPGData empty() {
    return RPPGData.builder()
        .heartbeats(new ArrayList<>())
        .minBpm(0)
        .maxBpm(0)
        .averageBpm(0)
        .baselineBpm(0) // Added baseline to empty constructor
        .durationSeconds(0)
        .signals(new ArrayList<>())
        .build();
  }

  @Builder
  @NoArgsConstructor
  @AllArgsConstructor
  @Getter
  public static class Signal {
    private double redChannel;
    private double greenChannel;
    private double blueChannel;
    private long timestamp;
  }
}
