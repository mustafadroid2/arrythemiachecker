package com.gunadarma.heartratearrhythmiachecker.util;

import androidx.room.TypeConverter;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class Converters {
  @TypeConverter
  public static List<Long> fromStringToLongList(String value) {
    if (value == null || value.isEmpty()) return null;
    return Arrays.stream(value.split(","))
        .map(Long::parseLong)
        .collect(Collectors.toList());
  }

  @TypeConverter
  public static String fromLongListToString(List<Long> list) {
    if (list == null || list.isEmpty()) return "";
    return list.stream()
        .map(String::valueOf)
        .collect(Collectors.joining(","));
  }

  @TypeConverter
  public static List<Double> fromStringToDoubleList(String value) {
    if (value == null || value.isEmpty()) return null;
    return Arrays.stream(value.split(","))
        .map(Double::parseDouble)
        .collect(Collectors.toList());
  }

  @TypeConverter
  public static String fromDoubleListToString(List<Double> list) {
    if (list == null || list.isEmpty()) return "";
    return list.stream()
        .map(String::valueOf)
        .collect(Collectors.joining(","));
  }
}
