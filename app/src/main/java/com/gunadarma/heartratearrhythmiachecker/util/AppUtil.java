package com.gunadarma.heartratearrhythmiachecker.util;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class AppUtil {
  public static String toDate(long timestamp) {
    if (timestamp <= 0) return "N/A";

    long now = System.currentTimeMillis();
    long diff = now - timestamp;

    // Convert to appropriate units
    long seconds = diff / 1000;
    long minutes = seconds / 60;
    long hours = minutes / 60;
    long days = hours / 24;

    if (days > 0) {
      Date dateObject = new Date(timestamp);
      SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd hh:mm", Locale.getDefault());
      return dateFormat.format(dateObject);
    } else if (hours > 0) {
      return hours + (hours == 1 ? " hour ago" : " hours ago");
    } else if (minutes > 0) {
      return minutes + (minutes == 1 ? " minute ago" : " minutes ago");
    } else {
      return seconds <= 0 ? "just now" : seconds + (seconds == 1 ? " second ago" : " seconds ago");
    }
  }
}
