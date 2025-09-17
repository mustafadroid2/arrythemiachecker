package com.gunadarma.heartratearrhythmiachecker.util;

import com.gunadarma.heartratearrhythmiachecker.model.RecordEntry;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class AppUtil {
  public static String toDetailDate(long timestamp) {
    if (timestamp <= 0) return "N/A";

    Date dateObject = new Date(timestamp);
    SimpleDateFormat dateFormat = new SimpleDateFormat("dd MMM yyyy, HH:mm:ss", Locale.getDefault()); // HH:mm:ss
    return dateFormat.format(dateObject);
  }

  public static String toDate(long timestamp) {
    if (timestamp <= 0) return "N/A";

    long now  = System.currentTimeMillis();
    long diff = now - timestamp;

    // Convert to appropriate units
    long seconds = diff / 1000;
    long minutes = seconds / 60;
    long hours   = minutes / 60;
    long days    = hours / 24;
    long years   = days / 365;

    if (days > 0) {
      Date dateObject = new Date(timestamp);
      SimpleDateFormat dateFormat;
      if (years < 1) {
        dateFormat = new SimpleDateFormat("dd MMM, HH:mm", Locale.getDefault()); // HH:mm
      } else {
        dateFormat = new SimpleDateFormat("dd MMM yy, HH:mm", Locale.getDefault()); // HH:mm
      }
      return dateFormat.format(dateObject);
    } else if (hours > 0) {
      return hours + (hours == 1 ? " hour ago" : " hours ago");
    } else if (minutes > 0) {
      return minutes + (minutes == 1 ? " minute ago" : " minutes ago");
    } else {
      return seconds <= 0 ? "just now" : seconds + (seconds == 1 ? " second ago" : " seconds ago");
    }
  }

  public static String patientNameOrDefault(RecordEntry record, boolean withoutId) {
    String postfix = " (#"+ record.getId() +")";
    String name = (record.getPatientName() != null && !record.getPatientName().isEmpty()) ? record.getPatientName() : "...";

    if (withoutId) {
      return name;
    }
    return name + postfix;
  }

  public static String patientNameOrDefault(RecordEntry record) {
    return patientNameOrDefault(record, false);
  }
}
