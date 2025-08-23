package com.gunadarma.heartratearrhythmiachecker.util;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class AppUtil {
  public static String toDate(long timestamp) {
    if (timestamp <= 0) return "N/A";

    Date dateObject = new Date(timestamp);
    SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd", Locale.getDefault());
    return dateFormat.format(dateObject);
  }
}
