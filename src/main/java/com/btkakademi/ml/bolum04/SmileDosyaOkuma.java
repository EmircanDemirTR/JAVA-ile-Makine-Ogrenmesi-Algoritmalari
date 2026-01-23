package com.btkakademi.ml.bolum04;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

public class SmileDosyaOkuma {

  public static void main(String[] args) {
    csvDosyasiOku();
  }

  private static void csvDosyasiOku() {
    try {
      InputStream inputStream = SmileDosyaOkuma.class.getClassLoader()
          .getResourceAsStream("datasets/iris.csv");

      if (inputStream == null) {
        System.out.println("HATA: iris.csv dosyasi bulunamadi.");
        return;
      }

      Reader reader = new InputStreamReader(inputStream, StandardCharsets.UTF_8);
      CSVParser parser = CSVFormat.DEFAULT.builder().setHeader().setSkipHeaderRecord(true).build()
          .parse(reader);

      List<double[]> ozellikListesi = new ArrayList<>();
      List<String> etiketListesi = new ArrayList<>();
      List<String> basliklar = parser.getHeaderNames();

      for (CSVRecord kayit : parser) {
        double[] ozellikler = new double[4];
        ozellikler[0] = Double.parseDouble(kayit.get("sepalLength"));
        ozellikler[1] = Double.parseDouble(kayit.get("sepalWidth"));
        ozellikler[2] = Double.parseDouble(kayit.get("petalLength"));
        ozellikler[3] = Double.parseDouble(kayit.get("petalWidth"));
        ozellikListesi.add(ozellikler);

        etiketListesi.add(kayit.get("class"));
      }

      double[][] X = ozellikListesi.toArray(new double[0][]);

      Map<String, Integer> sinifHaritasi = new HashMap<>();
      sinifHaritasi.put("setosa", 0);
      sinifHaritasi.put("versicolor", 1);
      sinifHaritasi.put("virginica", 2);

      int[] y = new int[etiketListesi.size()];
      for (int i = 0; i < etiketListesi.size(); i++) {
        y[i] = sinifHaritasi.get(etiketListesi.get(i));
      }

      String[] sinifIsimleri = {"setosa", "versicolor", "virginica"};
      String[] ozellikIsimleri = {"sepalLength", "sepalWidth", "petalLength", "petalWidth"};

      veriSetiBilgileriniYazdir(X, y, ozellikIsimleri, sinifIsimleri);
      parser.close();
      reader.close();

    } catch (Exception e) {
      System.out.println("CSV okuma hatasi " + e.getMessage());
      e.printStackTrace();
    }
  }

  private static void veriSetiBilgileriniYazdir(double[][] X, int[] y, String[] ozellikIsimleri,
      String[] sinifIsimleri) {
    System.out.println("Dosya Formati: CSV");
    System.out.println("Toplam Ornek Sayisi: " + X.length);
    System.out.println("Toplam Ozellik sayisi " + X[0].length);
  }
}
