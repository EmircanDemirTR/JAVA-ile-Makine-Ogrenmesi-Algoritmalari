package com.btkakademi.ml.bolum04;

import java.io.InputStream;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;

public class WekaDosyaOkuma {

  public static void main(String[] args) {
    arffDosyasiOku();
    System.out.println("\n----------------------\n");
    csvDosyasiOku();
  }

  private static void arffDosyasiOku() {
    try {
      InputStream inputStream = WekaDosyaOkuma.class.getClassLoader()
          .getResourceAsStream("datasets/iris.arff");

      if (inputStream == null) {
        System.out.println("HATA: iris.arff dosyasi bulunamadi.");
        return;
      }

      ArffLoader loader = new ArffLoader();
      loader.setSource(inputStream);

      Instances veriSeti = loader.getDataSet();

      veriSeti.setClassIndex(veriSeti.numAttributes() - 1); // Sınıf özelliğini belirleme

      veriSetiBilgileriniYazdir(veriSeti, "ARFF");
    } catch (Exception e) {
      System.out.println("ARFF okuma hatasi " + e.getMessage());
      e.printStackTrace();
    }
  }

  private static void csvDosyasiOku() {
    try {
      InputStream inputStream = WekaDosyaOkuma.class.getClassLoader()
          .getResourceAsStream("datasets/iris.csv");

      if (inputStream == null) {
        System.out.println("HATA: iris.csv dosyasi bulunamadi.");
        return;
      }

      CSVLoader loader = new CSVLoader();
      loader.setSource(inputStream);

      Instances veriSeti = loader.getDataSet();

      veriSeti.setClassIndex(veriSeti.numAttributes() - 1); // Sınıf özelliğini belirleme

      veriSetiBilgileriniYazdir(veriSeti, "CSV");
    } catch (Exception e) {
      System.out.println("CSV okuma hatasi " + e.getMessage());
      e.printStackTrace();
    }
  }

  private static void veriSetiBilgileriniYazdir(Instances veriSeti, String format) {
    System.out.println("Dosya Formati: " + format);
    System.out.println("Veri Seti Adi: " + veriSeti.relationName());
    System.out.println("Toplam Ornek Sayisi: " + veriSeti.numInstances());
    System.out.println("Toplam Ozellik Sayisi: " + veriSeti.numAttributes());
    System.out.println("Sinif Ozelligi: " + veriSeti.classAttribute().name());

    // Ozellik Bilgileri
    System.out.println("Ozellikler");
    for (int i = 0; i < veriSeti.numAttributes(); i++) {
      Attribute attr = veriSeti.attribute(i);
      String tip = attr.isNumeric() ? "Sayisal" : "Kategorik";
      System.out.println(" " + i + ". " + attr.name() + " (" + tip + ")");
    }
    System.out.println();

    // Sinif Dagilimi
    System.out.println("Sinif Dagilimi");
    Attribute sinifAttr = veriSeti.classAttribute();
    for (int i = 0; i < sinifAttr.numValues(); i++) {
      String sinifAdi = sinifAttr.value(i);
      int sayi = 0;
      for (int j = 0; j < veriSeti.numInstances(); j++) {
        if (veriSeti.instance(j).stringValue(sinifAttr).equals(sinifAdi)) {
          sayi++;
        }
      }
      System.out.println(" " + sinifAdi + " " + sayi + " ornek");
    }
    System.out.println();
  }
}
