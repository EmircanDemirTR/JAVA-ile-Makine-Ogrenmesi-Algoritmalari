package com.btkakademi.ml.bolum04;

import java.util.Arrays;

public class SmileVeriYapisi {

  public static void main(String[] args) {

    // ============================================================
    // ADIM 1: VERI YAPISI - DOUBLE[][] VE INT[]
    // ============================================================

    System.out.println("--- SMILE 5.x Veri Yapisi ---\n");

    // Ozellik matrisi (Features): Her satir bir ornek, her sutun bir ozellik
    // Musteri verileri: [yas, maas]
    double[][] ozellikler = {
        {25, 5000},     // Musteri 1: 25 yas, 5000 TL
        {35, 8000},     // Musteri 2: 35 yas, 8000 TL
        {22, 3000},     // Musteri 3: 22 yas, 3000 TL
        {45, 12000},    // Musteri 4: 45 yas, 12000 TL
        {30, 4500}      // Musteri 5: 30 yas, 4500 TL
    };

    // Etiket dizisi (Labels): Her ornegin sinifi
    // 0 = Satin Almadi, 1 = Satin Aldi
    int[] etiketler = {1, 1, 0, 1, 0};

    // Ozellik ve sinif isimleri (dokumantasyon icin)
    String[] ozellikIsimleri = {"yas", "maas"};
    String[] sinifIsimleri = {"Hayir", "Evet"};

    // ============================================================
    // ADIM 2: VERI SETI BILGILERI
    // ============================================================

    System.out.println("--- Veri Seti Bilgileri ---");
    System.out.println("Ornek sayisi: " + ozellikler.length);
    System.out.println("Ozellik sayisi: " + ozellikler[0].length);
    System.out.println("Ozellikler: " + String.join(", ", ozellikIsimleri));
    System.out.println("Siniflar: " + String.join(", ", sinifIsimleri));
    System.out.println();

    // ============================================================
    // ADIM 3: VERILERI GORME
    // ============================================================

    System.out.println("--- Veri Seti Icerigi ---");
    System.out.println("Yas\tMaas\tSatin Alma");
    System.out.println("---\t----\t----------");

    for (int i = 0; i < ozellikler.length; i++) {
      double yas = ozellikler[i][0];
      double maas = ozellikler[i][1];
      String sinif = sinifIsimleri[etiketler[i]];
      System.out.printf("%.0f\t%.0f\t%s%n", yas, maas, sinif);
    }
    System.out.println();

    // ============================================================
    // ADIM 4: TEK BIR ORNEGE ERISIM
    // ============================================================

    System.out.println("--- Tek Bir Ornege Erisim ---");

    int index = 0;  // Ilk musteri
    System.out.println("Musteri " + (index + 1) + ":");
    System.out.println("  - Yas: " + ozellikler[index][0]);
    System.out.println("  - Maas: " + ozellikler[index][1]);
    System.out.println("  - Satin Alma: " + sinifIsimleri[etiketler[index]]);
    System.out.println();

    // ============================================================
    // ADIM 5: BASIT ISTATISTIKLER
    // ============================================================

    System.out.println("--- Basit Istatistikler ---");

    // Yas istatistikleri (sutun 0)
    double[] yaslar = new double[ozellikler.length];
    for (int i = 0; i < ozellikler.length; i++) {
      yaslar[i] = ozellikler[i][0];
    }
    System.out.printf("Yas - Min: %.0f, Max: %.0f, Ort: %.1f%n",
        Arrays.stream(yaslar).min().getAsDouble(),
        Arrays.stream(yaslar).max().getAsDouble(),
        Arrays.stream(yaslar).average().getAsDouble());

    // Maas istatistikleri (sutun 1)
    double[] maaslar = new double[ozellikler.length];
    for (int i = 0; i < ozellikler.length; i++) {
      maaslar[i] = ozellikler[i][1];
    }
    System.out.printf("Maas - Min: %.0f, Max: %.0f, Ort: %.1f%n",
        Arrays.stream(maaslar).min().getAsDouble(),
        Arrays.stream(maaslar).max().getAsDouble(),
        Arrays.stream(maaslar).average().getAsDouble());
    System.out.println();

    // ============================================================
    // ADIM 6: SINIF DAGILIMI
    // ============================================================

    System.out.println("--- Sinif Dagilimi ---");

    long satinAlmayanSayisi = Arrays.stream(etiketler).filter(e -> e == 0).count();
    long satinAlanSayisi = Arrays.stream(etiketler).filter(e -> e == 1).count();

    System.out.println("Satin Almayan (0): " + satinAlmayanSayisi);
    System.out.println("Satin Alan (1): " + satinAlanSayisi);
    System.out.println();

  }
}