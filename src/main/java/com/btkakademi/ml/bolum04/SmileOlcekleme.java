package com.btkakademi.ml.bolum04;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class SmileOlcekleme {
    public static void main() throws Exception {
        double[][] veri = veriYukle();

        // 1. Orijinal Veri
        System.out.println("1. Orijinal Veri");
        System.out.println("Satir: " + veri.length);
        System.out.println("İlk 3 örnek: ");
        for (int i = 0; i < 3; i++) {
            System.out.printf(" [%.0f, %.0f, %.0f, %.0f]", veri[i][0], veri[i][1], veri[i][2], veri[i][3]);
        }

        // 2. Min-Max Normalizasyon
        System.out.println("2. Min-Max Normalizasyon");
        double[][] minMax = minMaxNormalize(veri);
        System.out.println("Min Max ilk 3 örnek: ");
        for (int i = 0; i < 3; i++) {
            System.out.printf(" [%.3f, %.3f, %.3f, %.3f]", minMax[i][0], minMax[i][1], minMax[i][2], minMax[i][3]);
        }

        // 3. Z-Score Standardizasyon
        System.out.println("3. Z-Score Standardizasyon");
        double[][] zScore = zScoreStandardizasyon(veri);
        System.out.println("Z-Score ilk 3 örnek: ");
        for (int i = 0; i < 3; i++) {
            System.out.printf(" [%.3f, %.3f, %.3f, %.3f]", zScore[i][0], zScore[i][1], zScore[i][2], zScore[i][3]);
        }
    }

    private static double[][] veriYukle() throws Exception {
        var is = SmileOlcekleme.class.getClassLoader().getResourceAsStream("datasets/olcekleme.csv");
        var reader = new BufferedReader(new InputStreamReader(is));
        reader.readLine();

        List<double[]> satirlar = new ArrayList<>();
        String satir;
        while ((satir = reader.readLine()) != null) {
            String[] p = satir.split(",");
            satirlar.add(new double[]{
                    Double.parseDouble(p[0]), Double.parseDouble(p[1]), Double.parseDouble(p[2]), Double.parseDouble(p[3])
            });
        }
        reader.close();
        return satirlar.toArray(new double[0][]);
    }

    private static double[][] minMaxNormalize(double[][] veri) {
        int n = veri.length;
        int m = veri[0].length;
        double[][] sonuc = new double[n][m];

        for (int j = 0; j < m; j++) {

            double min = veri[0][j], max = veri[0][j];
            for (int i = 0; i < n; i++) {
                if (veri[i][j] < min) min = veri[i][j];
                if (veri[i][j] > max) max = veri[i][j];
            }

            for (int i = 0; i < n; i++) {
                sonuc[i][j] = (veri[i][j] - min) / (max - min);
            }
        }
        return sonuc;
    }

    private static double[][] zScoreStandardizasyon(double[][] veri) {
        int n = veri.length;
        int m = veri[0].length;
        double[][] sonuc = new double[n][m];

        for (int j = 0; j < m; j++) {

            double toplam = 0;
            for (int i = 0; i < n; i++) toplam += veri[i][j];
            double ort = toplam / n;

            //Standart sapma
            double kareToplam = 0;
            for (int i = 0; i < n; i++) {
                kareToplam += Math.pow(veri[i][j] - ort, 2);
            }

            double std = Math.sqrt(kareToplam / n);

            for (int i = 0; i < n; i++) {
                sonuc[i][j] = (veri[i][j] - ort) / std;
            }

        }
        return sonuc;
    }

}
