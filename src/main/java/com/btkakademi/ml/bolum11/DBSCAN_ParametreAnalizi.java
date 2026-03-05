package com.btkakademi.ml.bolum11;

import smile.clustering.DBSCAN;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class DBSCAN_ParametreAnalizi {
    static void main() {
        try {
            var is = DBSCAN_ParametreAnalizi.class.getClassLoader().getResourceAsStream("datasets/glass.csv");
            var reader = new BufferedReader(new InputStreamReader(is));

            String header = reader.readLine();
            String[] sutunlar = header.split(",");

            List<double[]> ozellikler = new ArrayList<>();
            List<Integer> etiketler = new ArrayList<>();

            String satir;
            while ((satir = reader.readLine()) != null) {
                String[] p = satir.split(",");
                double[] ozellik = new double[p.length - 1];
                for (int i = 0; i < p.length - 1; i++) {
                    ozellik[i] = Double.parseDouble(p[i].trim());
                }
                ozellikler.add(ozellik);
                etiketler.add(Integer.parseInt(p[p.length - 1].trim()));
            }
            reader.close();

            double[][] X = ozellikler.toArray(new double[0][]);
            int[] gercekEtiketler = etiketler.stream().mapToInt(i -> i).toArray();

            // Normalizasyon
            double[][] XNorm = normalize(X);

            // K - DISTANCE HESAPLAMA
            int kDistance = 5;
            double[] mesafeler = hesaplaKDistance(XNorm, kDistance);
            Arrays.sort(mesafeler);

            for (int i = 0; i < mesafeler.length / 2; i++) {
                double temp = mesafeler[i];
                mesafeler[i] = mesafeler[mesafeler.length - 1 - i];
                mesafeler[mesafeler.length - 1 - i] = temp;
            }

            System.out.println("En büyük 10 K-Distance");
            for (int i = 0; i < 10 && i < mesafeler.length; i++) {
                System.out.printf(" %d. -> %.4f\n", i + 1, mesafeler[i]);
            }

            System.out.println("En küçük 10 K-Distance");
            for (int i = Math.max(0, mesafeler.length - 10); i < mesafeler.length; i++) {
                System.out.printf(" %d. -> %.4f\n", i + 1, mesafeler[i]);
            }

            double ortMesafe = Arrays.stream(mesafeler).average().orElse(0);
            double medyan = mesafeler[mesafeler.length / 2];

            System.out.println("Ortalama K-Distance: " + ortMesafe);
            System.out.println("Medyan K-Distance: " + medyan);

            // Önerilen Eps: medyanın yarısı ile 1.5 katı arasında denenebilir
            // Önerilen epsilon aralığı: medyan * 0.5 ile medyan * 1.5 arası


            // Epsilon Etkisi Analizi
            double[] epsDegerleri = {0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0};
            int sabitMinPts = 5;

            for (double eps : epsDegerleri) {
                var model = DBSCAN.fit(XNorm, sabitMinPts, eps);
                int[] atamalar = model.group();
                int kumeSayisi = model.k();

                int noiseSayisi = 0;
                int[] kumeBoyutlari = new int[kumeSayisi];
                for (int atama : atamalar) {
                    if (atama == Integer.MAX_VALUE) noiseSayisi++;
                    else {
                        kumeBoyutlari[atama]++;
                    }
                }

                int enBuyukKume = kumeBoyutlari.length > 0 ?
                        Arrays.stream(kumeBoyutlari).max().orElse(0) : 0;

                System.out.println("Epsilon sayisi: " + eps + ", Kume Sayisi: " + kumeSayisi + ", Noise: " + noiseSayisi + " En büyük: " + enBuyukKume);


            }

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static double[][] normalize(double[][] X) {
        int n = X.length, m = X[0].length;

        double[] min = new double[m], max = new double[m];
        Arrays.fill(min, Double.MAX_VALUE);
        Arrays.fill(max, Double.MIN_VALUE);

        for (double[] row : X) {
            for (int j = 0; j < m; j++) {
                if (row[j] < min[j]) min[j] = row[j];
                if (row[j] > max[j]) max[j] = row[j];
            }
        }

        double[][] XNorm = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                XNorm[i][j] = (X[i][j] - min[j]) / (max[j] - min[j] + 1e-10);
            }
        }

        return XNorm;
    }

    private static double[] hesaplaKDistance(double[][] X, int k) {
        int n = X.length;
        double[] kDistances = new double[n];
        for (int i = 0; i < n; i++) {
            double[] mesafeler = new double[n];
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    mesafeler[j] = Double.MAX_VALUE;
                } else {
                    mesafeler[j] = oklitMesafe(X[i], X[j]);
                }
            }

            Arrays.sort(mesafeler);
            kDistances[i] = mesafeler[k - 1];
        }
        return kDistances;
    }

    private static double oklitMesafe(double[] a, double[] b) {
        double toplam = 0;
        for (int i = 0; i < a.length; i++) {
            double fark = a[i] - b[i];
            toplam += fark * fark;
        }
        return Math.sqrt(toplam);
    }
}
