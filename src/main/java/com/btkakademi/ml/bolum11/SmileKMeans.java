package com.btkakademi.ml.bolum11;

import smile.clustering.KMeans;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SmileKMeans {
    static void main() {
        try {
            var is = SmileKMeans.class.getClassLoader().getResourceAsStream("datasets/mall_customers.csv");
            var reader = new BufferedReader(new InputStreamReader(is));
            reader.readLine();

            List<double[]> ozellikler = new ArrayList<>();
            String satir;

            while ((satir = reader.readLine()) != null) {
                String[] parcalar = satir.split(",");
                ozellikler.add(new double[]{
                        Double.parseDouble(parcalar[2].trim()),
                        Double.parseDouble(parcalar[3].trim()),
                        Double.parseDouble(parcalar[4].trim()),
                });
            }

            reader.close();

            double[][] X = ozellikler.toArray(new double[0][]);

            // Min - Max Normalizasyonu
            double[][] XNorm = normalize(X);

            // K-Means Kümeleme
            int k = 5;
            int maxIter = 100;

            var model = KMeans.fit(XNorm, k, maxIter);
            int[] atamalar = model.group();// her örneğin hamgi kümeye atandığını döndürür

            System.out.println("Toplam İç Küme Mesafe: " + String.format("%.4f", model.distortion()));

            int[] kumeBoyutlari = new int[k];
            for (int atama : atamalar) {
                kumeBoyutlari[atama]++;
            }
            System.out.println("Kume Boyutlari");
            for (int i = 0; i < k; i++) {
                System.out.printf(" Küme %d: %d müşteri (%%%.1f)\n", i, kumeBoyutlari[i], kumeBoyutlari[i] * 100.0 / X.length);
            }

            // Küme Merkezleri Analizi
            double[][] merkezler = new double[k][];
            for (int i = 0; i < 5; i++) {
                merkezler[i] = model.center(i);
            }

            System.out.println("Küme Merkezleri (Normalize)");
            System.out.printf(" %-8s   %-8s   %-8s  %-8s\n", "Küme", "Age", "Income", "Score");
            for (int i = 0; i < k; i++) {
                System.out.printf(" Küme: %d:   %8.4f  %8.4f  %8.4f\n", i, merkezler[i][0], merkezler[i][1], merkezler[i][2]);
            }

            // Elbow Method
            System.out.println("Elbow Method");
            System.out.printf("%-5s %15s\n", "K", "Distortion");
            for (int ki = 2; ki <= 10; ki++) {
                var elbowModel = KMeans.fit(XNorm, ki, 100);
                System.out.printf("K=%-3d %15.4f\n", ki, elbowModel.distortion());
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
}
