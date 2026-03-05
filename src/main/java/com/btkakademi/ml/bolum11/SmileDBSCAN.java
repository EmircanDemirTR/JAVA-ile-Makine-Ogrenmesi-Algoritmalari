package com.btkakademi.ml.bolum11;

import smile.clustering.DBSCAN;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SmileDBSCAN {
    static void main() {
        try {
            var is = SmileDBSCAN.class.getClassLoader().getResourceAsStream("datasets/glass.csv");
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

            // DBSCAN KÜMELEME
            double eps = 0.5; // Komşuluk yarıçapı
            int minPts = 5; // Minimum komşu sayısı

            var model = DBSCAN.fit(XNorm, minPts, eps);

            int[] atamalar = model.group();
            int kumeSayisi = model.k();
            System.out.println("Bulunan Küme Sayisi: " + kumeSayisi);

            // Noise Analizi
            int noiseSayisi = 0;
            for (int atama : atamalar) {
                if (atama == Integer.MAX_VALUE) noiseSayisi++;
            }
            System.out.println("Gürültü (Noise) Noktası: " + noiseSayisi);

            // Her kümedeki eleman sayısını hesapla
            for (int c = 0; c < kumeSayisi; c++) {
                int boyut = 0;
                for (int atama : atamalar) {
                    if (c == atama) boyut++;
                }
                System.out.println("Kume " + c + ": Örnek Sayısı " + boyut);
            }
            System.out.println("Gürültü (Noise) Noktası: " + noiseSayisi);

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
