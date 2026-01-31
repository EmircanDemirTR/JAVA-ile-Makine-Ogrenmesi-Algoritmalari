package com.btkakademi.ml.bolum12;

import smile.feature.extraction.PCA;
import smile.manifold.TSNE;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class SmileTSNE {
    static void main() {
        try {
            var is = SmileTSNE.class.getClassLoader().getResourceAsStream("datasets/optdigits.csv");
            var reader = new BufferedReader(new InputStreamReader(is));
            reader.readLine();

            List<double[]> ozellikler = new ArrayList<>();
            List<Integer> etiketler = new ArrayList<>();

            String satir;
            while ((satir = reader.readLine()) != null) {
                if (satir.trim().isEmpty()) continue;
                String[] p = satir.split(",");
                double[] ozellik = new double[64];
                for (int i = 0; i < 64; i++) {
                    ozellik[i] = Double.parseDouble(p[i].trim());
                }
                ozellikler.add(ozellik);
                etiketler.add(Integer.parseInt(p[64].trim()));
            }
            reader.close();

            double[][] X = ozellikler.toArray(new double[0][]);
            int[] y = etiketler.stream().mapToInt(i -> i).toArray();

            // Standardizasyon
            double[][] XStd = zScoreNormalize(X);

            var pca = PCA.fit(XStd);
            var pca30 = pca.getProjection(30);
            double[][] XPca30 = pca30.apply(XStd);

            // Alt Küme Seçimi
            int altKumeBoyut = Math.min(2000, X.length);
            double[][] XAlt = new double[altKumeBoyut][];
            int[] yAlt = new int[altKumeBoyut];

            int[] altKumeIdx = new int[altKumeBoyut];

            int perDigit = altKumeBoyut / 10;
            int[] sayaclar = new int[10];
            int idx = 0;
            for (int i = 0; i < X.length && idx < altKumeBoyut; i++) {
                if (sayaclar[y[i]] < perDigit) {
                    XAlt[idx] = XPca30[i];
                    yAlt[idx] = y[i];
                    altKumeIdx[idx] = i;
                    sayaclar[y[i]]++;
                    idx++;
                }
            }

            for (int i = 0; i < X.length && idx < altKumeBoyut; i++) {
                XAlt[idx] = XPca30[i];
                yAlt[idx] = y[i];
                altKumeIdx[idx] = i;
                idx++;
            }
            System.out.println("Alt kume: " + altKumeBoyut + " ornek");


            // t-SNE Uygulaması
            var options = new TSNE.Options(2, 30.0, 200.0, 12.0, 1000);
            var tsne = TSNE.fit(XAlt, options);

            // Optimize edilmiş 2D Koordinatlar
            double[][] coords = tsne.coordinates();
            System.out.println("KL divergance: " + String.format("%.4f", tsne.cost()));


            //Rakam Koordinat Araliklari
            System.out.println("\nRakam koordinat araliklari:");
            for (int d = 0; d < 10; d++) {
                double xMin = Double.MAX_VALUE, xMax = -Double.MAX_VALUE;
                double yMin = Double.MAX_VALUE, yMax = -Double.MAX_VALUE;
                for (int i = 0; i < altKumeBoyut; i++) {
                    if (yAlt[i] == d) {
                        if (coords[i][0] < xMin) xMin = coords[i][0];
                        if (coords[i][0] > xMax) xMax = coords[i][0];
                        if (coords[i][1] < yMin) yMin = coords[i][1];
                        if (coords[i][1] > yMax) yMax = coords[i][1];
                    }
                }
                System.out.println("  Rakam " + d + ": X[" + String.format("%.1f", xMin)
                        + ", " + String.format("%.1f", xMax) + "] Y[" + String.format("%.1f", yMin)
                        + ", " + String.format("%.1f", yMax) + "]");
            }


            // Perplexity Etkisi - Etkili Komşu Sayısı

            double[] perpDegerleri = {5, 15, 30, 50};
            for (double perp : perpDegerleri) {
                var perpOptions = new TSNE.Options(2, perp, 200.0, 12.0, 500);
                var perpTsne = TSNE.fit(XAlt, perpOptions);
                System.out.println("Perplexity: " + (int) perp + " için Cost: " + perpTsne.cost());
            }


        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static double[][] zScoreNormalize(double[][] X) {
        int n = X.length, m = X[0].length;
        double[] mean = new double[m], std = new double[m];
        for (double[] row : X) {
            for (int j = 0; j < m; j++) mean[j] += row[j];
        }
        for (int j = 0; j < m; j++) mean[j] /= n;
        for (double[] row : X) {
            for (int j = 0; j < m; j++) {
                double fark = row[j] - mean[j];
                std[j] += fark * fark;
            }
        }

        for (int j = 0; j < m; j++) std[j] = Math.sqrt(std[j] / n);
        double[][] XStd = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                XStd[i][j] = (X[i][j] - mean[j]) / (std[j] + 1e-10);
            }
        }
        return XStd;
    }

}
