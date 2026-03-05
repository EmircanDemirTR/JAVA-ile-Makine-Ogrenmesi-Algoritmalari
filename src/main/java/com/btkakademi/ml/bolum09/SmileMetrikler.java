package com.btkakademi.ml.bolum09;

import smile.classification.LogisticRegression;
import smile.validation.metric.Accuracy;
import smile.validation.metric.ConfusionMatrix;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.*;

public class SmileMetrikler {
    static void main() {
        try {
            var is = SmileMetrikler.class.getClassLoader().getResourceAsStream("datasets/glass.csv");
            var reader = new BufferedReader(new InputStreamReader(is));
            String header = reader.readLine();

            List<double[]> ozellikler = new ArrayList<>();
            List<Integer> etiketler = new ArrayList<>();

            Map<Integer, Integer> sinifDagilimi = new TreeMap<>();

            String satir;
            while ((satir = reader.readLine()) != null) {
                String[] parcalar = satir.split(",");

                double[] ozellik = new double[9];
                for (int i = 0; i < 9; i++) {
                    ozellik[i] = Double.parseDouble(parcalar[i]);
                }
                ozellikler.add(ozellik);

                int sinif = Integer.parseInt(parcalar[9].trim());
                int sinifIndex = sinifToIndex(sinif);
                etiketler.add(sinifIndex);
                sinifDagilimi.merge(sinif, 1, Integer::sum);
            }
            reader.close();

            // List to Array

            double[][] X = ozellikler.toArray(new double[0][]);
            int[] y = etiketler.stream().mapToInt(i -> i).toArray();

            // Normalizasyon
            double[][] XNorm = normalize(X);

            // Train - Test Split
            int n = X.length;
            int egitimBoyut = (int) (n * 0.7);

            int[] idx = new int[n];
            for (int i = 0; i < n; i++) idx[i] = i;

            Random rand = new Random(42);
            for (int i = n - 1; i > 0; i--) {
                int j = rand.nextInt(i + 1);
                int temp = idx[i];
                idx[i] = idx[j];
                idx[j] = temp;
            }

            double[][] XTrain = new double[egitimBoyut][];
            double[][] XTest = new double[n - egitimBoyut][];
            int[] yTrain = new int[egitimBoyut];
            int[] yTest = new int[n - egitimBoyut];

            for (int i = 0; i < egitimBoyut; i++) {
                XTrain[i] = XNorm[idx[i]];
                yTrain[i] = y[idx[i]];
            }

            for (int i = egitimBoyut; i < n; i++) {
                XTest[i - egitimBoyut] = XNorm[idx[i]];
                yTest[i - egitimBoyut] = y[idx[i]];
            }

            System.out.println("Egitim: " + XTrain.length);
            System.out.println("Test: " + XTest.length);

            // Model Egitimi

            var model = LogisticRegression.fit(XTrain, yTrain);

            // Tahmin
            int[] tahminler = model.predict(XTest);

            // Smile Accuracy
            double accuracy = Accuracy.of(yTest, tahminler);
            System.out.println("Doğruluk Orani: " + accuracy * 100);

            // Karmaşıklık Matrisi
            var cm = ConfusionMatrix.of(yTest, tahminler);
            System.out.println("Satir=Gercek, Sutun=Tahmin");
            System.out.println(cm);

            // Precision ve Recall Hesaplama
            int sinifSayisi = 0;
            for (int yi : yTest) {
                if (yi > sinifSayisi) sinifSayisi = yi;
            }
            sinifSayisi++;

            int[][] matrix = new int[sinifSayisi][sinifSayisi];
            for (int i = 0; i < yTest.length; i++) {
                int gercek = yTest[i];
                int tahmin = tahminler[i];
                matrix[gercek][tahmin]++;
            }

            System.out.println("SINIF BAZLI METRİKLER");
            double totalPrecision = 0, totalRecall = 0, totalF1 = 0;
            int totalSupport = 0;
            double weightedPrecision = 0, weightedRecall = 0, weightedF1 = 0;

            for (int sinif = 0; sinif < sinifSayisi; sinif++) {
                int tp = matrix[sinif][sinif];

                int fp = 0;
                for (int i = 0; i < sinifSayisi; i++) {
                    if (i != sinif) fp += matrix[i][sinif];
                }

                int fn = 0;
                for (int j = 0; j < sinifSayisi; j++) {
                    if (j != sinif) fn += matrix[sinif][j];
                }

                int support = tp + fn;

                double precision = (tp + fp) > 0 ? (double) tp / (tp + fp) : 0;
                double recall = (tp + fn) > 0 ? (double) tp / (tp + fn) : 0;
                double f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0;

                System.out.printf("%d\t%.4f\t\t%.4f\t\t%.4f\t%d\n",
                        sinif, precision, recall, f1, support);
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static int sinifToIndex(int sinif) {
        return switch (sinif) {
            case 1 -> 0;
            case 2 -> 1;
            case 3 -> 2;
            case 5 -> 3;
            case 6 -> 4;
            case 7 -> 5;
            default -> -1;
        };
    }

    private static double[][] normalize(double[][] X) {
        int n = X.length; // Örnek Sayısı
        int m = X[0].length; // Özellik Sayısı

        double[] min = new double[m];
        double[] max = new double[m];
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
