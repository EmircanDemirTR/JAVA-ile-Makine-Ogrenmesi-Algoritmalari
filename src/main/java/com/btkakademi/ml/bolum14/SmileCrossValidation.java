package com.btkakademi.ml.bolum14;

import smile.classification.KNN;
import smile.validation.Bag;
import smile.validation.CrossValidation;
import smile.validation.metric.Accuracy;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.*;

public class SmileCrossValidation {
    static void main() {
        try {
            var is = SmileCrossValidation.class.getClassLoader().getResourceAsStream("datasets/breast-cancer.csv");
            var reader = new BufferedReader(new InputStreamReader(is));
            reader.readLine();

            List<double[]> ozellikler = new ArrayList<>();
            List<Integer> etiketler = new ArrayList<>();

            String satir;
            while ((satir = reader.readLine()) != null) {
                String[] parcalar = satir.split(",");
                double[] ozellik = new double[30];
                for (int j = 0; j < 30; j++) {
                    ozellik[j] = Double.parseDouble(parcalar[j + 2].trim());
                }
                ozellikler.add(ozellik);
                etiketler.add(parcalar[1].trim().equals("M") ? 1 : 0);
            }
            reader.close();

            double[][] X = ozellikler.toArray(new double[0][]);
            int[] y = etiketler.stream().mapToInt(i -> i).toArray();
            int n = X.length;

            // Normalizasyon
            int m = X[0].length;
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
                    XNorm[i][j] = (X[i][j] - min[j]) / (max[j] - min[j]);
                }
            }

            // HOLD-OUT SORUNLARI
            double toplamAcc = 0;
            double[] accDizi = new double[5];

            for (int s = 0; s < 5; s++) {
                int seed = s * 10 + 1;

                Integer[] idx = new Integer[n];
                for (int i = 0; i < n; i++) {
                    idx[i] = i;
                }

                Collections.shuffle(Arrays.asList(idx), new Random(seed));

                int egitimBoyut = (int) (0.7 * n);
                double[][] XTrain = new double[egitimBoyut][];
                int[] yTrain = new int[egitimBoyut];
                double[][] XTest = new double[n - egitimBoyut][];
                int[] yTest = new int[n - egitimBoyut];

                for (int i = 0; i < egitimBoyut; i++) {
                    XTrain[i] = XNorm[idx[i]];
                    yTrain[i] = y[idx[i]];
                }

                for (int i = egitimBoyut; i < n; i++) {
                    XTest[i - egitimBoyut] = XNorm[idx[i]];
                    yTest[i - egitimBoyut] = y[idx[i]];
                }

                var model = KNN.fit(XTrain, yTrain, 5);
                int[] tahminler = new int[XTest.length];
                for (int i = 0; i < XTest.length; i++) {
                    tahminler[i] = model.predict(XTest[i]);
                }
                double acc = Accuracy.of(yTest, tahminler);

                accDizi[s] = acc;
                toplamAcc += acc;
                System.out.println("Seed: " + seed + " Doğruluk: " + acc);
            }

            double ortAcc = toplamAcc / 5;
            double stdAcc = hesaplaStd(accDizi, ortAcc);
            System.out.println("Ortalama Doğruluk: " + ortAcc);
            System.out.println("Doğruluk Std Sapma: " + stdAcc);


            // MANUEL K-FOLD CV + METRİKLER
            int kFold = 10;
            Bag[] foldlar = CrossValidation.of(n, kFold);
            System.out.println("Fold Sayisi: " + foldlar.length);

            double[] foldDogruluk = new double[kFold];
            double[] foldPrecision = new double[kFold];
            double[] foldRecall = new double[kFold];
            double[] foldF1 = new double[kFold];

            for (int f = 0; f < kFold; f++) {
                int[] trainIdx = foldlar[f].samples();
                int[] testIdx = foldlar[f].oob();
                double[][] XTrain = new double[trainIdx.length][];
                int[] yTrain = new int[trainIdx.length];
                for (int i = 0; i < trainIdx.length; i++) {
                    XTrain[i] = XNorm[trainIdx[i]];
                    yTrain[i] = y[trainIdx[i]];
                }

                double[][] XTest = new double[testIdx.length][];
                int[] yTest = new int[testIdx.length];
                for (int i = 0; i < testIdx.length; i++) {
                    XTest[i] = XNorm[testIdx[i]];
                    yTest[i] = y[testIdx[i]];
                }

                var model = KNN.fit(XTrain, yTrain, 5);
                int[] tahminler = new int[XTest.length];
                for (int i = 0; i < XTest.length; i++) {
                    tahminler[i] = model.predict(XTest[i]);
                }
                foldDogruluk[f] = Accuracy.of(yTest, tahminler);

                // TP FP FN Hesaplama
                int tp = 0, fp = 0, fn = 0;
                for (int i = 0; i < yTest.length; i++) {
                    if (yTest[i] == 1 && tahminler[i] == 1) tp++;
                    if (yTest[i] == 0 && tahminler[i] == 1) fp++;
                    if (yTest[i] == 1 && tahminler[i] == 0) fn++;
                }

                // Precision ve Recall
                foldPrecision[f] = tp + fp == 0 ? 0 : (double) tp / (tp + fp);
                foldRecall[f] = tp + fn == 0 ? 0 : (double) tp / (tp + fn);
                foldF1[f] = (foldPrecision[f] + foldRecall[f]) == 0 ? 0 : 2 * foldPrecision[f] * foldRecall[f] / (foldPrecision[f] + foldRecall[f]);

                System.out.println("Fold " + (f + 1) + " Doğruluk: " + foldDogruluk[f] +
                        " Precision: " + foldPrecision[f] +
                        " Recall: " + foldRecall[f] +
                        " F1-Score: " + foldF1[f]);
            }


        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static double hesaplaStd(double[] dizi, double ortalama) {
        double toplam = 0;
        for (double v : dizi) {
            toplam += Math.pow(v - ortalama, 2);
        }
        return Math.sqrt(toplam / dizi.length);
    }
}
