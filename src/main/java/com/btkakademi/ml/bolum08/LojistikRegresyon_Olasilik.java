package com.btkakademi.ml.bolum08;

import smile.classification.LogisticRegression;
import smile.validation.metric.Accuracy;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

public class LojistikRegresyon_Olasilik {
    static void main() {
        try {
            var is = LojistikRegresyon_Olasilik.class.getClassLoader().getResourceAsStream("datasets/iris.csv");

            var reader = new BufferedReader(new InputStreamReader(is));
            reader.readLine();

            List<double[]> ozellikler = new ArrayList<>();
            List<Integer> etiketler = new ArrayList<>();

            Map<String, Integer> etiketMap = Map.of(
                    "setosa", 0,
                    "versicolor", 1,
                    "virginica", 2
            );

            String[] siniflar = {"setosa", "versicolor", "virginica"};

            String satir;
            while ((satir = reader.readLine()) != null) {
                String[] p = satir.split(",");
                ozellikler.add(new double[]{
                        Double.parseDouble(p[0]),
                        Double.parseDouble(p[1]),
                        Double.parseDouble(p[2]),
                        Double.parseDouble(p[3]),
                });
                etiketler.add(etiketMap.get(p[4].trim()));
            }
            reader.close();

            double[][] X = ozellikler.toArray(new double[0][]);
            int[] y = etiketler.stream().mapToInt(i -> i).toArray();

            // Train Test Split
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
                XTrain[i] = X[idx[i]];
                yTrain[i] = y[idx[i]];
            }

            for (int i = egitimBoyut; i < n; i++) {
                XTest[i - egitimBoyut] = X[idx[i]];
                yTest[i - egitimBoyut] = y[idx[i]];
            }

            System.out.println("Egitim " + XTrain.length + ", Test: " + XTest.length);


            // Model Egitimi
            var model = LogisticRegression.fit(XTrain, yTrain);

            // Hard Classification
            int[] hardTahminler = model.predict(XTest);
            double hardAcc = Accuracy.of(yTest, hardTahminler) * 100;
            System.out.println("Hard Classification Dogruluk: " + hardAcc);

            // Soft Classification
            System.out.println("Soft Classification Ornekleri: ");
            System.out.println("Gercek      Tahmin    P(setosa) P(versic) P(virgin.)  Guven  Sonuc");
            System.out.println("----------  --------   --------   -------   -------    -----   ------");

            for (int i = 0; i < Math.min(10, XTest.length); i++) {
                double[] probs = new double[3];
                int hardTahmin = model.predict(XTest[i], probs);

                double maxProb = Arrays.stream(probs).max().orElse(0);

                String guven = maxProb > 0.9 ? "YUKSEK" : maxProb > 0.7 ? "ORTA " : "DUSUK";

                String gercek = siniflar[yTest[i]];
                String tahmin = siniflar[hardTahmin];
                String dogru = gercek.equals(tahmin) ? "OK" : "YANLIS";

                System.out.printf("%-10s  %-10s %.3f      %.3f     %s  %s%n",
                        gercek, tahmin, probs[0], probs[1], probs[2], guven, dogru);
            }

            // Threshold Ayarlama
            double[] thresholds = {0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
            System.out.println("Threshold Precision Recall   F-1 Score");
            System.out.println("--------  ------    -----    --------");

            for (double threshold : thresholds) {
                int truePositive = 0;
                int falsePositive = 0;
                int falseNegative = 0;
                int trueNegative = 0;

                for (int i = 0; i < XTest.length; i++) {
                    double[] probs = new double[3];
                    model.predict(XTest[i], probs);

                    boolean tahminVirginica = probs[2] >= threshold;
                    boolean gercekVirginica = yTest[i] == 2;

                    if (gercekVirginica && tahminVirginica) truePositive++;
                    else if (!gercekVirginica && tahminVirginica) falsePositive++;
                    else if (gercekVirginica && !tahminVirginica) falseNegative++;
                    else trueNegative++;
                }

                double precision = (truePositive + falsePositive) > 0 ?
                        (double) truePositive / (truePositive + falsePositive) : 0;

                double recall = (truePositive + falsePositive) > 0 ?
                        (double) truePositive / (truePositive + falseNegative) : 0;

                double f1 = (precision + recall) > 0 ?
                        2 * precision * recall / (precision + recall) : 0;

                System.out.printf("%.1f        %.3f     %.3f     %.3f%n",
                        threshold, precision, recall, f1);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
