package com.btkakademi.ml.bolum08;

import smile.classification.DiscreteNaiveBayes;
import smile.validation.metric.Accuracy;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.*;

public class SmileNaiveBayes {
    private static final int BINS = 10;

    static void main() {
        try {
            var is = SmileNaiveBayes.class.getClassLoader().getResourceAsStream("datasets/breast-cancer.csv");
            var reader = new BufferedReader(new InputStreamReader(is));
            reader.readLine();

            List<double[]> ozellikler = new ArrayList<>();
            List<Integer> etiketler = new ArrayList<>();

            int malignantCount = 0, beningCount = 0;

            String satir;
            while ((satir = reader.readLine()) != null) {
                String[] p = satir.split(",");

                String sinif = p[1].trim();
                if (sinif.equals("M")) {
                    etiketler.add(1);
                    malignantCount++;
                } else {
                    etiketler.add(0);
                    beningCount++;
                }

                double[] ozellik = new double[30];
                for (int i = 0; i < 30; i++) {
                    ozellik[i] = Double.parseDouble(p[i + 2].trim());
                }
                ozellikler.add(ozellik);
            }
            reader.close();

            double[][] X = ozellikler.toArray(new double[0][]);
            int[] y = etiketler.stream().mapToInt(i -> i).toArray();

            // Train Test Split
            int n = X.length;
            int egitimBoyut = (int) (n * 0.7);

            Integer[] idx = new Integer[n];
            for (int i = 0; i < n; i++) idx[i] = i;
            Collections.shuffle(Arrays.asList(idx), new Random(42));

            // Eğitim ve tes dizileri oluşturma
            double[][] XTrainContinuos = new double[egitimBoyut][];
            double[][] XTestContinuos = new double[n - egitimBoyut][];
            int[] yTrain = new int[egitimBoyut];
            int[] yTest = new int[n - egitimBoyut];

            for (int i = 0; i < egitimBoyut; i++) {
                XTrainContinuos[i] = X[idx[i]];
                yTrain[i] = y[idx[i]];
            }

            for (int i = egitimBoyut; i < n; i++) {
                XTestContinuos[i - egitimBoyut] = X[idx[i]];
                yTest[i - egitimBoyut] = y[idx[i]];
            }

            System.out.println("Egitim: " + XTrainContinuos.length + ", Test: " + XTestContinuos.length);


            // Discretization - Sürekli Veriyi Kategorikleştirme
            // Equal - Width Binning Algoritması
            // width = (max - min) / BINS
            System.out.println("Discretization");
            System.out.println("Bin sayisi: " + BINS);

            double[] min = new double[30];
            double[] max = new double[30];

            Arrays.fill(min, Double.MAX_VALUE);
            Arrays.fill(max, Double.MIN_VALUE);

            for (double[] row : XTrainContinuos) {
                for (int j = 0; j < 30; j++) {
                    if (row[j] < min[j]) min[j] = row[j];
                    if (row[j] > max[j]) max[j] = row[j];
                }
            }

            int[][] XTrain = discretize(XTrainContinuos, min, max, BINS);
            int[][] XTest = discretize(XTestContinuos, min, max, BINS);


            // Model Oluşturma
            int k = 2; // Sınıf sayısı
            int p = 30; // Özellik sayımız

            var model = new DiscreteNaiveBayes(
                    DiscreteNaiveBayes.Model.MULTINOMIAL,
                    k,
                    p
            );

            // Online Learning
            for (int i = 0; i < XTrain.length; i++) {
                model.update(XTrain[i], yTrain[i]);
            }


            // Tahmin ve Değerlendirme

            int[] tahminler = new int[XTest.length];
            for (int i = 0; i < XTest.length; i++) {
                tahminler[i] = model.predict(XTest[i]);
            }

            double dogruluk = Accuracy.of(yTest, tahminler);
            System.out.println("Dogruluk Orani: " + dogruluk * 100);


        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static int[][] discretize(double[][] X, double[] min, double[] max, int bins) {
        int n = X.length;
        int m = X[0].length;

        int[][] XDisc = new int[n][m];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                double norm = (X[i][j] - min[j]) / (max[j] - min[j] + 1e-10);
                int bin = (int) (norm * (bins - 1));
                bin = Math.max(0, Math.min(bin, bins - 1));

                XDisc[i][j] = bin;
            }
        }
        return XDisc;
    }
}
