package com.btkakademi.ml.bolum14;

import org.apache.commons.csv.CSVFormat;
import smile.classification.RandomForest;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.io.Read;
import smile.model.cart.SplitRule;
import smile.util.Index;
import smile.validation.Bag;
import smile.validation.CrossValidation;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class SmileGridSearch {
    public static void main(String[] args) {
        try {
            var is = SmileGridSearch.class.getClassLoader().getResourceAsStream("datasets/wine.csv");
            var reader = new BufferedReader(new InputStreamReader(is));
            reader.readLine();

            List<double[]> ozellikler = new ArrayList<>();
            List<Integer> etiketler = new ArrayList<>();

            String satir;

            while ((satir = reader.readLine()) != null) {
                if (satir.trim().isEmpty()) continue;
                String[] p = satir.split(",");
                etiketler.add(Integer.parseInt(p[0].trim()) - 1);
                double[] ozellik = new double[13];
                for (int j = 0; j < 13; j++) {
                    ozellik[j] = Double.parseDouble(p[j + 1].trim());
                }
                ozellikler.add(ozellik);
            }
            reader.close();

            double[][] X = ozellikler.toArray(new double[0][]);
            int[] y = etiketler.stream().mapToInt(i -> i).toArray();
            int n = X.length;

            // Normalizasyon
            X = normalize(X);

            // Tek Parametre Icin Grid Search
            double enIyiKnnAcc = 0;
            int enIyiK = 1;

            for (int k = 1; k <= 20; k++) {
                double[] accs = knnCvHesapla(X, y, n, 5, k);
                double ort = hesaplaOrtalama(accs);
                double std = hesaplaStd(accs, ort);

                if (ort > enIyiKnnAcc) {
                    enIyiKnnAcc = ort;
                    enIyiK = k;
                }

                System.out.printf("K=%d, Acc: %.4f, Std: %.4f%n", k, ort, std);
            }


            // ÇOKLU PARAMETRE İÇİN GRID SEARCH
            var url = SmileGridSearch.class.getClassLoader().getResource("datasets/wine.csv");
            var format = CSVFormat.DEFAULT.builder()
                    .setHeader().setSkipHeaderRecord(true).setIgnoreHeaderCase(true).setTrim(true).get();
            DataFrame veri = Read.csv(Path.of(url.toURI()), format);

            veri = veri.factorize("class");
            Formula formula = Formula.lhs("class");

            int[] ntreessDegerleri = {50, 100, 200};
            int[] maxDepthDegerleri = {5, 10, 15, 20};

            double enIyiRfAcc = 0;
            int enIyiNtrees = 50;
            int enIyiMaxDepth = 5;
            int toplamKombi = 0;

            for (int ntrees : ntreessDegerleri) {
                for (int maxDepth : maxDepthDegerleri) {
                    toplamKombi++;
                    Bag[] rfFoldlar = CrossValidation.of(n, 5);
                    double toplamRfAcc = 0;

                    for (int f = 0; f < 5; f++) {
                        int[] rfTrainIdx = rfFoldlar[f].samples();
                        int[] rfTestIdx = rfFoldlar[f].oob();

                        DataFrame trainDf = veri.get(Index.of(rfTrainIdx));
                        DataFrame testDf = veri.get(Index.of(rfTestIdx));

                        var rfOptions = new RandomForest.Options(
                                ntrees,
                                0,
                                SplitRule.GINI,
                                maxDepth,
                                0,
                                5,
                                1.0,
                                null,
                                null,
                                null
                        );
                        var rf = RandomForest.fit(formula, trainDf, rfOptions);

                        int[] gercek = formula.y(testDf).toIntArray();
                        int dogru = 0;
                        for (int i = 0; i < testDf.nrow(); i++) {
                            if (rf.predict(testDf.get(i)) == gercek[i]) {
                                dogru++;
                            }
                        }
                        toplamRfAcc += (double) dogru / testDf.nrow();
                    }
                    double cvAcc = toplamRfAcc / 5.0;

                    if (cvAcc > enIyiRfAcc) {
                        enIyiRfAcc = cvAcc;
                        enIyiNtrees = ntrees;
                        enIyiMaxDepth = maxDepth;
                    }
                    System.out.printf("Ntrees: %d, MaxDepth: %d, CV Acc: %.4f%n", ntrees, maxDepth, cvAcc);
                }
                System.out.println();
            }

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static double[][] normalize(double[][] X) {
        int n = X.length;
        int m = X[0].length;
        double[] min = new double[m];
        double[] max = new double[m];

        for (int j = 0; j < m; j++) {
            min[j] = Double.MAX_VALUE;
            max[j] = Double.MIN_VALUE;
            for (int i = 0; i < n; i++) {
                if (X[i][j] < min[j]) min[j] = X[i][j];
                if (X[i][j] > max[j]) max[j] = X[i][j];
            }
        }

        double[][] X_norm = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                X_norm[i][j] = (X[i][j] - min[j]) / (max[j] - min[j] + 1e-10); // avoid division by zero
            }
        }

        return X_norm;
    }

    private static double[] knnCvHesapla(double[][] X, int[] y, int n, int foldSayisi, int k) {
        Bag[] foldlar = CrossValidation.of(n, foldSayisi);
        double[] accs = new double[foldSayisi];

        for (int f = 0; f < foldSayisi; f++) {
            int[] trainIndex = foldlar[f].samples();
            int[] testIndex = foldlar[f].oob();

            double[][] X_train = new double[trainIndex.length][];
            int[] y_train = new int[trainIndex.length];
            for (int i = 0; i < trainIndex.length; i++) {
                X_train[i] = X[trainIndex[i]];
                y_train[i] = y[trainIndex[i]];
            }

            double[][] X_test = new double[testIndex.length][];
            int[] y_test = new int[testIndex.length];
            for (int i = 0; i < testIndex.length; i++) {
                X_test[i] = X[testIndex[i]];
                y_test[i] = y[testIndex[i]];
            }

            var knn = smile.classification.KNN.fit(X_train, y_train, k);
            int correct = 0;
            for (int i = 0; i < X_test.length; i++) {
                int pred = knn.predict(X_test[i]);
                if (pred == y_test[i]) correct++;
            }
            accs[f] = (double) correct / X_test.length;
        }
        return accs;
    }

    private static double hesaplaOrtalama(double[] values) {
        double sum = 0;
        for (double v : values) {
            sum += v;
        }
        return sum / values.length;
    }

    private static double hesaplaStd(double[] values, double mean) {
        double sum = 0;
        for (double v : values) {
            sum += (v - mean) * (v - mean);
        }
        return Math.sqrt(sum / values.length);
    }
}