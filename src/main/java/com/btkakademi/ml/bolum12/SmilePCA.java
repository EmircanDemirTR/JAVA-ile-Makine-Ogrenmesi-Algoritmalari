package com.btkakademi.ml.bolum12;

import smile.classification.LogisticRegression;
import smile.feature.extraction.PCA;
import smile.validation.metric.Accuracy;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.*;

public class SmilePCA {
    static void main() {
        try {
            var is = SmilePCA.class.getClassLoader().getResourceAsStream("datasets/breast-cancer.csv");
            var reader = new BufferedReader(new InputStreamReader(is));
            reader.readLine();

            List<double[]> ozellikler = new ArrayList<>();
            List<Integer> etiketler = new ArrayList<>();

            String satir;
            while ((satir = reader.readLine()) != null) {
                String[] p = satir.split(",");
                etiketler.add(p[1].trim().equals("M") ? 1 : 0);
                double[] ozellik = new double[30];
                for (int i = 0; i < 30; i++) {
                    ozellik[i] = Double.parseDouble(p[i + 2].trim());
                }
                ozellikler.add(ozellik);
            }
            reader.close();

            double[][] X = ozellikler.toArray(new double[0][]);
            int[] y = etiketler.stream().mapToInt(i -> i).toArray();

            // Keşifsel Analiz
            double[][] XStd = zScoreNormalize(X);
            var pca = PCA.fit(XStd);

            // Varyans Analizi
            var varyansVec = pca.varianceProportion(); // Her bileşenin bireysel varyans orani
            double[] varyansOrani = varyansVec.toArray(new double[varyansVec.size()]);
            var kumulatifVec = pca.cumulativeVarianceProportion();
            double[] kumulatifVaryans = kumulatifVec.toArray(new double[kumulatifVec.size()]);

            System.out.println("Açıklanan Varyans Oranlari: ");
            for (int i = 0; i < Math.min(10, varyansOrani.length); i++) {
                System.out.println("PC" + (i + 1) + ": Varyans Orani: " + varyansOrani[i] * 100 + ", Kümülatif Varyans: " + kumulatifVaryans[i] * 100);
            }

            // %95 eşik yaygın bir seçimdir, bilginin %95'ini koruyarak boyut indirgenir.
            int bilesen95 = 0;
            for (int i = 0; i < kumulatifVaryans.length; i++) {
                if (kumulatifVaryans[i] >= 0.95) {
                    bilesen95 = i + 1;
                    break;
                }
            }
            System.out.println("%95 varyans için gereken bileşen: " + bilesen95);

            // Farklı Bileşen Sayılarıyla Projeksiyon
            System.out.println("Projeksiyon");
            var bilesenSet = new java.util.LinkedHashSet<>(List.of(2, 5, 10, bilesen95));
            int[] bilesenSayilari = bilesenSet.stream().mapToInt(i -> i).toArray();

            for (int p : bilesenSayilari) {
                var pcaP = pca.getProjection(p);
                double[][] XProj = pcaP.apply(XStd);
                System.out.println("PC=" + p + ": " + X[0].length + " -> " + XProj[0].length + "boyut");
            }

            // Sınıflandırma Karşılaştırması (Data Leakage Önleme)

            int n = X.length;
            int egitimBoyut = (int) (n * 0.7);
            Integer[] idx = new Integer[n];
            for (int i = 0; i < n; i++) idx[i] = i;
            Collections.shuffle(Arrays.asList(idx), new Random(42));

            double[][] XTrainHam = new double[egitimBoyut][];
            double[][] XTestHam = new double[n - egitimBoyut][];
            int[] yTrain = new int[egitimBoyut];
            int[] yTest = new int[n - egitimBoyut];

            for (int i = 0; i < egitimBoyut; i++) {
                XTrainHam[i] = X[idx[i]];
                yTrain[i] = y[idx[i]];
            }

            for (int i = egitimBoyut; i < n; i++) {
                XTestHam[i - egitimBoyut] = X[idx[i]];
                yTest[i - egitimBoyut] = y[idx[i]];
            }

            double[] egitimOrt = hesaplaOrtalama(XTrainHam);
            double[] egitimStd = hesaplaStdSapma(XTrainHam, egitimOrt);
            double[][] XTrainStd = standardizeEt(XTrainHam, egitimOrt, egitimStd);
            double[][] XTestStd = standardizeEt(XTestHam, egitimOrt, egitimStd);

            var pcaTrain = PCA.fit(XTrainStd);

            var modelOrijinal = LogisticRegression.fit(XTrainStd, yTrain);
            int[] tahminOrijinal = modelOrijinal.predict(XTestStd);
            double accOrijinal = Accuracy.of(yTest, tahminOrijinal);
            System.out.println("Orijinal (30 boyut): " + accOrijinal * 100);

            // PCA ile farklı bileşen sayıları
            for (int p : bilesenSayilari) {
                var pcaP = pcaTrain.getProjection(p);
                double[][] XTrainP = pcaP.apply(XTrainStd);
                double[][] XTestP = pcaP.apply(XTestStd);

                var modelP = LogisticRegression.fit(XTrainP, yTrain);
                int[] tahminP = modelP.predict(XTestP);
                double accP = Accuracy.of(yTest, tahminP);
                System.out.println("PCA=" + p + " Doğruluk: " + accP * 100);

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

    private static double[] hesaplaOrtalama(double[][] X) {
        int n = X.length, m = X[0].length;
        double[] ort = new double[m];
        for (double[] row : X) {
            for (int j = 0; j < m; j++) ort[j] += row[j];
        }
        for (int j = 0; j < m; j++) ort[j] /= n;
        return ort;
    }

    private static double[] hesaplaStdSapma(double[][] X, double[] ortalama) {
        int n = X.length, m = X[0].length;
        double[] std = new double[m];
        for (double[] row : X) {
            for (int j = 0; j < m; j++) {
                double fark = row[j] - ortalama[j];
                std[j] += fark * fark;
            }
        }
        for (int j = 0; j < m; j++) std[j] = Math.sqrt(std[j] / n);
        return std;
    }

    private static double[][] standardizeEt(double[][] X, double[] ortalama, double[] stdSapma) {
        int n = X.length, m = X[0].length;
        double[][] XStd = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                XStd[i][j] = (X[i][j] - ortalama[j]) / (stdSapma[j] + 1e-10);
            }
        }
        return XStd;
    }
}
