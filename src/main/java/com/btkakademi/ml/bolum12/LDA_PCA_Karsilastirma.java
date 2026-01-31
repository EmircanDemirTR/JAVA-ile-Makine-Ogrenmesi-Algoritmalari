package com.btkakademi.ml.bolum12;

import smile.classification.FLD;
import smile.classification.KNN;
import smile.feature.extraction.PCA;
import smile.validation.metric.Accuracy;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.*;

public class LDA_PCA_Karsilastirma {

    static void main() {
        try {
            var is = LDA_PCA_Karsilastirma.class.getClassLoader().getResourceAsStream("datasets/iris.csv");
            var reader = new BufferedReader(new InputStreamReader(is));
            reader.readLine();

            List<double[]> ozellikler = new ArrayList<>();
            List<Integer> etiketler = new ArrayList<>();
            Map<String, Integer> etiketMap = Map.of("setosa", 0, "versicolor", 1, "virginica", 2);

            String satir;
            while ((satir = reader.readLine()) != null) {
                String[] p = satir.split(",");
                ozellikler.add(new double[]{
                        Double.parseDouble(p[0]), Double.parseDouble(p[1]),
                        Double.parseDouble(p[2]), Double.parseDouble(p[3]),
                });
                etiketler.add(etiketMap.get(p[4].trim()));
            }
            reader.close();

            double[][] X = ozellikler.toArray(new double[0][]);
            int[] y = etiketler.stream().mapToInt(i -> i).toArray();
            int n = X.length;

            // Train Test Split
            int egitimBoyut = (int) (n * 0.7);
            Integer[] idx = new Integer[n];
            for (int i = 0; i < n; i++) idx[i] = i;
            Collections.shuffle(Arrays.asList(idx), new Random(42));

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

            // PCA ile Boyut İndirgeme (4 -> 2)
            var pca = PCA.fit(XTrain);
            var pca2d = pca.getProjection(2);

            double[][] XTrainPCA = pca2d.apply(XTrain);
            double[][] XTestPCA = pca2d.apply(XTest);

            var varyansVec = pca.varianceProportion();
            double[] varyansOrani = varyansVec.toArray(new double[varyansVec.size()]);

            System.out.println("\nPCA: 4 boyut -> 2 boyut");
            System.out.println("  PC1: " + String.format("%.2f", varyansOrani[0] * 100)
                    + ", PC2: " + String.format("%.2f", varyansOrani[1] * 100)
                    + ", Toplam: " + String.format("%.2f", (varyansOrani[0] + varyansOrani[1]) * 100));


            // LDA / FLD ile Boyut İndirgeme (4 -> 2)
            var fld = FLD.fit(XTrain, yTrain);
            var projTrain = fld.project(XTrain);
            double[][] XTrainLDA = new double[XTrain.length][];
            for (int i = 0; i < XTrain.length; i++) {
                // Vector.toArray(hedefDizi): Vector'u double[] dizisine kopyalar
                XTrainLDA[i] = projTrain[i].toArray(new double[projTrain[i].size()]);
            }

            // Test verisini de ayni LDA donusumuyle yansit
            var projTest = fld.project(XTest);
            double[][] XTestLDA = new double[XTest.length][];
            for (int i = 0; i < XTest.length; i++) {
                XTestLDA[i] = projTest[i].toArray(new double[projTest[i].size()]);
            }

            // Sınıflandırma Karşılaştırması
            int knnK = 5;

            // knn 4 boyut
            var knnOrijinal = KNN.fit(XTrain, yTrain, knnK);
            int[] tahOrijinal = knnOrijinal.predict(XTest);
            double accOrijinal = Accuracy.of(yTest, tahOrijinal);

            // pca 2 boyut
            var knnPCA = KNN.fit(XTrainPCA, yTrain, knnK);
            int[] tahPCA = knnPCA.predict(XTestPCA);
            double accPCA = Accuracy.of(yTest, tahPCA);

            // LDA 2 boyut
            var knnLDA = KNN.fit(XTrainLDA, yTrain, knnK);
            int[] tahLDA = knnLDA.predict(XTestLDA);
            double accLDA = Accuracy.of(yTest, tahLDA);

            // FLD
            int[] tahFLD = new int[yTest.length];
            for (int i = 0; i < yTest.length; i++) {
                tahFLD[i] = fld.predict(XTest[i]);
            }
            double accFLD = Accuracy.of(yTest, tahFLD);

            System.out.println("SONUÇLAR");
            System.out.println("Orijinal KNN: " + accOrijinal * 100);
            System.out.println("pca 2 boyut KNN: " + accPCA * 100);
            System.out.println("LDA 2 boyut KNN: " + accLDA * 100);
            System.out.println("FLD KNN: " + accFLD * 100);


        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
