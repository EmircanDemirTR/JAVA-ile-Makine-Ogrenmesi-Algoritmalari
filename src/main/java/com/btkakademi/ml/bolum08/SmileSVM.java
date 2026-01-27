package com.btkakademi.ml.bolum08;

import smile.classification.SVM;
import smile.math.kernel.GaussianKernel;
import smile.validation.metric.Accuracy;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.*;

public class SmileSVM {
    static void main() {
        try {
            var is = SmileSVM.class.getClassLoader().getResourceAsStream("datasets/breast-cancer.csv");
            var reader = new BufferedReader(new InputStreamReader(is));
            reader.readLine();

            List<double[]> ozellikler = new ArrayList<>(); // X: 30 özellik
            List<Integer> etiketler = new ArrayList<>(); // y: -1 veya +1

            int mCount = 0, bCount = 0;

            String satir;
            while ((satir = reader.readLine()) != null) {
                String[] p = satir.split(",");

                String diagnosis = p[1].trim(); // M veya B
                double[] ozellik = new double[30];
                for (int i = 0; i < 30; i++) {
                    ozellik[i] = Double.parseDouble(p[i + 2].trim());
                }
                ozellikler.add(ozellik);

                // SVM için etiketler: M-> +1 , B -> -1
                if (diagnosis.equals("M")) {
                    etiketler.add(1);
                    mCount++;
                } else {
                    etiketler.add(-1);
                    bCount++;
                }
            }

            reader.close();

            double[][] X = ozellikler.toArray(new double[0][]);
            int[] y = etiketler.stream().mapToInt(i -> i).toArray();

            // Veri Setini Yazdırma
            System.out.println("Malignant Sayısı: " + mCount + ", Bening: " + bCount);

            // Normalizasyon
            double[][] XNorm = normalize(X);

            // Train Test Split

            int n = X.length;
            int egitimBoyut = (int) (n * 0.7);
            Integer[] idx = new Integer[n];
            for (int i = 0; i < n; i++) idx[i] = i;
            Collections.shuffle(Arrays.asList(idx), new Random(42));

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

            System.out.println("Egitim veri seti: " + XTrain.length + ", Test veri seti: " + XTest.length);

            // Model Eğitimi
            double sigma = 1.0; // RBF Kernel Genişliği
            double C = 1.0; // Regularization
            double tol = 1e-3; // Tolerans
            int epochs = 100; // Max iterasyon

            var kernel = new GaussianKernel(sigma);
            var options = new SVM.Options(C, tol, epochs);

            // Model Eğitimi
            var model = SVM.fit(XTrain, yTrain, kernel, options);

            // Tahmin ve Değerlendirme
            int[] tahminler = model.predict(XTest);

            double dogruluk = Accuracy.of(yTest, tahminler);
            System.out.println("Dogruluk degeri: " + dogruluk * 100);

            
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static double[][] normalize(double[][] X) {
        int n = X.length;
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
                XNorm[i][j] = (X[i][j] - min[j]) / (max[j] - min[j] + 1e-10);
            }
        }
        return XNorm;
    }
}
