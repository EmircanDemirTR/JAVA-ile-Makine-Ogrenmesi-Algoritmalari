package com.btkakademi.ml.bolum08;

import smile.classification.KNN;
import smile.math.distance.ChebyshevDistance;
import smile.math.distance.EuclideanDistance;
import smile.math.distance.ManhattanDistance;
import smile.math.distance.MinkowskiDistance;
import smile.validation.metric.Accuracy;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.*;

public class KNN_MesafeMetrikleri {
    static void main() {
        try {
            var is = SmileKNN.class.getClassLoader().getResourceAsStream("datasets/iris.csv");
            if (is == null) throw new RuntimeException("iris.csv dosyasi bulunamadi");

            var reader = new BufferedReader(new InputStreamReader(is));
            reader.readLine();

            List<double[]> ozellikListesi = new ArrayList<>();
            List<Integer> etiketListesi = new ArrayList<>();

            Map<String, Integer> etiketMap = new LinkedHashMap<>();
            etiketMap.put("setosa", 0);
            etiketMap.put("versicolor", 1);
            etiketMap.put("virginica", 2);

            String satir;
            while ((satir = reader.readLine()) != null) {
                String[] p = satir.split(",");
                ozellikListesi.add(new double[]{
                        Double.parseDouble(p[0]),
                        Double.parseDouble(p[1]),
                        Double.parseDouble(p[2]),
                        Double.parseDouble(p[3]),
                });
                etiketListesi.add(etiketMap.get(p[4].trim()));
            }

            reader.close();

            double[][] X = ozellikListesi.toArray(new double[0][]);
            int[] y = etiketListesi.stream().mapToInt(i -> i).toArray();

            // Normaliazsyon
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
            int[] yTrain = new int[egitimBoyut];
            double[][] XTest = new double[n - egitimBoyut][];
            int[] yTest = new int[n - egitimBoyut];

            for (int i = 0; i < egitimBoyut; i++) {
                XTrain[i] = X[idx[i]];
                yTrain[i] = y[idx[i]];
            }

            for (int i = egitimBoyut; i < n; i++) {
                XTest[i - egitimBoyut] = X[idx[i]];
                yTest[i - egitimBoyut] = y[idx[i]];
            }

            System.out.println("Egitim Seti: " + XTrain.length);
            System.out.println("Test Seti: " + XTest.length);


            // Mesafe Metrikleri
            int k = 5;
            double enIyiDogruluk = 0;
            String enIyiMetrik = "";

            // Euclidean
            var modelEuc = KNN.fit(XTrain, yTrain, k, new EuclideanDistance());
            double accEuc = Accuracy.of(yTest, modelEuc.predict(XTest)) * 100;
            System.out.println("Euclidean Doğruluk Orani: " + accEuc);
            if (accEuc > enIyiDogruluk) {
                enIyiDogruluk = accEuc;
                enIyiMetrik = "Euclidean";
            }

            // Manhattan
            var modelMan = KNN.fit(XTrain, yTrain, k, new ManhattanDistance());
            double accMan = Accuracy.of(yTest, modelMan.predict(XTest)) * 100;
            System.out.println("Manhattan Doğruluk Orani: " + accMan);
            if (accMan > enIyiDogruluk) {
                enIyiDogruluk = accMan;
                enIyiMetrik = "Manhattan";
            }

            // Chebyshev
            var modelCheb = KNN.fit(XTrain, yTrain, k, new ChebyshevDistance());
            double accCheb = Accuracy.of(yTest, modelCheb.predict(XTest)) * 100;
            System.out.println("Chebyshev Doğruluk Orani: " + accCheb);
            if (accCheb > enIyiDogruluk) {
                enIyiDogruluk = accCheb;
                enIyiMetrik = "Chebyshev";
            }

            // Minkowski
            var modelMink = KNN.fit(XTrain, yTrain, k, new MinkowskiDistance(3));
            double accMink = Accuracy.of(yTest, modelMink.predict(XTest)) * 100;
            System.out.println("Minkowski Doğruluk Orani: " + accMink);
            if (accMink > enIyiDogruluk) {
                enIyiDogruluk = accMink;
                enIyiMetrik = "Minkowski";
            }

            System.out.println("\nSecilen metrik: " + enIyiMetrik);

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
                XNorm[i][j] = (X[i][j] - min[j]) / (max[j] - min[j]);
            }
        }
        return XNorm;
    }
}
