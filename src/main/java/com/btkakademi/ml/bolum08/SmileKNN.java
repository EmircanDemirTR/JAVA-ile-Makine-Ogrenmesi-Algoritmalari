package com.btkakademi.ml.bolum08;

import smile.classification.KNN;
import smile.validation.metric.Accuracy;
import smile.validation.metric.ConfusionMatrix;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.*;

public class SmileKNN {
    static void main(String[] args) {
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

            String[] sinifIsimleri = {"setosa", "versicolor", "virginica"};

            String satir;
            while ((satir = reader.readLine()) != null) {
                String[] p = satir.split(",");
                double[] ozellikler = {
                        Double.parseDouble(p[0].trim()),
                        Double.parseDouble(p[1].trim()),
                        Double.parseDouble(p[2].trim()),
                        Double.parseDouble(p[3].trim())
                };
                ozellikListesi.add(ozellikler);
                etiketListesi.add(etiketMap.get(p[4].trim()));
            }
            reader.close();

            double[][] X = ozellikListesi.toArray(new double[0][]);
            int[] y = etiketListesi.stream().mapToInt(i -> i).toArray();

            System.out.println("Ozellik Sayisi: " + X[0].length);
            System.out.println("Sinif Sayisi: " + etiketMap.size());

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

            // Model Eğitimi
            int k = 5;
            var model = KNN.fit(XTrain, yTrain, k);

            // Tahmin ve Değerlendirme
            int[] tahminler = model.predict(XTest);

            // Doğruluk hesabı
            double dogruluk = Accuracy.of(yTest, tahminler) * 100;
            System.out.println("Doğruluk Orani: " + dogruluk);

            // Confusion Matrix -- Satirlar:gerçek sınıfları, Sütunlar:tahmin edilen sınıflar
            System.out.println("Karmasiklik Matrisi");
            var cm = ConfusionMatrix.of(yTest, tahminler);
            System.out.println(cm);

            // Yeni Veri Üzerinde Tahmin
            System.out.println("Yeni Veri Üzerinde Tahmin");
            double[][] yeniVeriler = {
                    {5.1, 3.5, 1.4, 0.2}, // setosa
                    {6.0, 2.7, 4.5, 1.5}, // versicolor
                    {6.5, 3.0, 5.5, 2.0} // virginica
            };

            for (double[] yeniVeri : yeniVeriler) {
                int tahmin = model.predict(yeniVeri);
                System.out.println(sinifIsimleri[tahmin]);
            }

        } catch (Exception e) {
            System.err.println("Hata: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
