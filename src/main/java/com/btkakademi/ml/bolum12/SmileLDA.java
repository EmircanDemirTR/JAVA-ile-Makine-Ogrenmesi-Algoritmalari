package com.btkakademi.ml.bolum12;

import smile.classification.FLD;
import smile.validation.metric.Accuracy;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.*;

public class SmileLDA {
    static void main() {
        try {

            var is = SmileLDA.class.getClassLoader().getResourceAsStream("datasets/wine.csv");
            var reader = new BufferedReader(new InputStreamReader(is));
            reader.readLine();

            List<double[]> ozellikler = new ArrayList<>();
            List<Integer> etiketler = new ArrayList<>();

            String satir;
            while ((satir = reader.readLine()) != null) {
                String[] p = satir.split(",");
                etiketler.add(Integer.parseInt(p[0].trim()) - 1);
                double[] ozellik = new double[13];
                for (int i = 0; i < 13; i++) {
                    ozellik[i] = Double.parseDouble(p[i + 1].trim());
                }
                ozellikler.add(ozellik);
            }
            reader.close();

            double[][] X = ozellikler.toArray(new double[0][]);
            int[] y = etiketler.stream().mapToInt(i -> i).toArray();

            // Keşifsel FLD Projeksiyon
            var fldKesif = FLD.fit(X, y);
            var projections = fldKesif.project(X);
            System.out.println("Ilk 10 ornegin 2D Projeksiyonu: ");
            for (int i = 0; i < 10; i++) {
                double[] proj = projections[i].toArray(new double[projections[i].size()]);
                // LD1 = birinci diskriminant eksen (en cok ayrim saglayan)
                // LD2 = ikinci diskriminant eksen
                System.out.println("  #" + (i + 1) + ": LD1=" + String.format("%.4f", proj[0])
                        + ", LD2=" + String.format("%.4f", proj.length > 1 ? proj[1] : 0.0)
                        + " (Sinif " + y[i] + ")");
            }


            // Sınıflandırma
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
                XTrain[i] = X[idx[i]];
                yTrain[i] = y[idx[i]];
            }
            for (int i = egitimBoyut; i < n; i++) {
                XTest[i - egitimBoyut] = X[idx[i]];
                yTest[i - egitimBoyut] = y[idx[i]];
            }

            var fldModel = FLD.fit(XTrain, yTrain);

            int[] tahminler = new int[yTest.length];
            for (int i = 0; i < yTest.length; i++) {
                tahminler[i] = fldModel.predict(XTest[i]);
            }

            double dogruluk = Accuracy.of(yTest, tahminler);
            System.out.println("FLD Test Doğruluğu: " + dogruluk * 100);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
