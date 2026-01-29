package com.btkakademi.ml.bolum10;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Random;

public class WekaRandomForest_Regresyon {
    static void main() {
        try {
            var is = WekaRandomForest_Regresyon.class.getClassLoader().getResourceAsStream("datasets/winequality-red.csv");
            var reader = new BufferedReader(new InputStreamReader(is));
            String headerLine = reader.readLine();
            String[] sutunlar = headerLine.split(",");

            ArrayList<Attribute> attributes = new ArrayList<>();
            for (String sutun : sutunlar) {
                attributes.add(new Attribute(sutun.trim()));
            }

            Instances veri = new Instances("winequality-red", attributes, 2000);
            veri.setClassIndex(veri.numAttributes() - 1);

            String satir;
            while ((satir = reader.readLine()) != null) {
                String[] degerler = satir.split(",");
                DenseInstance instance = new DenseInstance(veri.numAttributes());
                for (int i = 0; i < degerler.length; i++) {
                    instance.setValue(i, Double.parseDouble(degerler[i].trim()));
                }
                instance.setDataset(veri);
                veri.add(instance);
            }
            reader.close();

            // Hedef İstatistikleri

            double minQ = Double.MAX_VALUE, maxQ = Double.MIN_VALUE, sumQ = 0;
            for (int i = 0; i < veri.numInstances(); i++) {
                double q = veri.instance(i).classValue();
                if (q < minQ) minQ = q;
                if (q > maxQ) maxQ = q;
                sumQ += q;
            }
            System.out.printf("Quality: %.0f - %.0f (ort: %.2f)\n", minQ, maxQ, sumQ / veri.numInstances());

            // Train - Test Split
            veri.randomize(new Random(42));

            int egitimBoyut = (int) (veri.numInstances() * 0.7);
            Instances egitim = new Instances(veri, 0, egitimBoyut);
            Instances test = new Instances(veri, egitimBoyut, veri.numInstances() - egitimBoyut);

            // Random Forest Modeli
            RandomForest rf = new RandomForest();
            rf.setNumIterations(100);
            rf.setNumFeatures(0);
            rf.setMaxDepth(20);
            rf.setBagSizePercent(100);
            rf.setSeed(42);

            // Modelin Eğitimi
            long baslangic = System.currentTimeMillis();
            rf.buildClassifier(egitim);
            long sure = System.currentTimeMillis() - baslangic;
            System.out.println("Modelimiz egitildi! Suresi: " + sure + "ms");

            // Regresyon Metrikleri

            Evaluation eval = new Evaluation(egitim);
            eval.evaluateModel(rf, test);

            System.out.println("MAE: " + eval.meanAbsoluteError());
            System.out.println("RMSE: " + eval.rootMeanSquaredError());
            System.out.println("Korelasyon: " + eval.correlationCoefficient());

            double r = eval.correlationCoefficient();
            double r2 = r * r;
            System.out.println("R2 Degeri: " + r2 * 100);

            // Lineer Regresyon Karşılaştırması
            LinearRegression lr = new LinearRegression();
            lr.setEliminateColinearAttributes(true);
            lr.buildClassifier(egitim);

            Evaluation lrEval = new Evaluation(egitim);
            lrEval.evaluateModel(lr, test);

            System.out.println("MAE: " + lrEval.meanAbsoluteError());
            System.out.println("RMSE: " + lrEval.rootMeanSquaredError());
            System.out.println("Korelasyon: " + lrEval.correlationCoefficient());

            double lrR = lrEval.correlationCoefficient();
            double lrR2 = lrR * lrR;
            System.out.println("R2 Degeri: " + lrR2 * 100);


        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
