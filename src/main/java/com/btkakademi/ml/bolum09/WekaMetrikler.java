package com.btkakademi.ml.bolum09;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.util.Random;

public class WekaMetrikler {
    static void main() {
        try {
            var is = WekaMetrikler.class.getClassLoader().getResourceAsStream("datasets/wine.arff");
            ArffLoader loader = new ArffLoader();
            loader.setSource(is);

            Instances veri = loader.getDataSet();
            veri.setClassIndex(veri.numAttributes() - 1);

            // Train Test Split
            veri.randomize(new Random(42));
            int egitimBoyut = (int) (veri.numInstances() * 0.7);

            Instances egitimVeri = new Instances(veri, 0, egitimBoyut);
            Instances testVeri = new Instances(veri, egitimBoyut, veri.numInstances() - egitimBoyut);

            System.out.println("Egitim: " + egitimVeri.numInstances());
            System.out.println("Test " + testVeri.numInstances());

            // Model Eğitimi
            J48 model = new J48();
            model.setConfidenceFactor(0.25f);
            model.buildClassifier(egitimVeri);

            // Model Değerlendirme
            Evaluation eval = new Evaluation(egitimVeri);
            eval.evaluateModel(model, testVeri);

            System.out.println("Genel Metrikler");
            System.out.println("Doğruluk Oranı: " + eval.pctCorrect());
            System.out.println("Kappa: " + eval.kappa());
            System.out.println("MAE: " + eval.meanAbsoluteError());
            System.out.println("RMSE: " + eval.rootMeanSquaredError());

            System.out.println("Sinif\tPrecision\tRecall\t\tF1-Score");
            // Sınıf Bazlı Metrikler
            for (int i = 0; i < veri.numClasses(); i++) {
                String sinifAdi = veri.classAttribute().value(i);
                double precision = eval.precision(i);
                double recall = eval.recall(i);
                double f1 = eval.fMeasure(i);
                System.out.printf("%s\t\t%.4f\t\t%.4f\t\t%.4f\n", sinifAdi, precision, recall, f1);
            }

            // Ağırlıklı Metrikler
            System.out.println("Ağırlıklı Precision Ortalaması: " + eval.weightedPrecision());
            System.out.println("Ağırlıklı Recall Ortalaması: " + eval.weightedRecall());
            System.out.println("Ağırlıklı F1 Ortalaması: " + eval.weightedFMeasure());

            // Karışıklık Matrisi
            System.out.println(eval.toMatrixString());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
