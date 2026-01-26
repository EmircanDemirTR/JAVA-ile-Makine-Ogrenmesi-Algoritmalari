package com.btkakademi.ml.bolum08;

import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.util.Random;

public class WekaKNN {
    static void main() {
        try {
            var is = WekaKNN.class.getClassLoader().getResourceAsStream("datasets/iris.arff");
            if (is == null) throw new RuntimeException("iris.arff bulunamadi");

            ArffLoader loader = new ArffLoader();
            loader.setSource(is);

            Instances veri = loader.getDataSet();
            veri.setClassIndex(veri.numAttributes() - 1);

            System.out.println("Ozellikler: ");
            for (int i = 0; i < veri.numAttributes() - 1; i++) {
                System.out.print(veri.attribute(i).name());
                if (i < veri.numAttributes() - 2) System.out.print(", ");
            }
            System.out.println();

            // Train / Test Split
            veri.randomize(new Random(42));
            int egitimBoyut = (int) (veri.numInstances() * 0.7);

            Instances egitimVeri = new Instances(veri, 0, egitimBoyut);
            Instances testVeri = new Instances(veri, egitimBoyut, veri.numInstances() - egitimBoyut);

            System.out.println("Train / Test Split");
            System.out.println("Egitim seti: " + egitimVeri.numInstances());
            System.out.println("Test seti: " + testVeri.numInstances());

            // Model Eğitimi
            int k = 5;
            // IBk: I=Instance, B=Based, k=komşu sayısı
            IBk knn = new IBk();
            knn.setKNN(k);

            knn.buildClassifier(egitimVeri);

            System.out.println("\nModel Bilgileri: ");
            System.out.println("K degeri: " + k);

            // Değerlendirme
            Evaluation eval = new Evaluation(egitimVeri);
            eval.evaluateModel(knn, testVeri);

            System.out.println("Doğruluk: " + eval.pctCorrect());
            System.out.println("Yanlış Sınıflandırma: " + eval.pctIncorrect());

            System.out.println("Kappa Istatistigi: " + eval.kappa()); // 1.0 mukemmel, 0 rastgele, <0 rastgeleden kotu

            System.out.println("Mean Absolute Error" + eval.meanAbsoluteError());

            System.out.println("RMSE: " + eval.rootMeanSquaredError());

            // Confusion Matrix
            // satirlar=gerçek, sutunlar=tahmin
            System.out.println("Karmaşıklık Matrisi: ");
            System.out.println(eval.toMatrixString());

            // Sınıf Bazında Metrikler
            String[] siniflar = {"setosa", "versicolor", "virginica"};
            System.out.println("SINIF BAZINDA PERFORMANS");
            for (int i = 0; i < siniflar.length; i++) {
                System.out.println(siniflar[i]);
                System.out.println("Precision: " + eval.precision(i));
                System.out.println("Recall: " + eval.recall(i));
                System.out.println("F1-Score: " + eval.fMeasure(i));
            }

        } catch (Exception e) {
            System.out.println("Hata: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
