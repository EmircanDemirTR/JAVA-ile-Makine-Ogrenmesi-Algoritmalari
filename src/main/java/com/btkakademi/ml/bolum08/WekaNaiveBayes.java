package com.btkakademi.ml.bolum08;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.util.Random;

public class WekaNaiveBayes {
    static void main() {
        try {
            var is = WekaNaiveBayes.class.getClassLoader().getResourceAsStream("datasets/mushroom.arff");
            ArffLoader loader = new ArffLoader();
            loader.setSource(is);

            Instances veri = loader.getDataSet();
            veri.setClassIndex(veri.numAttributes() - 1);

            int edibleCount = 0, poisonousCount = 0;
            for (int i = 0; i < veri.numInstances(); i++) {
                String sinif = veri.instance(i).stringValue(veri.classIndex());
                if (sinif.equals("e")) edibleCount++;
                else poisonousCount++;
            }

            System.out.println("Yenilebilir sayısı: " + edibleCount);
            System.out.println("Zehirli sayısı: " + poisonousCount);

            // Train Test Split
            veri.randomize(new Random(42));
            int egitimBoyut = (int) (veri.numInstances() * 0.7);

            Instances egitim = new Instances(veri, 0, egitimBoyut);
            Instances test = new Instances(veri, egitimBoyut, veri.numInstances() - egitimBoyut);

            // Model Oluşturma ve Egitim
            NaiveBayes nb = new NaiveBayes();
            nb.buildClassifier(egitim);

            // Değerlendirme
            Evaluation eval = new Evaluation(egitim);
            eval.evaluateModel(nb, test);

            System.out.println("Dogruluk: " + eval.pctCorrect());
            System.out.println("Kappa: " + eval.kappa());

            System.out.println(eval.toMatrixString());

            // Sınıf Bazında Metrikler

            for (int i = 0; i < veri.numClasses(); i++) {
                String sinifAdi = veri.classAttribute().value(i);
                String aciklama = sinifAdi.equals("e") ? "yenilebilir" : "zehirli";
                double p = eval.precision(i);
                double r = eval.recall(i);
                double f1 = eval.fMeasure(i);
                System.out.printf("Class %s (%s): P=%.3f, R=%.3f, F1=%.3f%n",
                        sinifAdi, aciklama, p, r, f1);
            }

            // False Negative Analizi
            double[][] cm = eval.confusionMatrix();
            // cm[actual][predicted]
            // Sınıf sırası: e=0, p=1
            int fn = (int) cm[1][0];

            if (fn > 0) {
                System.out.println("\nUYARI " + fn + "zehirli mantar tespit edilemedi.");
                System.out.println("Bu bir orman olsaydı " + fn + "kisi zehirlenebilirdi");
            }


        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
