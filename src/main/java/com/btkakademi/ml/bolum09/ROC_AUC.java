package com.btkakademi.ml.bolum09;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.util.Random;

public class ROC_AUC {
    static void main() {
        try {
            var is = ROC_AUC.class.getClassLoader().getResourceAsStream("datasets/breast-cancer.arff");
            ArffLoader loader = new ArffLoader();
            loader.setSource(is);
            Instances veri = loader.getDataSet();

            veri.setClassIndex(veri.numAttributes() - 1);
            veri.deleteAttributeAt(0);

            int[] sinifDagilimi = veri.attributeStats(veri.classIndex()).nominalCounts;
            String sinif0 = veri.classAttribute().value(0);
            String sinif1 = veri.classAttribute().value(1);

            System.out.printf(" %s: %d ornek\n", sinif0, sinifDagilimi[0]);
            System.out.printf(" %s: %d ornek\n", sinif1, sinifDagilimi[1]);

            int pozitifIndex = sinif0.equals("M") ? 0 : 1;
            System.out.println("Pozitif Sınıf (kanser)" + veri.classAttribute().value(pozitifIndex));

            // Train - Test Split
            veri.randomize(new Random(42));
            int egitimBoyut = (int) (veri.numInstances() * 0.7);

            Instances egitimVeri = new Instances(veri, 0, egitimBoyut);
            Instances testVeri = new Instances(veri, egitimBoyut, veri.numInstances() - egitimBoyut);

            // Model Eğitimi
            Logistic model = new Logistic();
            model.buildClassifier(egitimVeri);

            // Temel Değerlendirme
            Evaluation eval = new Evaluation(egitimVeri);
            eval.evaluateModel(model, testVeri);
            System.out.println("Doğruluk Oranı: " + eval.pctCorrect());
            System.out.println("Precision: " + eval.precision(pozitifIndex));
            System.out.println("Recall (TPR): " + eval.recall(pozitifIndex));

            // FPR hesabı
            double fp = eval.numFalsePositives(pozitifIndex);
            double tn = eval.numTrueNegatives(pozitifIndex);
            double fpr = fp / (fp + tn);

            // Karmaşıklık Matrisi
            System.out.println(eval.toMatrixString());

            // AUC Değeri Hesaplama
            double auc = eval.areaUnderROC(pozitifIndex);
            System.out.println("AUC Değeri: " + auc);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
