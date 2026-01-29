package com.btkakademi.ml.bolum10;

import weka.classifiers.Evaluation;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.InputStream;
import java.util.Random;

public class WekaAdaBoost {
    static void main() {
        try {
            InputStream is = WekaAdaBoost.class.getClassLoader().getResourceAsStream("datasets/glass.arff");
            ArffLoader loader = new ArffLoader();
            loader.setSource(is);

            Instances veri = loader.getDataSet();
            veri.setClassIndex(veri.numAttributes() - 1);

            // Train Test Split
            veri.randomize(new Random(42));
            int egitimBoyut = (int) (veri.numInstances() * 0.7);
            Instances egitim = new Instances(veri, 0, egitimBoyut);
            Instances test = new Instances(veri, egitimBoyut, veri.numInstances() - egitimBoyut);

            // AdaBoost Modeli
            AdaBoostM1 adaBoost = new AdaBoostM1();
            adaBoost.setNumIterations(50);
            adaBoost.setClassifier(new DecisionStump());
            adaBoost.setUseResampling(false);
            adaBoost.setSeed(42);

            // Model Eğitimi ve Değerlendirme
            long baslangic = System.currentTimeMillis();
            adaBoost.buildClassifier(egitim);
            long sure = System.currentTimeMillis() - baslangic;
            System.out.println("Model egitildi ve sure: " + sure + "ms");

            Evaluation eval = new Evaluation(egitim);
            eval.evaluateModel(adaBoost, test);

            System.out.println("Dogruluk: " + eval.pctCorrect());
            System.out.println("Kappa: " + eval.kappa());

            // Random Forest Karşılaştırması
            RandomForest rf = new RandomForest();
            rf.setNumIterations(100);
            rf.setMaxDepth(20);
            rf.setSeed(42);
            rf.buildClassifier(egitim);

            Evaluation rfEval = new Evaluation(egitim);
            rfEval.evaluateModel(rf, test);
            System.out.println("AdaBoost Dogruluk: " + eval.pctCorrect());
            System.out.println("RandomForest Dogruluk: " + rfEval.pctCorrect());

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
