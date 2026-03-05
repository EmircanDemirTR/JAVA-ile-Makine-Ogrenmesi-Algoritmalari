package com.btkakademi.ml.bolum10;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.InputStream;
import java.util.Random;

public class WekaRandomForest {
    static void main() {
        try {
            InputStream is = WekaRandomForest.class.getClassLoader().getResourceAsStream("datasets/segment.arff");
            ArffLoader loader = new ArffLoader();
            loader.setSource(is);

            Instances veri = loader.getDataSet();
            veri.setClassIndex(veri.numAttributes() - 1);


            // Train Test Split
            veri.randomize(new Random(42));
            int egitimBoyut = (int) (veri.numInstances() * 0.7);
            int testBoyut = veri.numInstances() - egitimBoyut;

            Instances egitim = new Instances(veri, 0, egitimBoyut);
            Instances testVeri = new Instances(veri, egitimBoyut, testBoyut);

            System.out.println("Egitim " + egitim.numInstances());
            System.out.println("Test: " + testVeri.numInstances());

            // RandomForest Modeli
            RandomForest rf = new RandomForest();

            // Parametre Ayarları
            rf.setNumIterations(100); // Kaç tane karar ağacı eğitilecek
            rf.setNumFeatures(0); // Her bölünmede kaç özellik denenecek
            rf.setMaxDepth(20); // Ağaçların maksimum derinliği
            rf.setBagSizePercent(100); // Bootstrap ornekleme yuzdesi
            rf.setSeed(42);

            int mtryHesap = (int) (Math.log(19) / Math.log(2) + 1);

            // Model Eğitimi
            long baslangic = System.currentTimeMillis();
            rf.buildClassifier(egitim);
            long sure = System.currentTimeMillis() - baslangic;
            System.out.println("Model Egitildi, sure: " + sure + "ms");


            // Değerlendirme
            Evaluation eval = new Evaluation(egitim);
            eval.evaluateModel(rf, testVeri);

            System.out.println("Dogruluk Orani: " + eval.pctCorrect());
            System.out.println("Kappa Istatistigi: " + eval.kappa());


            // Özellik Önemliği
            AttributeSelection attSelection = new AttributeSelection();

            InfoGainAttributeEval infoGainEval = new InfoGainAttributeEval();

            Ranker ranker = new Ranker();
            ranker.setNumToSelect(-1);

            attSelection.setEvaluator(infoGainEval);
            attSelection.setSearch(ranker);

            attSelection.SelectAttributes(veri);

            double[][] rankedAttrs = attSelection.rankedAttributes();

            System.out.println("En Onemli 5 Özellik");
            for (int i = 0; i < Math.min(5, rankedAttrs.length); i++) {
                int attrIdx = (int) rankedAttrs[i][0];
                double score = rankedAttrs[i][1];
                String attrName = veri.attribute(attrIdx).name();
                System.out.printf("   %d. %-22s: %.4f\n", i + 1, attrName, score);
            }


            // Ağaç Sayısı Etkisi
            System.out.println("Ağaç Sayısı Etkisi");
            int[] agacSayilari = {10, 25, 50, 100, 200, 300};

            for (int n : agacSayilari) {
                RandomForest testRf = new RandomForest();
                testRf.setNumIterations(n);
                testRf.setNumFeatures(0);
                testRf.setMaxDepth(20);
                testRf.setSeed(42);

                testRf.buildClassifier(egitim);
                Evaluation testEval = new Evaluation(egitim);
                testEval.evaluateModel(testRf, testVeri);

                System.out.println(n + "= " + testEval.pctCorrect());
            }


        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
