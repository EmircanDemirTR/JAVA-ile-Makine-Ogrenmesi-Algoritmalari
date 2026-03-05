package com.btkakademi.ml.bolum08;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.util.Random;

public class KararAgaci_Pruning {
    static void main() {
        try {
            var is = KararAgaci_Pruning.class.getClassLoader().getResourceAsStream("datasets/glass.arff");
            ArffLoader loader = new ArffLoader();
            loader.setSource(is);
            Instances veri = loader.getDataSet();

            veri.setClassIndex(veri.numAttributes() - 1);

            // Train - Test Split
            veri.randomize(new Random(42));
            int egitimBoyut = (int) (veri.numInstances() * 0.7);
            Instances egitim = new Instances(veri, 0, egitimBoyut);
            Instances test = new Instances(veri, egitimBoyut, veri.numInstances() - egitimBoyut);

            System.out.println("Egitim " + egitim.numInstances() + ", Test: " + test.numInstances());


            // Pruned vs Unpruned Karşılaştırması

            // UNPRUNED AĞAÇ
            J48 unpruned = new J48();
            unpruned.setUnpruned(true);
            unpruned.buildClassifier(egitim);

            Evaluation evalUnpruned = new Evaluation(egitim);
            evalUnpruned.evaluateModel(unpruned, test);
            int sizeUnpruned = (int) unpruned.measureTreeSize();
            int leavesUnpruned = (int) unpruned.measureNumLeaves();

            System.out.printf("Unpruned: Doğruluk:%.2f%%, Ağaç=%d dugum, %d yaprak%n", evalUnpruned.pctCorrect(), sizeUnpruned, leavesUnpruned);

            // PRUNED AĞAÇ CF = 0.25
            J48 pruned = new J48();
            pruned.buildClassifier(egitim);

            Evaluation evalpruned = new Evaluation(egitim);
            evalpruned.evaluateModel(pruned, test);
            int sizePruned = (int) pruned.measureTreeSize();
            int leavesPruned = (int) pruned.measureNumLeaves();

            System.out.printf("Pruned: Doğruluk:%.2f%%, Ağaç=%d dugum, %d yaprak%n", evalpruned.pctCorrect(), sizePruned, leavesPruned);

            // PRUNED AĞAÇ CF=0.05
            J48 prunedAggressive = new J48();
            prunedAggressive.setConfidenceFactor(0.05f);
            prunedAggressive.buildClassifier(egitim);

            Evaluation evalPrunedAggressive = new Evaluation(egitim);
            evalPrunedAggressive.evaluateModel(prunedAggressive, test);
            int sizePrunedAgg = (int) prunedAggressive.measureTreeSize();
            int leavesPrunedAgg = (int) prunedAggressive.measureNumLeaves();

            System.out.printf("Pruned Aggressive: Doğruluk:%.2f%%, Ağaç=%d dugum, %d yaprak%n", evalPrunedAggressive.pctCorrect(), sizePrunedAgg, leavesPrunedAgg);

            // Confidence Factor - Grid Search
            float[] cfDegerleri = {0.05f, 0.1f, 0.15f, 0.2f, 0.25f, 0.3f, 0.4f, 0.5f};
            double enIyiDogruluk = 0;
            float enIyiCF = 0.25f;

            for (float cf : cfDegerleri) {
                J48 tree = new J48();
                tree.setConfidenceFactor(cf);
                tree.buildClassifier(egitim);

                Evaluation eval = new Evaluation(egitim);
                eval.evaluateModel(tree, test);

                int size = (int) tree.measureTreeSize();
                double acc = eval.pctCorrect();

                String marker = "";
                if (acc > enIyiDogruluk) {
                    enIyiDogruluk = acc;
                    enIyiCF = cf;
                    marker = " <--- EN IYI";
                }
                System.out.printf("CF=%.2f -> Dogruluk=%.2f%%, Boyut=%d%s%n", cf, acc, size, marker);
            }
            System.out.println("\nSecilen CF " + enIyiCF);

            // MIN NUM OBJ GRID SEARCH
            // minNumObj : Yaprak düğümde minimum ornek sayisi
            int[] minObjDegerleri = {1, 2, 3, 5, 10, 15, 20};
            enIyiDogruluk = 0;
            int enIyiMinObj = 2;

            for (int minObj : minObjDegerleri) {
                J48 tree = new J48();
                tree.setConfidenceFactor(enIyiCF);

                tree.setMinNumObj(minObj);
                tree.buildClassifier(egitim);

                Evaluation eval = new Evaluation(egitim);
                eval.evaluateModel(tree, test);

                int leaves = (int) tree.measureNumLeaves();
                double acc = eval.pctCorrect();

                String marker = "";
                if (acc > enIyiDogruluk) {
                    enIyiDogruluk = acc;
                    enIyiMinObj = minObj;
                    marker = " <--- EN IYI";
                }
                System.out.printf("minObj=%2d -> Dogruluk=%.2f%%, Yaprak=%d%s%n", minObj, acc, leaves, marker);
            }

            System.out.println("\nSecilen minObj: " + enIyiMinObj);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
