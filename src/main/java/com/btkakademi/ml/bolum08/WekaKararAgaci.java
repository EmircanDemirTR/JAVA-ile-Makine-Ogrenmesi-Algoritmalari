package com.btkakademi.ml.bolum08;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.util.Random;

public class WekaKararAgaci {
    static void main() {
        try {
            var is = WekaKararAgaci.class.getClassLoader().getResourceAsStream("datasets/wine.arff");
            ArffLoader loader = new ArffLoader();
            loader.setSource(is);

            Instances veri = loader.getDataSet();
            veri.setClassIndex(veri.numAttributes() - 1);

            // Train/Test Split
            veri.randomize(new Random(42));
            int egitimBoyut = (int) (veri.numInstances() * 0.7);
            Instances egitim = new Instances(veri, 0, egitimBoyut);
            Instances test = new Instances(veri, egitimBoyut, veri.numInstances() - egitimBoyut);

            System.out.println("Egitim " + egitim.numInstances() + ", Test: " + test.numInstances());

            // J48 Model Eğitimi
            System.out.println("J48 Model Eğitimi");
            J48 tree = new J48();

            long baslangic = System.currentTimeMillis();
            tree.buildClassifier(egitim);
            long sure = System.currentTimeMillis() - baslangic;

            System.out.println("J48 modelimiz egitildi, sure olarak: " + sure + " ms");

            // Ağaç Metrikleri
            System.out.println("Ağaç Metrikleri");
            System.out.printf("Dugum: %.0f, Yaprak: %.0f, Ic Dugum: %.0f%n",
                    tree.measureTreeSize(), tree.measureNumLeaves(), tree.measureTreeSize() - tree.measureNumLeaves());


            // Değerlendirme
            System.out.println("Değerlendirme");
            Evaluation eval = new Evaluation(egitim);
            eval.evaluateModel(tree, test);

            System.out.println("Dogruluk: " + eval.pctCorrect() + ", Kappa: " + eval.kappa());

            // Confusion Matrix
            System.out.println(eval.toMatrixString());

            // Ağaç Yapısı
            System.out.println(tree);

            // Örnek Tahminler
            System.out.println("Örnek Tahminler");

            for (int i = 0; i < Math.min(5, test.numInstances()); i++) {
                var ornek = test.instance(i);
                double tahminIdx = tree.classifyInstance(ornek);
                String tahmin = test.classAttribute().value((int) tahminIdx);
                String gercek = ornek.stringValue(ornek.classIndex());

                String sonuc = gercek.equals(tahmin) ? "OK" : "YANLIS";
                System.out.printf("Ornek: %d: Gercek=%-12s Tahmin=%-12s [%s]%n", i + 1, gercek, tahmin, sonuc);
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
