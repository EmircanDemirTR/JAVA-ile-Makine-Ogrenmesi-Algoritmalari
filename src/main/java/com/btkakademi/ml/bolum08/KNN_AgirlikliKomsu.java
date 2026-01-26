package com.btkakademi.ml.bolum08;

import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.converters.ArffLoader;

import java.util.Random;

public class KNN_AgirlikliKomsu {
    static void main() {
        try {
            var is = KNN_AgirlikliKomsu.class.getClassLoader().getResourceAsStream("datasets/iris.arff");
            if (is == null) throw new RuntimeException("iris.arff bulunamadi");

            ArffLoader loader = new ArffLoader();
            loader.setSource(is);

            Instances veri = loader.getDataSet();
            veri.setClassIndex(veri.numAttributes() - 1);

            // Train / Test Split
            veri.randomize(new Random(42));
            int egitimBoyut = (int) (veri.numInstances() * 0.7);
            Instances egitim = new Instances(veri, 0, egitimBoyut);
            Instances test = new Instances(veri, egitimBoyut, veri.numInstances() - egitimBoyut);

            System.out.println("Egitim: " + egitim.numInstances() + ", Test: " + test.numInstances());

            // Agirlik Tipi Karşılaştırması
            System.out.println("Agirlik Tipi Karşılaştırması");
            int k = 7;

            String[] agirlikAdlari = {"No Weighting", "1/distance", "1-distance"};
            int[] agirlikTipleri = {IBk.WEIGHT_NONE, IBk.WEIGHT_INVERSE, IBk.WEIGHT_SIMILARITY};

            double enIyiDogruluk = 0;
            String enIyiTip = "";

            for (int i = 0; i < agirlikTipleri.length; i++) {
                IBk knn = new IBk(k);
                knn.setDistanceWeighting(new SelectedTag(agirlikTipleri[i], IBk.TAGS_WEIGHTING));
                knn.buildClassifier(egitim);

                Evaluation eval = new Evaluation(egitim);
                eval.evaluateModel(knn, test);

                double acc = eval.pctCorrect();

                String marker = "";
                if (acc > enIyiDogruluk) {
                    enIyiDogruluk = acc;
                    enIyiTip = agirlikAdlari[i];
                    marker = " <--- EN IYI";
                }

                System.out.printf("%-15s: %.2f%% (Kappa: %.4f)%s%n", agirlikAdlari[i], acc, eval.kappa(), marker);
            }

        } catch (Exception e) {
            System.err.println("Hata: " + e.getMessage());
        }
    }
}
