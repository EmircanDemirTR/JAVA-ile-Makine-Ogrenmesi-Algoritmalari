package com.btkakademi.ml.bolum08;

import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.util.Random;

public class KNN_KDegeriBelirleme {
    static void main() {
        try {
            var is = KNN_KDegeriBelirleme.class.getClassLoader().getResourceAsStream("datasets/iris.arff");

            ArffLoader loader = new ArffLoader();
            loader.setSource(is);

            Instances veri = loader.getDataSet();
            veri.setClassIndex(veri.numAttributes() - 1);

            System.out.println("sqrn(n) ile başlanabilir: " + (int) Math.sqrt(veri.numInstances()));

            // Train/Test Split
            veri.randomize(new Random());
            int egitimBoyut = (int) (veri.numInstances() * 0.7);
            Instances egitim = new Instances(veri, 0, egitimBoyut);
            Instances test = new Instances(veri, egitimBoyut, veri.numInstances() - egitimBoyut);

            // 1. Yöntem - Holdout Validation
            int[] kDegerleri = {1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21};
            double[] holdoutSonuc = new double[kDegerleri.length];
            double enIyiHoldout = 0;
            int enIyiKHoldout = 1;

            for (int i = 0; i < kDegerleri.length; i++) {
                int k = kDegerleri[i];
                IBk knn = new IBk(k);
                knn.buildClassifier(egitim);

                Evaluation eval = new Evaluation(egitim);
                eval.evaluateModel(knn, test);
                holdoutSonuc[i] = eval.pctCorrect();

                if (holdoutSonuc[i] > enIyiHoldout) {
                    enIyiHoldout = holdoutSonuc[i];
                    enIyiKHoldout = k;
                }
                System.out.println("k değeri: " + k + ", holdoutSonuç: " + holdoutSonuc[i]);
            }

            System.out.println("En iyi k değeri: " + enIyiKHoldout);

            // 2. Yöntem: k-Fold Çapraz Doğrulama (Cross Validation)
            double[] cvSonuc = new double[kDegerleri.length];
            double enIyiCV = 0;
            double enIyiKCV = 1;

            for (int i = 0; i < kDegerleri.length; i++) {
                int k = kDegerleri[i];
                IBk knn = new IBk(k);

                Evaluation cvEval = new Evaluation(veri);
                cvEval.crossValidateModel(knn, veri, 10, new Random(42));
                cvSonuc[i] = cvEval.pctCorrect();

                if (cvSonuc[i] > enIyiCV) {
                    enIyiCV = cvSonuc[i];
                    enIyiKCV = k;
                }

                System.out.println("K değeri: " + k + ", CVsonuç: " + cvSonuc[i]);
            }
            System.out.println("CV için en iyi K değeri: " + enIyiKCV);

            // 3. Yöntem - Elbow Method
            System.out.println("\nElbow Method");
            System.out.println("K      | Hata    | Grafik");
            for (int i = 0; i < kDegerleri.length; i++) {
                double hata = 100 - cvSonuc[i];
                int bar = (int) (hata * 3);
                String grafik = "*".repeat(Math.max(1, bar));
                String marker = "";
                if (i > 0 && i < kDegerleri.length - 1) {
                    double oncekiFark = (100 - cvSonuc[i - 1]) - (100 - cvSonuc[i]);
                    double sonrakiFark = (100 - cvSonuc[i]) - (100 - cvSonuc[i + 1]);
                    if (oncekiFark > sonrakiFark * 2 && oncekiFark > 0.5) {
                        marker = "Elbow burasıdır.";
                    }
                }

                System.out.printf("%2d   | %5.2f  | %s%s%n", kDegerleri[i], hata, grafik, marker);
            }

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
