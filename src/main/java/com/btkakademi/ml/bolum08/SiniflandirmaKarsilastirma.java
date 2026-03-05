package com.btkakademi.ml.bolum08;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

public class SiniflandirmaKarsilastirma {
    static void main() {
        try {
            var is = SiniflandirmaKarsilastirma.class.getClassLoader().getResourceAsStream("datasets/wine.arff");

            ArffLoader loader = new ArffLoader();
            loader.setSource(is);
            Instances veri = loader.getDataSet();

            veri.setClassIndex(veri.numAttributes() - 1);

            // Min-Max Normlizasyon
            Normalize norm = new Normalize();
            norm.setInputFormat(veri);
            Instances normalVeri = Filter.useFilter(veri, norm);

            // Train Test Split
            normalVeri.randomize(new Random(42));

            int egitimBoyut = (int) (normalVeri.numInstances() * 0.7);
            Instances egitim = new Instances(normalVeri, 0, egitimBoyut);
            Instances test = new Instances(normalVeri, egitimBoyut, normalVeri.numInstances() - egitimBoyut);

            // Sınıflandırma Algoritmalarını Tanımla
            Map<String, Classifier> algoritmalar = new LinkedHashMap<>();
            // 1. KNN K-En Yakın Komşu Algoritması
            IBk knn = new IBk(5);
            algoritmalar.put("KNN (K=5)", knn);

            // 2. Decision Tree (J47 / C.45)
            J48 dt = new J48();
            algoritmalar.put("Decision Tree(J78)", dt);

            // 3. Logistic Regression
            Logistic lr = new Logistic();
            algoritmalar.put("Logistic Regression", lr);

            // 4. SVM - Destek Vektör Makinesi
            SMO svm = new SMO();
            algoritmalar.put("SVM", svm);

            // 5. Naive Bayes
            NaiveBayes nb = new NaiveBayes();
            algoritmalar.put("Naive Bayes", nb);

            // Hold Out Validation - Değerlendirme
            System.out.println("Hold-Out Validation");
            String enIyiAlgoritma = "";
            double enIyiDogruluk = 0;

            for (var entry : algoritmalar.entrySet()) {
                String isim = entry.getKey();
                Classifier model = entry.getValue();

                long baslangic = System.currentTimeMillis();
                model.buildClassifier(egitim);
                long egitimSuresi = System.currentTimeMillis() - baslangic;

                Evaluation eval = new Evaluation(egitim);
                eval.evaluateModel(model, test);

                // Metrikler
                double dogruluk = eval.pctCorrect();
                double kappa = eval.kappa();
                double f1 = eval.weightedFMeasure();

                if (dogruluk > enIyiDogruluk) {
                    enIyiDogruluk = dogruluk;
                    enIyiAlgoritma = isim;
                }
                System.out.printf("%-25s Acc=%.2f%%, Kappa=%.3f, F1=%.3f Süre:(%dms)%n", isim, dogruluk, kappa, f1, egitimSuresi);
            }
            System.out.println("En Iyı Dogruluk: " + enIyiDogruluk);
            System.out.println("En iyi Algoritma" + enIyiAlgoritma);

            System.out.println();
            System.out.println();

            // k-Fold Cross Validation (Çapraz Doğrulama)
            enIyiAlgoritma = "";
            enIyiDogruluk = 0;

            for (var entry : algoritmalar.entrySet()) {
                String isim = entry.getKey();
                Classifier model = entry.getValue();

                Classifier freshModel = model.getClass().getDeclaredConstructor().newInstance();
                Evaluation cvEval = new Evaluation(normalVeri);
                cvEval.crossValidateModel(freshModel, normalVeri, 10, new Random(42));

                double dogruluk = cvEval.pctCorrect();
                double kappa = cvEval.kappa();
                double f1 = cvEval.weightedFMeasure();

                if (dogruluk > enIyiDogruluk) {
                    enIyiDogruluk = dogruluk;
                    enIyiAlgoritma = isim;
                }
                System.out.printf("%-25s Acc=%.2f%%, Kappa=%.3f, F1=%.3f%n", isim, dogruluk, kappa, f1);
            }
            System.out.println("En Iyı Dogruluk: " + enIyiDogruluk);
            System.out.println("En iyi Algoritma" + enIyiAlgoritma);

        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
