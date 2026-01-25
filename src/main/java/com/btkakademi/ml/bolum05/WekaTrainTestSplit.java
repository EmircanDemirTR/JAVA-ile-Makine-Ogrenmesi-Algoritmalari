package com.btkakademi.ml.bolum05;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.util.Random;

public class WekaTrainTestSplit {
    static void main() throws Exception {
        // ARFF dosyasi yukleme
        var is = WekaTrainTestSplit.class.getClassLoader().getResourceAsStream("datasets/iris.arff");
        ArffLoader loader = new ArffLoader();
        loader.setSource(is);
        Instances veri = loader.getDataSet();

        // Son sütunu hedef olarak belirle
        veri.setClassIndex(veri.numAttributes() - 1);

        System.out.println("Toplam veri: " + veri.numInstances() + " ornek");

        // 2. Veriyi rastgele karıştır
        veri.randomize(new Random(42));

        // 3. Veriyi eğitim ve test olarak ayırma
        int egitimBoyut = (int) (veri.numInstances() * 0.7);

        Instances egitimVeri = new Instances(veri, 0, egitimBoyut);
        Instances testVeri = new Instances(veri, egitimBoyut, veri.numInstances() - egitimBoyut);

        System.out.println("Egitim seti: " + egitimVeri.numInstances());
        System.out.println("Test seti: " + testVeri.numInstances());

        // 4. J48 ile Sadece Egitim Veri Setiyle Egitme
        J48 model = new J48();
        model.buildClassifier(egitimVeri);

        // 5. Modeli Test Ettirelim
        Evaluation eval = new Evaluation(egitimVeri);
        eval.evaluateModel(model, testVeri);

        // 6. Sonuçları Ekrana Bastırma
        System.out.println("Degerlendirme Sonuclari");
        System.out.println("Dogruluk Orani: " + eval.pctCorrect());
        System.out.println("Hatali Siniflandirma: " + eval.pctIncorrect());
    }
}
