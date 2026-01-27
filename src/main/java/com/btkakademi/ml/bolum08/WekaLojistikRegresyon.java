package com.btkakademi.ml.bolum08;

import weka.classifiers.functions.Logistic;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.util.Random;

public class WekaLojistikRegresyon {
    static void main() {
        try {
            var is = WekaLojistikRegresyon.class.getClassLoader().getResourceAsStream("datasets/wine.arff");
            ArffLoader loader = new ArffLoader();
            loader.setSource(is);
            Instances veri = loader.getDataSet();

            veri.setClassIndex(veri.numAttributes() - 1);

            // Sınıf Dağılımı
            for (int i = 0; i < veri.numClasses(); i++) {
                String sinifAdi = veri.classAttribute().value(i);
                int sayi = 0;
                for (int j = 0; j < veri.numInstances(); j++) {
                    if (veri.instance(j).classValue() == i) sayi++;
                }
                System.out.print(sinifAdi + "=" + sayi + " ");
            }
            System.out.println();

            // Train Test Split
            veri.randomize(new Random(42));
            int egitimBoyut = (int) (veri.numInstances() * 0.7);

            Instances egitim = new Instances(veri, 0, egitimBoyut);
            Instances test = new Instances(veri, egitimBoyut, veri.numInstances() - egitimBoyut);

            System.out.println("Egitim: " + egitim.numInstances() + ", Test: " + test.numInstances());

            // Model Egitimi
            long baslangic = System.currentTimeMillis();
            Logistic lr = new Logistic();
            lr.buildClassifier(egitim);
            long sure = System.currentTimeMillis() - baslangic;
            System.out.println("Lojistik Regreson modeli egitildi. Sure: " + sure + "ms");
            // lr.setRidge(0.01) // Daha fazla regularization
            // lr.setMaxIts(-1) // İterasyon sayısı (-1, sınırsız)

            // Model Katsayıları
            System.out.println("Model Katsayıları: ");
            double[][] katsayilar = lr.coefficients();
            String[] ozellikAdlari = {"intercept", "alcohol", "malic_acid", "ash", "alcalinity", "magnesium",
                    "phenols", "flavanoids", "nonflavanoid", "proanthocyanins", "color_intensity", "hue", "od280", "proline"};

            System.out.println("Ozellik Katsıları (Sınıf 1 ve Sinif 2 icin)");
            System.out.println("Ozellik           Sinif1           Sinif2");
            System.out.println("-------            -------          -------");
            int limit = Math.min(6, katsayilar.length);
            for (int i = 0; i < limit; i++) {
                String ozellik = i < ozellikAdlari.length ? ozellikAdlari[i] : "ozellik" + i;
                System.out.printf("%-13s %+8.3f   %+8.3f%n",
                        ozellik, katsayilar[i][0], katsayilar[i][1]);
            }

            System.out.println(".... toplam " + katsayilar.length + " ozellik");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
