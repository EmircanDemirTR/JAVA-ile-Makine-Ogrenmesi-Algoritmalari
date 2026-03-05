package com.btkakademi.ml.bolum06;

import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class WekaCokluLineerRegresyon {
    static void main(String[] args) {
        try {
            String dosyaYolu = "src/main/resources/datasets/boston-housing.arff";
            DataSource kaynak = new DataSource(dosyaYolu);
            Instances veriSeti = kaynak.getDataSet();

            veriSeti.setClassIndex(veriSeti.numAttributes() - 1);
            System.out.println("Veri Seti: " + veriSeti.numInstances());
            System.out.println("Ozellik Sayisi: " + (veriSeti.numAttributes() - 1));

            // Model Oluşturma ve Eğitme
            LinearRegression model = new LinearRegression();
            model.buildClassifier(veriSeti);

            // Katsayıları Al
            double[] katsayilar = model.coefficients();
            System.out.println("Model Katsayilari: ");

            for (int i = 0; i < veriSeti.numAttributes() - 1; i++) {
                String ozellik = veriSeti.attribute(i).name();
                double katsayi = katsayilar[i];
                System.out.printf(" %s: %.4f%n", ozellik, katsayi);
            }

            double b0 = katsayilar[katsayilar.length - 1];
            System.out.printf("Kesişim (b0): %.4f%n", b0);


            // Tahmin ve Test
            System.out.println("\nTahmin Örnekleri: ");
            for (int i = 0; i < 5; i++) {
                Instance ornek = veriSeti.instance(i);
                double gercek = ornek.classValue();
                double tahmin = model.classifyInstance(ornek);
                System.out.printf(" Örnek %d: Gerçek=%.1f, Tahmin=%.2f, Hata=%.2f%n", (i + 1), gercek, tahmin, (gercek - tahmin));
            }

            // WEKA MODEL OZETİ
            System.out.println("\n\nMODEL ÖZETİ\n\n");
            System.out.println(model.toString());

        } catch (Exception e) {
            System.err.println("Hata" + e.getMessage());
            e.printStackTrace();
        }
    }
}
