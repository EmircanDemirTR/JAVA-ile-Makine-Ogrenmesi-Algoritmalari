package com.btkakademi.ml.bolum06;

import weka.classifiers.functions.LinearRegression;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class WekaLineerRegresyon {
    static void main() {
        try {
            String dosyaYolu = "src/main/resources/datasets/ogrenci-performans.arff";
            DataSource kaynak = new DataSource(dosyaYolu);

            Instances veriSeti = kaynak.getDataSet();

            // Hedef Değişken tespiti
            veriSeti.setClassIndex(veriSeti.numAttributes() - 1);

            System.out.println("Veri seti: " + veriSeti.numInstances());

            // Model Oluşturma ve Eğitme
            // LinearRegression - OLS
            LinearRegression model = new LinearRegression();
            model.buildClassifier(veriSeti);

            // Katsayilari al

            double[] katsayilar = model.coefficients();
            double b1 = katsayilar[0];
            double b0 = katsayilar[katsayilar.length - 1];

            System.out.println("Model Katsayilari: ");
            System.out.printf("Kesişim (b0): %.4f%n", b0);
            System.out.printf("Eğim (b1): %.4f%n", b1);
            System.out.printf("Denklem: sinav_notu = %.4f + %.4f * calisma_saati%n", b0, b1);

            // Tahmin ve Test
            System.out.println("\nTahmin Örnekleri: ");
            double[] testSaatleri = {2.0, 5.0, 8.0, 10.0};
            for (double saat : testSaatleri) {
                double tahmin = b0 + b1 * saat;
                System.out.printf("Çalışma: %.1f saat - > Tahmini not: %.2f%n", saat, tahmin);
            }

        } catch (Exception e) {
            System.err.println("Hata: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
