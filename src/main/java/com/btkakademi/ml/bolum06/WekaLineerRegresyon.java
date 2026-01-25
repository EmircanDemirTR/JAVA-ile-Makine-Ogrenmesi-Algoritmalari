package com.btkakademi.ml.bolum06;

import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * ============================================================
 * BÖLÜM 6.2: WEKA ile Basit Lineer Regresyon
 * ============================================================
 * Veri Seti: Öğrenci Performansı
 * Bağımsız Değişken (X): Haftalık çalışma saati
 * Bağımlı Değişken (Y): Sınav notu (0-100)
 * <p>
 * Hipotez: Daha fazla çalışma → Daha yüksek not
 * Model: sinav_notu = b0 + b1 * calisma_saati
 * ============================================================
 */
public class WekaLineerRegresyon {

    public static void main(String[] args) {
        try {
            // ============================================================
            // 1. ADIM: VERİ SETİNİ YÜKLE
            // ============================================================
            // ARFF dosyasını proje kaynaklarından yükle
            // DataSource: WEKA'nın veri yükleme sınıfı
            String dosyaYolu = "src/main/resources/datasets/ogrenci-performans.arff";
            DataSource kaynak = new DataSource(dosyaYolu);

            // Instances: WEKA'nın veri seti temsil sınıfı
            // getDataSet(): ARFF dosyasını belleğe yükler
            Instances veriSeti = kaynak.getDataSet();

            // ============================================================
            // 2. ADIM: HEDEF DEĞİŞKENİ BELİRLE
            // ============================================================
            // setClassIndex(): Tahmin edilecek (hedef) değişkeni belirler
            // numAttributes() - 1: Son sütun hedef değişken (sinav_notu)
            veriSeti.setClassIndex(veriSeti.numAttributes() - 1);

            System.out.println("=== WEKA - Basit Lineer Regresyon ===");
            System.out.println("Veri seti: " + veriSeti.numInstances() + " örnek");

            // ============================================================
            // 3. ADIM: MODEL OLUŞTUR VE EĞİT
            // ============================================================
            // LinearRegression: WEKA'nın lineer regresyon sınıfı
            // OLS (Ordinary Least Squares) yöntemini kullanır
            LinearRegression model = new LinearRegression();

            // buildClassifier(): Modeli eğitir
            model.buildClassifier(veriSeti);

            // ============================================================
            // 4. ADIM: KATSAYILARI AL
            // ============================================================
            // coefficients(): Regresyon katsayılarını döndürür
            // Format: [b1, ..., bn, b0] - Son eleman intercept
            double[] katsayilar = model.coefficients();
            double b1 = katsayilar[0];                      // Eğim (slope)
            double b0 = katsayilar[katsayilar.length - 1];  // Kesişim (intercept)

            System.out.println("\n--- Model Katsayıları ---");
            System.out.printf("Kesişim (b0): %.4f%n", b0);
            System.out.printf("Eğim (b1): %.4f%n", b1);
            System.out.printf("Denklem: sinav_notu = %.4f + %.4f * calisma_saati%n", b0, b1);

            // ============================================================
            // 5. ADIM: TAHMİN VE TEST
            // ============================================================
            System.out.println("\n--- Tahmin Örnekleri ---");
            double[] testSaatleri = {2.0, 5.0, 8.0, 10.0};
            for (double saat : testSaatleri) {
                double tahmin = b0 + b1 * saat;
                System.out.printf("Çalışma: %.1f saat -> Tahmini not: %.2f%n", saat, tahmin);
            }

            // ============================================================
            // 6. ADIM: MODEL PERFORMANSI
            // ============================================================
            double toplamKareHata = 0;
            for (int i = 0; i < veriSeti.numInstances(); i++) {
                Instance ornek = veriSeti.instance(i);
                double gercek = ornek.classValue();
                double tahmin = model.classifyInstance(ornek);
                toplamKareHata += Math.pow(gercek - tahmin, 2);
            }
            double rmse = Math.sqrt(toplamKareHata / veriSeti.numInstances());

            System.out.println("\n--- Model Performansı ---");
            System.out.printf("RMSE: %.4f%n", rmse);

            // WEKA model özeti
            System.out.println("\n--- WEKA Model Özeti ---");
            System.out.println(model.toString());

        } catch (Exception e) {
            System.err.println("Hata: " + e.getMessage());
            e.printStackTrace();
        }
    }
}