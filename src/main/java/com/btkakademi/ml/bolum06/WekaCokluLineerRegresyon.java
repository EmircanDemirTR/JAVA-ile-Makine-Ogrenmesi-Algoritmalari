package com.btkakademi.ml.bolum06;

import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * ============================================================
 * BÖLÜM 6.4: WEKA ile Çoklu Lineer Regresyon
 * ============================================================
 * Veri Seti: Boston Housing
 * Bağımsız Değişkenler (X):
 *   - RM: Konut başına ortalama oda sayısı
 *   - LSTAT: Düşük statülü nüfus yüzdesi
 *   - PTRATIO: Öğrenci-öğretmen oranı
 *   - CRIM: Kişi başı suç oranı
 * Bağımlı Değişken (Y): MEDV (Ev fiyatı $1000)
 * 
 * Model: MEDV = b0 + b1*RM + b2*LSTAT + b3*PTRATIO + b4*CRIM
 * ============================================================
 */
public class WekaCokluLineerRegresyon {

    public static void main(String[] args) {
        try {
            // ============================================================
            // 1. ADIM: VERİ SETİNİ YÜKLE
            // ============================================================
            // Boston Housing veri setini yükle
            String dosyaYolu = "src/main/resources/datasets/boston-housing.arff";
            DataSource kaynak = new DataSource(dosyaYolu);
            Instances veriSeti = kaynak.getDataSet();
            
            // ============================================================
            // 2. ADIM: HEDEF DEĞİŞKENİ BELİRLE
            // ============================================================
            // Son sütun (MEDV) hedef değişken
            veriSeti.setClassIndex(veriSeti.numAttributes() - 1);
            
            System.out.println("=== WEKA - Çoklu Lineer Regresyon ===");
            System.out.println("Veri seti: " + veriSeti.numInstances() + " örnek");
            System.out.println("Özellik sayısı: " + (veriSeti.numAttributes() - 1));
            
            // Özellik isimlerini listele
            System.out.println("\n--- Özellikler ---");
            for (int i = 0; i < veriSeti.numAttributes() - 1; i++) {
                System.out.println("  " + (i + 1) + ". " + veriSeti.attribute(i).name());
            }
            System.out.println("  Hedef: " + veriSeti.attribute(veriSeti.classIndex()).name());
            
            // ============================================================
            // 3. ADIM: MODEL OLUŞTUR VE EĞİT
            // ============================================================
            // LinearRegression: Çoklu lineer regresyon için de aynı sınıf
            LinearRegression model = new LinearRegression();
            
            // Modeli eğit
            model.buildClassifier(veriSeti);
            
            // ============================================================
            // 4. ADIM: KATSAYILARI AL
            // ============================================================
            // coefficients(): [b1, b2, b3, b4, 0 (hedef), b0 (intercept)]
            double[] katsayilar = model.coefficients();
            
            System.out.println("\n--- Model Katsayıları ---");
            // Bağımsız değişken katsayıları
            for (int i = 0; i < veriSeti.numAttributes() - 1; i++) {
                String ozellik = veriSeti.attribute(i).name();
                double katsayi = katsayilar[i];
                System.out.printf("  %s: %.4f%n", ozellik, katsayi);
            }
            // Intercept (son eleman)
            double b0 = katsayilar[katsayilar.length - 1];
            System.out.printf("  Intercept (b0): %.4f%n", b0);
            
            // ============================================================
            // 5. ADIM: TAHMİN ÖRNEKLERİ
            // ============================================================
            System.out.println("\n--- Tahmin Örnekleri (İlk 5 örnek) ---");
            for (int i = 0; i < 5; i++) {
                Instance ornek = veriSeti.instance(i);
                double gercek = ornek.classValue();
                double tahmin = model.classifyInstance(ornek);
                System.out.printf("  Örnek %d: Gerçek=$%.1fK, Tahmin=$%.2fK, Hata=%.2f%n",
                        (i + 1), gercek, tahmin, (gercek - tahmin));
            }
            
            // ============================================================
            // 6. ADIM: MODEL PERFORMANSI
            // ============================================================
            double toplamKareHata = 0;
            double toplamMutlakHata = 0;
            double toplamGercek = 0;
            
            for (int i = 0; i < veriSeti.numInstances(); i++) {
                Instance ornek = veriSeti.instance(i);
                double gercek = ornek.classValue();
                double tahmin = model.classifyInstance(ornek);
                double hata = gercek - tahmin;
                
                toplamKareHata += hata * hata;
                toplamMutlakHata += Math.abs(hata);
                toplamGercek += gercek;
            }
            
            int n = veriSeti.numInstances();
            double rmse = Math.sqrt(toplamKareHata / n);
            double mae = toplamMutlakHata / n;
            
            System.out.println("\n--- Model Performansı ---");
            System.out.printf("  RMSE: %.4f ($K)%n", rmse);
            System.out.printf("  MAE: %.4f ($K)%n", mae);
            
            // ============================================================
            // 7. ADIM: WEKA MODEL ÖZETİ
            // ============================================================
            System.out.println("\n--- WEKA Model Özeti ---");
            System.out.println(model.toString());
            
        } catch (Exception e) {
            System.err.println("Hata: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
