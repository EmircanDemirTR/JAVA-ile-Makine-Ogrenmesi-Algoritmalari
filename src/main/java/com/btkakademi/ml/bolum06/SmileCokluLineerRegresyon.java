package com.btkakademi.ml.bolum06;

import org.apache.commons.csv.CSVFormat;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.io.Read;
import smile.regression.LinearModel;
import smile.regression.OLS;

import java.nio.file.Path;

/**
 * ============================================================
 * BÖLÜM 6.5: SMILE ile Çoklu Lineer Regresyon
 * ============================================================
 * Veri Seti: Auto MPG
 * Bağımsız Değişkenler (X):
 *   - cylinders: Silindir sayısı
 *   - displacement: Motor hacmi (cubic inches)
 *   - horsepower: Beygir gücü
 *   - weight: Araç ağırlığı (lbs)
 *   - acceleration: 0-60 mph hızlanma süresi (saniye)
 * Bağımlı Değişken (Y): mpg (miles per gallon - yakıt verimliliği)
 * 
 * Model: mpg = b0 + b1*cylinders + b2*displacement + ... + b5*acceleration
 * ============================================================
 * NOT: SMILE 5.x için OpenBLAS gerektirir.
 * JVM Argümanı: --enable-native-access=ALL-UNNAMED
 * ============================================================
 */
public class SmileCokluLineerRegresyon {

    public static void main(String[] args) {
        try {
            // ============================================================
            // 1. ADIM: VERİ SETİNİ YÜKLE
            // ============================================================
            String dosyaYolu = "src/main/resources/datasets/auto-mpg.csv";
            
            // CSVFormat: Başlık satırını doğru şekilde oku
            // setHeader(): İlk satırı sütun isimleri olarak kullan
            // setSkipHeaderRecord(): Başlık satırını veri olarak atlat
            CSVFormat format = CSVFormat.DEFAULT.builder()
                    .setHeader()
                    .setSkipHeaderRecord(true)
                    .setIgnoreHeaderCase(true)
                    .setTrim(true)
                    .build();
            
            DataFrame veriSeti = Read.csv(Path.of(dosyaYolu), format);
            
            System.out.println("=== SMILE - Çoklu Lineer Regresyon ===");
            System.out.println("Veri seti: " + veriSeti.nrow() + " örnek");
            System.out.println("Özellik sayısı: " + (veriSeti.ncol() - 1));
            
            // Sütun isimlerini listele
            String[] sutunlar = veriSeti.names();
            System.out.println("\n--- Özellikler ---");
            for (int i = 0; i < sutunlar.length - 1; i++) {
                System.out.println("  " + (i + 1) + ". " + sutunlar[i]);
            }
            System.out.println("  Hedef: " + sutunlar[sutunlar.length - 1]);
            
            // ============================================================
            // 2. ADIM: FORMÜL TANIMLA
            // ============================================================
            // Formula.lhs(): Hedef değişkeni belirle
            // "mpg" tahmin edilecek, diğer tüm sütunlar bağımsız değişken
            Formula formula = Formula.lhs("mpg");
            
            // ============================================================
            // 3. ADIM: MODEL OLUŞTUR VE EĞİT
            // ============================================================
            // OLS.fit(): Ordinary Least Squares ile çoklu regresyon
            LinearModel model = OLS.fit(formula, veriSeti);
            
            // ============================================================
            // 4. ADIM: KATSAYILARI AL
            // ============================================================
            // intercept(): Kesişim noktası (b0)
            // coefficients(): SMILE 5.x'te Vector döndürür
            // Vector.get(i): i. katsayıyı al
            // Vector.size(): Katsayı sayısı
            double b0 = model.intercept();
            int katsayiSayisi = model.coefficients().size();
            
            System.out.println("\n--- Model Katsayıları ---");
            // Bağımsız değişken katsayıları
            for (int i = 0; i < katsayiSayisi; i++) {
                String ozellik = sutunlar[i];
                double katsayi = model.coefficients().get(i);  // Vector.get(i)
                System.out.printf("  %s: %.4f%n", ozellik, katsayi);
            }
            System.out.printf("  Intercept (b0): %.4f%n", b0);
            
            // ============================================================
            // 5. ADIM: TAHMİN ÖRNEKLERİ
            // ============================================================
            System.out.println("\n--- Tahmin Örnekleri (İlk 5 örnek) ---");
            int hedefIndex = veriSeti.ncol() - 1;  // Son sütun hedef (mpg)
            
            for (int i = 0; i < 5; i++) {
                double gercek = veriSeti.getDouble(i, hedefIndex);
                double tahmin = model.predict(veriSeti.get(i));
                System.out.printf("  Örnek %d: Gerçek=%.1f mpg, Tahmin=%.2f mpg, Hata=%.2f%n",
                        (i + 1), gercek, tahmin, (gercek - tahmin));
            }
            
            // ============================================================
            // 6. ADIM: MODEL PERFORMANSI
            // ============================================================
            double toplamKareHata = 0;
            double toplamMutlakHata = 0;
            
            for (int i = 0; i < veriSeti.nrow(); i++) {
                double gercek = veriSeti.getDouble(i, hedefIndex);
                double tahmin = model.predict(veriSeti.get(i));
                double hata = gercek - tahmin;
                
                toplamKareHata += hata * hata;
                toplamMutlakHata += Math.abs(hata);
            }
            
            int n = veriSeti.nrow();
            double rmse = Math.sqrt(toplamKareHata / n);
            double mae = toplamMutlakHata / n;
            
            System.out.println("\n--- Model Performansı ---");
            System.out.printf("  RMSE: %.4f (mpg)%n", rmse);
            System.out.printf("  MAE: %.4f (mpg)%n", mae);
            
            // ============================================================
            // 7. ADIM: SMILE MODEL ÖZETİ
            // ============================================================
            System.out.println("\n--- SMILE Model Özeti ---");
            System.out.println(model);
            
        } catch (Exception e) {
            System.err.println("Hata: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
