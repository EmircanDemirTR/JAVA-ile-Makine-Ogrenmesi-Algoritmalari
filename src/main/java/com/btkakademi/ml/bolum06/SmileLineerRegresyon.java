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
 * BÖLÜM 6.3: SMILE ile Basit Lineer Regresyon
 * ============================================================
 * Veri Seti: Kariyer Maaş
 * Bağımsız Değişken (X): Deneyim yılı (0-15 yıl)
 * Bağımlı Değişken (Y): Aylık maaş ($K - bin dolar)
 * <p>
 * Hipotez: Daha fazla deneyim → Daha yüksek maaş
 * Model: maas = b0 + b1 * deneyim_yili
 * ============================================================
 * NOT: SMILE 5.x için OpenBLAS gerektirir.
 * JVM Argümanı: --enable-native-access=ALL-UNNAMED
 * ============================================================
 */
public class SmileLineerRegresyon {

    public static void main(String[] args) {
        try {
            // ============================================================
            // 1. ADIM: VERİ SETİNİ YÜKLE
            // ============================================================
            // CSV dosyasını oku
            String dosyaYolu = "src/main/resources/datasets/kariyer-maas.csv";

            // CSVFormat: Başlık satırını tanımla
            // setHeader(): İlk satırı başlık olarak kullan
            // setSkipHeaderRecord(): Başlığı veri olarak atlat
            CSVFormat format = CSVFormat.DEFAULT.builder()
                    .setHeader()
                    .setSkipHeaderRecord(true)
                    .setIgnoreHeaderCase(true)
                    .setTrim(true)
                    .build();

            // DataFrame olarak oku
            DataFrame veriSeti = Read.csv(Path.of(dosyaYolu), format);

            System.out.println("=== SMILE - Basit Lineer Regresyon ===");
            System.out.println("Veri seti: " + veriSeti.nrow() + " örnek");
            System.out.println("Sütunlar: " + veriSeti.names());

            // ============================================================
            // 2. ADIM: FORMÜL TANIMLA
            // ============================================================
            // Formula.lhs(): Hedef değişkeni belirle
            // "maas" tahmin edilecek, diğerleri bağımsız değişken
            Formula formula = Formula.lhs("maas");

            // ============================================================
            // 3. ADIM: MODEL OLUŞTUR VE EĞİT
            // ============================================================
            // OLS.fit(): Ordinary Least Squares regresyonu
            // QR Decomposition ile katsayıları hesaplar
            LinearModel model = OLS.fit(formula, veriSeti);

            // ============================================================
            // 4. ADIM: KATSAYILARI AL
            // ============================================================
            // intercept(): Kesişim (b0)
            // coefficients(): SMILE 5.x'te Vector döndürür
            double b0 = model.intercept();
            double b1 = model.coefficients().get(0);  // Vector.get(0)

            System.out.println("\n--- Model Katsayıları ---");
            System.out.printf("Kesişim (b0): %.4f%n", b0);
            System.out.printf("Eğim (b1): %.4f%n", b1);
            System.out.printf("Denklem: maas = %.4f + %.4f * deneyim_yili%n", b0, b1);

            // ============================================================
            // 5. ADIM: TAHMİN VE TEST
            // ============================================================
            System.out.println("\n--- Tahmin Örnekleri ---");
            double[] testDeneyimler = {0.0, 2.0, 5.0, 10.0, 15.0};
            for (double deneyim : testDeneyimler) {
                double tahmin = b0 + b1 * deneyim;
                System.out.printf("Deneyim: %.1f yıl -> Tahmini maaş: $%.2fK%n", deneyim, tahmin);
            }

            // ============================================================
            // 6. ADIM: MODEL PERFORMANSI
            // ============================================================
            double toplamKareHata = 0;
            for (int i = 0; i < veriSeti.nrow(); i++) {
                double gercek = veriSeti.getDouble(i, 1);
                double tahmin = model.predict(veriSeti.get(i));
                toplamKareHata += Math.pow(gercek - tahmin, 2);
            }
            double rmse = Math.sqrt(toplamKareHata / veriSeti.nrow());

            System.out.println("\n--- Model Performansı ---");
            System.out.printf("RMSE: %.4f%n", rmse);

            // SMILE model özeti (istatistiksel detaylar)
            System.out.println("\n--- SMILE Model Özeti ---");
            System.out.println(model);

        } catch (Exception e) {
            System.err.println("Hata: " + e.getMessage());
            e.printStackTrace();
        }
    }
}