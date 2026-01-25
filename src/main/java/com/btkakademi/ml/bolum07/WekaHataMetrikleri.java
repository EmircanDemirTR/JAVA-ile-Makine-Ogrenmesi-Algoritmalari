package com.btkakademi.ml.bolum07;

import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class WekaHataMetrikleri {
    static void main(String[] args) {
        try {

            // 1. VERİ SETİNİ YÜKLEME

            // ARFF = Attribute-Relation File Format (WEKA'nın veri formatı)
            String dosyaYolu = "src/main/resources/datasets/winequality-red.arff";

            DataSource kaynak = new DataSource(dosyaYolu);
            Instances veriSeti = kaynak.getDataSet();
            veriSeti.setClassIndex(veriSeti.numAttributes() - 1);

            int n = veriSeti.numInstances();          // Örnek sayısı (satır)
            int p = veriSeti.numAttributes() - 1;     // Özellik sayısı (hedef hariç)

            // Bilgilendirme çıktısı
            System.out.println("=== WEKA - Regresyon Hata Metrikleri ===\n");
            System.out.println("Veri Seti: Wine Quality");
            System.out.println("Örnek Sayısı (n): " + n);
            System.out.println("Özellik Sayısı (p): " + p);
            System.out.println("Hedef Değişken: " + veriSeti.classAttribute().name());


            // 3. MODELİ OLUŞTURMA VE EĞİTME

            // LinearRegression: WEKA'nın lineer regresyon sınıfı
            // y = b0 + b1*x1 + b2*x2 + ... + bp*xp formülünü öğrenir
            LinearRegression model = new LinearRegression();

            // buildClassifier: Modeli veri setiyle eğit
            model.buildClassifier(veriSeti);

            System.out.println("\nModel eğitimi tamamlandı.");

            // 4. TAHMİN YAPMA VE HATALARI TOPLAMA

            double toplamMutlakHata = 0;   // |gerçek - tahmin| toplamı (MAE için)
            double toplamKareHata = 0;      // (gerçek - tahmin)² toplamı (MSE için)
            double toplamGercek = 0;        // Gerçek değerlerin toplamı (ortalama için)

            double[] gercekDegerler = new double[n];
            double[] tahminDegerler = new double[n];

            for (int i = 0; i < n; i++) {
                Instance ornek = veriSeti.instance(i);
                gercekDegerler[i] = ornek.classValue();
                tahminDegerler[i] = model.classifyInstance(ornek);

                double hata = gercekDegerler[i] - tahminDegerler[i];

                toplamMutlakHata += Math.abs(hata);
                toplamKareHata += hata * hata;
                toplamGercek += gercekDegerler[i];
            }

            // 5. METRİKLERİ HESAPLAMA

            double mae = toplamMutlakHata / n;
            double mse = toplamKareHata / n;
            double rmse = Math.sqrt(mse);

            double ortalama = toplamGercek / n;

            double ssTot = 0;

            for (double gercek : gercekDegerler) {
                ssTot += Math.pow(gercek - ortalama, 2);
            }

            double r2 = 1 - (toplamKareHata / ssTot);
            double adjR2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1));

            // 6. SONUÇLARI YAZDIRMA

            System.out.println("\n--- Hesaplanan Metrikler ---\n");
            System.out.printf("MAE  (Ortalama Mutlak Hata)  : %.4f%n", mae);
            System.out.printf("MSE  (Ortalama Kare Hata)    : %.4f%n", mse);
            System.out.printf("RMSE (Kök Ort. Kare Hata)    : %.4f%n", rmse);
            System.out.printf("R²   (Belirlilik Katsayısı)  : %.4f%n", r2);
            System.out.printf("Adjusted R²                  : %.4f%n", adjR2);

            // Metrik yorumları
            System.out.println("\n--- Metrik Yorumları ---\n");
            System.out.printf("MAE = %.4f : Ortalamada %.2f puan hata yapıyoruz%n", mae, mae);
            System.out.printf("RMSE = %.4f: Büyük hatalar dikkate alındığında hata ~%.2f puan%n", rmse, rmse);
            System.out.printf("RMSE > MAE : Bazı örneklerde büyük hatalar var (fark: %.4f)%n", rmse - mae);

            // 7. ÖRNEK TAHMİNLER

            System.out.println("\n--- Örnek Tahminler (İlk 10) ---\n");
            for (int i = 0; i < 10; i++) {
                double hata = gercekDegerler[i] - tahminDegerler[i];
                double yuzdeHata = Math.abs(hata / gercekDegerler[i]) * 100;
                System.out.printf("Örnek %2d: Gerçek=%1.0f, Tahmin=%.2f, Hata=%+.2f (%%%.1f)%n",
                        (i + 1), gercekDegerler[i], tahminDegerler[i], hata, yuzdeHata);
            }

            // 8. HATA DAĞILIMI ANALİZİ

            int kucukHata = 0;      // |hata| < 0.5
            int ortaHata = 0;       // 0.5 <= |hata| < 1.0
            int buyukHata = 0;      // |hata| >= 1.0
            double maxHata = 0;
            double minHata = Double.MAX_VALUE;

            for (int i = 0; i < n; i++) {
                double hata = Math.abs(gercekDegerler[i] - tahminDegerler[i]);

                if (hata < 0.5) kucukHata++;
                else if (hata < 1.0) ortaHata++;
                else buyukHata++;

                if (hata > maxHata) maxHata = hata;
                if (hata < minHata) minHata = hata;
            }

            System.out.println("\n--- Hata Dağılımı Analizi ---\n");
            System.out.printf("Küçük hata (< 0.5 puan)  : %d örnek (%%%.1f)%n",
                    kucukHata, (kucukHata * 100.0 / n));
            System.out.printf("Orta hata (0.5-1.0 puan) : %d örnek (%%%.1f)%n",
                    ortaHata, (ortaHata * 100.0 / n));
            System.out.printf("Büyük hata (>= 1.0 puan) : %d örnek (%%%.1f)%n",
                    buyukHata, (buyukHata * 100.0 / n));
            System.out.printf("En küçük hata: %.4f, En büyük hata: %.4f%n", minHata, maxHata);

            // 9. MODEL DEĞERLENDİRMESİ

            System.out.println("\n--- Model Değerlendirmesi ---\n");
            System.out.printf("Model varyansın %%%.1f'ini açıklıyor%n", r2 * 100);

            // R² bazlı performans değerlendirmesi
            // Bu eşikler genel kabul görmüş değerlerdir
            String performans;
            if (r2 >= 0.9) {
                performans = "Mükemmel";
            } else if (r2 >= 0.7) {
                performans = "İyi";
            } else if (r2 >= 0.5) {
                performans = "Orta";
            } else if (r2 >= 0.3) {
                performans = "Zayıf";
            } else {
                performans = "Çok Zayıf";
            }
            System.out.println("R² Bazlı Performans: " + performans);

            // Overfitting kontrolü
            // R² ve Adj R² arasındaki fark büyükse, model aşırı uyum sağlamış olabilir
            double r2Farki = r2 - adjR2;
            System.out.printf("R² - Adj R² Farkı: %.4f%n", r2Farki);
            if (r2Farki > 0.05) {
                System.out.println("Uyarı: Fark yüksek, overfitting riski olabilir");
            } else {
                System.out.println("Fark kabul edilebilir seviyede");
            }

            // 10. ÖNERİLER
            // ============================================================

            System.out.println("\n--- Model İyileştirme Önerileri ---\n");
            if (r2 < 0.5) {
                System.out.println("1. Daha fazla özellik eklenebilir (feature engineering)");
                System.out.println("2. Non-lineer modeller denenebilir (Karar Ağacı, Random Forest)");
                System.out.println("3. Polinom özellikler eklenebilir");
                System.out.println("4. Özellikler arasındaki etkileşimler incelenebilir");
            }

        } catch (Exception e) {
            System.err.println("Hata: " + e.getMessage());
            e.printStackTrace();
        }
    }
}